import os
import time
from argparse import ArgumentParser

import torch
from torch import distributed as dist
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import optim
from torch.nn import SyncBatchNorm
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from ssvos.models.byol import BYOL
from ssvos.datasets.kinetics import Kinetics
from ssvos.models.backbones.torchvision_resnet import resnet18
from ssvos.utils import collect_env, get_root_logger


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--root", default='/data/datasets/Kinetics-400/train_256')
    parser.add_argument("--samples_per_gpu", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base_lr", type=float, default=0.05)
    parser.add_argument("--use_fp16", action='store_true')
    parser.add_argument("--clip_grad_norm", type=float, default=1.)
    parser.add_argument("--log_dir", default='exps/vfs')
    parser.add_argument("--n_frames", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = arg_parser()
    # turn on benchmark mode
    torch.backends.cudnn.benchmark = True

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(fp16=args.use_fp16, kwargs_handlers=[ddp_kwargs])

    if accelerator.is_main_process:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, 'tf_logs'))
        time_stamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        logger = get_root_logger(log_file=os.path.join(
            args.log_dir, f'{time_stamp}.log'))
        # log env info
        logger.info('--------------------Env info--------------------')
        for key, value in sorted(collect_env().items()):
            logger.info(str(key) + ': ' + str(value))
        # log args
        logger.info('----------------------Args-----------------------')
        for key, value in sorted(vars(args).items()):
            logger.info(str(key) + ': ' + str(value))
        logger.info('---------------------------------------------------')

    train_dataset = Kinetics(root=args.root, n_frames=args.n_frames)
    train_dataloader = DataLoader(train_dataset, batch_size=args.samples_per_gpu,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ## define model
    model = BYOL(resnet18(zero_init_residual=True), feat_dim=512, projector_hidden_dim=[
        2048, 2048], out_dim=2048, predictor_hidden_dim=512, simsiam=True)
    # optimizer
    init_lr = args.base_lr*dist.get_world_size()*args.samples_per_gpu/256
    optimizer = optim.SGD(model.parameters(), lr=init_lr,
                          weight_decay=1e-4, momentum=0.9)
    ## recover states
    start_epoch = 1
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']+1
        if accelerator.is_main_process:
            logger.info(f"Resume from epoch {start_epoch-1}...")
    else:
        if accelerator.is_main_process:
            logger.info("Start training from scratch...")
    # convert BatchNorm to SyncBatchNorm
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    # prepare to be DDP models
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)
    # lr_scheduler
    total_steps = len(train_dataloader)*args.epochs
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    # training
    for e in range(start_epoch, args.epochs+1):
        model.train()
        for i, batch in enumerate(train_dataloader):
            frames1, frames2 = batch
            loss = model(frames1, frames2)
            accelerator.backward(loss)
            # clip grad if true
            if args.clip_grad_norm is not None:
                grad_norm = accelerator.clip_grad_norm_(
                    model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # sync before logging
            accelerator.wait_for_everyone()
            ## log and tensorboard
            if accelerator.is_main_process:
                if i % args.log_interval == 0:
                    writer.add_scalar('loss', loss.item(),
                                      (e-1)*len(train_dataloader)+i)
                    lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('lr', lr,
                                      (e-1)*len(train_dataloader)+i)
                    loss_str = f"loss: {loss.item():.4f}"
                    epoch_iter_str = f"Epoch: [{e}] [{i}/{len(train_dataloader)}], "
                    if args.clip_grad_norm is not None:
                        logger.info(
                            epoch_iter_str+f'lr: {lr}, '+loss_str+f', grad_norm: {grad_norm}')
                    else:
                        logger.info(epoch_iter_str+f'lr: {lr}, '+loss_str)

            lr_scheduler.step()
        if accelerator.is_main_process:
            if e % args.save_interval == 0:
                save_path = os.path.join(args.log_dir, f'epoch_{e}.pth')
                torch.save(
                    {'state_dict': model.module.state_dict(), 'epoch': e, 'args': args, 'optimizer': optimizer.state_dict()}, save_path)
                logger.info(f"Checkpoint has been saved at {save_path}")


if __name__ == "__main__":
    main()
