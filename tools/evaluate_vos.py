"""Evaluation script on VOS datasets. Support multi-gpu test."""
import argparse
import os

import numpy as np
import torch
from accelerate import Accelerator
from einops import rearrange
from PIL import Image
from spatial_correlation_sampler import spatial_correlation_sample
from ssvos.datasets.davis import DAVIS_VAL
from ssvos.datasets.utils import default_palette, imwrite_indexed, norm_mask
from ssvos.models.backbones.torchvision_resnet import resnet18, resnet_encoder
from ssvos.utils.load import load_encoder
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def add_args():
    parser = argparse.ArgumentParser(
        'Evaluation with video object segmentation on DAVIS 2017')
    parser.add_argument('--pretrained_weights', default='',
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['deit_tiny', 'deit_small', 'vit_base', 'resnet18'], help='Architecture (support only ViT atm).')
    parser.add_argument('--out_stride', default=8, type=int,
                        help='Output stride of the model.')
    parser.add_argument("--encoder_key", default="online_encoder.encoder",
                        type=str, help='Key to use in the checkpoint (example: "encoder")')
    parser.add_argument('--output_dir', default=".",
                        help='Path where to save segmentations')
    parser.add_argument(
        '--data_path', default='/data/datasets/DAVIS', type=str)
    parser.add_argument("--n_last_frames", type=int,
                        default=5, help="Number of preceeding frames")
    parser.add_argument("--radius", default=12, type=int,
                        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    parser.add_argument("--topk", type=int, default=5,
                        help="accumulate label from top k neighbors")
    parser.add_argument("--propagation_type", default='soft', choices=['soft', 'hard'],
                        help="Whether to quantize the predicted seg. `hard` means quantize.")
    return parser.parse_args()


def extract_feat(model, frame):
    """Extract one frame feature and L2-normalize it."""
    with torch.no_grad():
        ##Define the way your model extract feature here########################################
        feat = model(frame)  # BCHW
        ############################################################################
    feat = F.normalize(feat, p=2, dim=1)
    return feat  # BCHW


def propagate_label(feat_tar, preceding_feats, preceding_segs, radius, topk=5, quantize_seg=False):
    aff = []
    for feat_s in preceding_feats:
        corr = spatial_correlation_sample(
            feat_tar, feat_s, patch_size=2*radius+1)  # BPPHW
        aff.append(corr)
    # prepare affinity
    aff = torch.stack(aff)  # NBPPHW
    n, b, p, p, h, w = aff.shape
    aff = rearrange(aff, 'n b p1 p2 h w -> b (n p1 p2) h w')
    aff = torch.exp(aff/0.07)  # temperature
    tk_val, _ = torch.topk(aff, dim=1, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=1)
    aff[aff < tk_val_min] = 0
    aff = aff / torch.sum(aff, keepdim=True, axis=1)
    # prepare segs
    segs = torch.cat(preceding_segs)  # NCHW
    segs = F.unfold(segs, kernel_size=2*radius+1,
                    padding=radius)  # N(CKK)(HW)
    segs = rearrange(
        segs, 'n (c k1 k2) (h w) -> 1 c (n k1 k2) h w', k1=p, k2=p, h=h, w=w)
    seg_tar = (aff.unsqueeze(1)*segs).sum(2)  # 1CHW
    if quantize_seg:
        _seg = torch.argmax(seg_tar, dim=1).squeeze()  # HW
        quantized_seg = F.one_hot(_seg.long(), num_classes=seg_tar.shape[1])  # HWC
        return seg_tar, quantized_seg.permute(2, 0, 1).unsqueeze(0).float()  # 1CHW
    return seg_tar, torch.zeros_like(seg_tar)  # 1CHW


def main():
    args = add_args()
    accelerator = Accelerator()
    # dataloader
    dataset = DAVIS_VAL(args.data_path, out_stride=args.out_stride)
    seq_names = dataset.seq_names
    dataloader = DataLoader(dataset, shuffle=False,
                            pin_memory=True, num_workers=1)
    ## Build your own network here #########################################################
    if args.arch == 'resnet18':
        model = resnet_encoder(resnet18())
        ckpt = torch.load(args.pretrained_weights, map_location='cpu')
        load_encoder(model, ckpt['state_dict'],
                     pretrained_encoder_key=args.encoder_key, accelerator=accelerator)
        for param in model.parameters():
            param.requires_grad = False
    ################################################################################
    dataloader = accelerator.prepare(dataloader)
    model = model.to(accelerator.device)
    model.eval()
    accelerator.print('Start testing...')
    if accelerator.is_main_process:
        bar = tqdm(total=len(dataset))
    # start to test
    for (index, frames, ori_h, ori_w, first_seg, seg_ori) in dataloader:
        # make save dir
        seq_name = seq_names[index]
        seq_dir = os.path.join(args.output_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        # extract first frame feat and saving first segmentation
        first_feat = extract_feat(model, frames[0])
        out_path = os.path.join(seq_dir, "00000.png")
        imwrite_indexed(out_path, seg_ori[0].cpu().numpy(), default_palette)
        # The queue stores the n preceeding frames
        que = []
        for frame_index in range(1, len(frames)):
            # extract current frame feat
            feat_tar = extract_feat(model, frames[frame_index])
            # we use the first segmentation and the n previous ones
            used_frame_feats = [first_feat] + [pair[0]
                                               for pair in que]
            used_segs = [first_seg] + [pair[1] for pair in que]
            # label propagation
            quantize_seg = False if args.propagation_type == 'soft' else True
            seg_tar, quantized_seg = propagate_label(feat_tar, used_frame_feats, used_segs, args.radius,
                                                     args.topk, quantize_seg=quantize_seg)
            # pop out oldest frame if neccessary
            if len(que) == args.n_last_frames:
                del que[0]
            # push current results into queue
            seg = quantized_seg.clone() if quantize_seg else seg_tar.clone()
            que.append([feat_tar, seg])
            # upsampling & argmax
            seg_tar = F.interpolate(seg_tar, scale_factor=args.out_stride,
                                    mode='bicubic', align_corners=False, recompute_scale_factor=False)
            seg_tar = norm_mask(seg_tar[0])
            seg_tar = torch.argmax(seg_tar, dim=0)
            # saving to disk
            seg_tar = seg_tar.cpu().numpy().astype(np.uint8)
            seg_tar = np.array(Image.fromarray(
                seg_tar).resize((ori_w, ori_h), 0))
            seg_name = os.path.join(seq_dir, f'{frame_index:05}.png')
            imwrite_indexed(seg_name, seg_tar)
        if accelerator.is_main_process:
            bar.update(accelerator.num_processes)
    if accelerator.is_main_process:
        bar.close()
    accelerator.print(
        f'All videos has been evaludated, results saved at {args.data_path}.')


if __name__ == "__main__":
    main()
