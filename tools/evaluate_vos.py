"""Evaluation script on VOS datasets. Support multi-gpu test."""
import argparse
import copy
import os
import queue

import numpy as np
import torch
from accelerate import Accelerator
from einops import rearrange
from PIL import Image
from spatial_correlation_sampler import spatial_correlation_sample
from ssvos.datasets.davis import DAVIS_VAL
from ssvos.datasets.utils import default_palette, imwrite_indexed
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
    parser.add_argument("--use_correlation_sampler", action='store_true',
                        help="Whether to use correlation sampler")
    parser.add_argument("--bs", type=int, default=1,
                        help="Batch size, try to reduce if OOM")
    return parser.parse_args()


def norm_mask(mask):
    c, h, w = mask.shape
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


@torch.no_grad()
def extract_seq_feats(model, frames, bs=1):
    seq_feats = []
    for i in range(0, len(frames), bs):
        if not i+bs > len(frames):
            this_batch = frames[i:i+bs]
        else:
            this_batch = frames[i:]
        if isinstance(this_batch, list):
            this_batch = torch.cat(this_batch)
        ##Define the way your model extract feature here#########################################
        this_batch_feats = model(this_batch)  # BCHW
        ############################################################################
        seq_feats.append(this_batch_feats)
    return torch.cat(seq_feats)  # NCHW

def extract_feat(model, frame):
    with torch.no_grad():
        feat = model(frame)
    return feat # BCHW

def _propagate_label(feat_tar, preceding_feats, preceding_segs, radius, device, topk=5, mask_neighborhood=None):
    def restrict_neighborhood(h, w, radius, device):
        # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
        mask = torch.zeros(h, w, h, w, device=device)
        for i in range(h):
            for j in range(w):
                for p in range(2 * radius + 1):
                    for q in range(2 * radius + 1):
                        if i - radius + p < 0 or i - radius + p >= h:
                            continue
                        if j - radius + q < 0 or j - radius + q >= w:
                            continue
                        mask[i, j, i - radius + p, j - radius + q] = 1
        mask = mask.reshape(h * w, h * w)
        return mask
    # compute affinity
    ncontext = len(preceding_feats)
    feat_sources = torch.cat(preceding_feats)  # NCHW
    # nmb_context x dim x h*w
    feat_sources = rearrange(feat_sources, 'n c h w -> n c (h w)')
    h, w = feat_tar.shape[-2:]
    feat_tar = rearrange(feat_tar, '1 c h w -> (h w) c')
    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)
    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    # nmb_context x h*w (tar: query) x h*w (source: keys)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)
    # mask neighborhood
    if radius > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(
                h, w, radius=radius, device=device)
            mask_neighborhood = mask_neighborhood.unsqueeze(
                0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood
    # prepare affinity
    # nmb_context*h*w (source: keys) x h*w (tar: queries)
    aff = rearrange(aff, 'n q k -> (n k) q')
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0
    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    # prepare segs
    segs = torch.cat(preceding_segs)  # NCHW
    segs = rearrange(segs, 'n c h w -> c (n h w)')
    seg_tar = torch.mm(segs, aff)
    seg_tar = rearrange(seg_tar, 'c (h w) -> 1 c h w', h=h, w=w)  # 1CHW
    return seg_tar, mask_neighborhood

def propagate_label(feat_tar, preceding_feats, preceding_segs, radius, device, topk=5, use_correlation_sampler=False, mask_neighborhood=None):
    if not use_correlation_sampler:
        seg_tar, mask_neighborhood = _propagate_label(
            feat_tar, preceding_feats, preceding_segs, radius, device, topk, mask_neighborhood)
        return seg_tar, mask_neighborhood
    else:
        # feat_sources = torch.stack(preceding_feats)  # NBCHW
        aff = []
        for feat_s in preceding_feats:
            feat_s = F.normalize(feat_s, p=2, dim=1)
            feat_tar = F.normalize(feat_tar, p=2, dim=1)
            corr = spatial_correlation_sample(
                feat_tar, feat_s, patch_size=2*radius+1)  # BPPHW
            aff.append(corr)
        # prepare affinity
        aff = torch.stack(aff)  # NBPPHW
        n, b, p, p, h, w = aff.shape
        aff = rearrange(aff, 'n b p1 p2 h w -> b (n p1 p2) h w')
        # aff = F.softmax(aff/0.1, dim=1)
        aff = torch.exp(aff/0.1)
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
        return seg_tar


def main():
    args = add_args()
    accelerator = Accelerator()
    # dataloader
    dataset = DAVIS_VAL(args.data_path, out_stride=args.out_stride)
    seq_names = dataset.seq_names
    dataloader = DataLoader(dataset, shuffle=False,
                            pin_memory=True, num_workers=1)
    # model
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
        # saving first segmentation
        first_feat = extract_feat(model, frames[0])
        out_path = os.path.join(seq_dir, "00000.png")
        imwrite_indexed(out_path, seg_ori[0].cpu().numpy(), default_palette)
        # The queue stores the n preceeding frames
        que = []
        for frame_index in range(1, len(frames)):
            # we use the first segmentation and the n previous ones
            used_frame_feats = [first_feat] + [pair[0]
                                                 for pair in que]
            used_segs = [first_seg] + [pair[1] for pair in que]
            # label propagation
            feat_tar = extract_feat(model, frames[frame_index])
            if not args.use_correlation_sampler:
                mask_neighborhood = None
                seg_tar, mask_neighborhood = propagate_label(feat_tar, used_frame_feats, used_segs, args.radius,
                                                             accelerator.device, args.topk, mask_neighborhood=mask_neighborhood)
            else:
                seg_tar = propagate_label(feat_tar, used_frame_feats, used_segs, args.radius,
                                          accelerator.device, args.topk, True)
            # pop out oldest frame if neccessary
            if len(que) == args.n_last_frames:
                del que[0]
            # push current results into queue
            seg = seg_tar.clone()
            que.append([feat_tar, seg])
            # upsampling & argmax
            seg_tar = F.interpolate(seg_tar, scale_factor=args.out_stride,
                                    mode='bilinear', align_corners=False, recompute_scale_factor=False)
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
