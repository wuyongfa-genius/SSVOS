from decord import VideoReader
from PIL import Image
import random
import torch
import numpy as np


PATH_PALETTE = 'ssvos/datasets/palette.txt'
default_palette = np.loadtxt(PATH_PALETTE, dtype=np.uint8).reshape(-1, 3)


class VideoLoader:
    def __init__(self, n_frames=8, num_threads=4):
        self.n_frames = n_frames
        self.num_threads = num_threads

    def __call__(self, filename):
        with open(filename, 'rb') as f:
            vr = VideoReader(f, num_threads=self.num_threads)
            nframes = len(vr)
            chosen_frame_indices = []
            nframes = nframes-nframes % self.n_frames
            for i in range(0, nframes, nframes//self.n_frames):
                interval = random.randint(0, nframes//self.n_frames-1)
                chosen_frame_indices.append(i+interval)
            chosen_frames = []
            for i in range(self.n_frames):
                chosen_frame = chosen_frame_indices[i]
                frame = vr[chosen_frame].asnumpy()
                frame = Image.fromarray(frame)
                chosen_frames.append(frame)
            del vr
        return chosen_frames


def imread_indexed(filename):
    """ Load image given filename."""
    im = Image.open(filename)
    annotation = np.atleast_3d(im)[..., 0]
    return annotation, np.array(im.getpalette()).reshape((-1, 3))


def imwrite_indexed(filename, array, color_palette=default_palette):
    """ Save indexed png."""

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    _,h,w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims or (int(torch.max(y_tensor)) + 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1) # CHW

def norm_mask(mask):
    c, h, w = mask.shape
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


# if __name__=="__main__":
#     loader = VideoLoader(num_threads=4)
#     frames0 = loader()
