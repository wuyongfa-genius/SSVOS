from decord import VideoReader
from PIL import Image
import random
import numpy as np
import os

PATH_PALETTE = 'palette.txt'
default_palette = np.loadtxt(PATH_PALETTE, dtype=np.uint8).reshape(-1,3)


class VideoLoader:
    def __init__(self, n_frames=8, num_threads=1):
        self.n_frames = n_frames
        self.num_threads = num_threads
    
    def __call__(self, filename):
        with open(filename, 'rb') as f:
            vr = VideoReader(f, num_threads=self.num_threads)
            nframes = len(vr)
            chosen_frames = []
            nframes = nframes-nframes%self.n_frames
            for i in range(0, nframes, nframes//self.n_frames):
                interval = random.randint(0, nframes//self.n_frames-1)
                chosen_frame = i+interval
                frame = vr[chosen_frame].asnumpy()
                frame = Image.fromarray(frame)
                chosen_frames.append(frame)
        return chosen_frames


def imread_indexed(filename):
  """ Load image given filename."""
  im = Image.open(filename)
  annotation = np.atleast_3d(im)[...,0]
  return annotation,np.array(im.getpalette()).reshape((-1,3))

def imwrite_indexed(filename,array,color_palette=default_palette):
  """ Save indexed png."""

  if np.atleast_3d(array).shape[2] != 1:
    raise Exception("Saving indexed PNGs requires 2D array.")

  im = Image.fromarray(array)
  im.putpalette(color_palette.ravel())
  im.save(filename, format='PNG')


# if __name__=="__main__":
#     loader = VideoLoader(num_threads=4)
#     frames0 = loader()