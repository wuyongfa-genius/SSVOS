import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as TF
from torchvision.transforms import functional as F


class RandomResizedCrop(TF.RandomResizedCrop):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
    
    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        transformed_imgs = []
        for img in imgs:
            transformed_imgs.append(F.resized_crop(img, i, j, h, w, self.size, self.interpolation))
        return transformed_imgs


class RandomHorizontalFlip(TF.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)
    
    def __call__(self, imgs):
        if torch.rand(1) < self.p:
            transformed_imgs = []
            for img in imgs:
                transformed_imgs.append(F.hflip(img))
            return transformed_imgs
        return imgs


class RGB2LAB(object):
    def  __init__(self):
        super().__init__()
    
    def __call__(self, imgs):
        if isinstance(imgs[0], Image.Image):
            transformed_imgs = []
            for image in imgs:
                image = np.array(image)
                image = np.float32(image) / 255.0
                transformed_imgs.append(cv2.cvtColor(image, cv2.COLOR_RGB2Lab))
        else:
            raise NotImplementedError
        return transformed_imgs


class VideoToTensor(TF.ToTensor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, imgs):
        transformed_imgs = []
        for img in imgs:
            transformed_imgs.append(F.to_tensor(img)) 
        return transformed_imgs


class VideoNormalize(TF.Normalize): 
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean, std, inplace=inplace)
    
    def __call__(self, tensors):
        transformed_tensors = []
        for tensor in tensors:
            transformed_tensors.append(F.normalize(tensor, self.mean, self.std, self.inplace))
        return transformed_tensors
    