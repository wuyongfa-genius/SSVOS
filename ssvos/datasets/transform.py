"""Rewrite some torchvision augmentations so that they can directly apply to videos.
You can choose to apply the same augmentation to all the frames in a video(set `all_same` to True) or you can
apply different augmentations to different frames in a video(set `all_same` to False).
by wuyongfa. 2021.5.26
"""
import torch
import cv2
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms as TF
from torchvision.transforms import functional as F

__all__ = ['RandomResizedCrop', 'RandomHorizontalFlip', 'ColorJitter', 'RandomGrayscale',
           'GaussianBlur', 'Solarization', 'RGB2LAB', 'VideoToTensor', 'VideoNormalize']


class RandomResizedCrop(TF.RandomResizedCrop):
    def __init__(self, size, scale, ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, crop_num_per_img=1, all_same=False):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)
        self.all_same = all_same
        self.crop_num = crop_num_per_img

    def __call__(self, imgs):
        transformed_imgs = []
        if not self.all_same:
            for img in imgs:
                for _ in range(self.crop_num):
                    transformed_imgs.append(super().__call__(img))
        else:
            params = []
            for _ in range(self.crop_num):
                params.append(self.get_params())
            for img in imgs:
                for k in range(self.crop_num):
                    i, j, h, w = params[k]
                    transformed_imgs.append(F.resized_crop(
                        img, i, j, h, w, self.size, self.interpolation))
        return transformed_imgs


class RandomHorizontalFlip(TF.RandomHorizontalFlip):
    def __init__(self, p=0.5, all_same=False):
        super().__init__(p=p)
        self.all_same = all_same

    def __call__(self, imgs):
        transformed_imgs = []
        if not self.all_same:
            for img in imgs:
                transformed_imgs.append(super().forward(img))
        else:
            if torch.rand(1) < self.p:
                for img in imgs:
                    transformed_imgs.append(F.hflip(img))
            else:
                transformed_imgs = imgs
        return transformed_imgs


class ColorJitter(TF.ColorJitter):
    def __init__(self, brightness, contrast, saturation, hue, p=0.5, all_same=False):
        super().__init__(brightness=brightness,
                         contrast=contrast, saturation=saturation, hue=hue)
        self.p = p
        self.all_same = all_same

    def forward(self, imgs):
        transformed_imgs = []
        if not self.all_same:
            for img in imgs:
                if torch.rand(1) < self.p:
                    transformed_imgs.append(super().forward(img))
                else:
                    transformed_imgs.append(img)
        else:
            if torch.rand(1) < self.p:
                transform = super().get_params(
                    self.brightness, self.contrast, self.saturation, self.hue)
                for img in imgs:
                    transformed_imgs.append(transform(img))
            else:
                transformed_imgs = imgs
        return transformed_imgs


class RandomGrayscale(TF.RandomGrayscale):
    def __init__(self, p, all_same=False):
        super().__init__(p=p)
        self.all_same = all_same

    def __call__(self, imgs):
        transformed_imgs = []
        if not self.all_same:
            for img in imgs:
                transformed_imgs.append(super().__call__(img))
        else:
            if torch.rand(1) < self.p:
                num_output_channels = 1 if imgs[0].mode == 'L' else 3
                for img in imgs:
                    transformed_imgs.append(F.to_grayscale(
                        img, num_output_channels=num_output_channels))
            else:
                transformed_imgs = imgs
        return transformed_imgs


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2., all_same=False):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.all_same = all_same

    def get_params(self):
        return random.uniform(self.radius_min, self.radius_max)

    def __call__(self, imgs):
        transformed_imgs = []
        if not self.all_same:
            for img in imgs:
                if torch.rand(1) < self.p:
                    transformed_imgs.append(img.filter(
                        ImageFilter.GaussianBlur(radius=self.get_params())))
                else:
                    transformed_imgs.append(img)
        else:
            if torch.rand(1) < self.p:
                radius = self.get_params()
                for img in imgs:
                    transformed_imgs.append(img.filter(
                        ImageFilter.GaussianBlur(radius=radius)))
            else:
                transformed_imgs = imgs
        return transformed_imgs


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p, all_same=False):
        self.p = p
        self.all_same = all_same

    def __call__(self, imgs):
        transformed_imgs = []
        if not self.all_same:
            for img in imgs:
                if torch.rand(1) < self.p:
                    transformed_imgs.append(ImageOps.solarize(img))
                else:
                    transformed_imgs.append(img)
        else:
            if torch.rand(1) < self.p:
                for img in imgs:
                    transformed_imgs.append(ImageOps.solarize(img))
            else:
                transformed_imgs = imgs
        return transformed_imgs


class RGB2LAB(object):
    def __init__(self):
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
            transformed_tensors.append(F.normalize(
                tensor, self.mean, self.std, self.inplace))
        return transformed_tensors
