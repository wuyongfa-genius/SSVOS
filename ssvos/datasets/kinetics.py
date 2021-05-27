"""A dataset class to load kinetics-400 datasets."""
from torch.utils.data import Dataset
from ssvos.datasets.utils import VideoLoader
import os
import random
from ssvos.datasets.transform import (ColorJitter, GaussianBlur, RandomGrayscale,
                        RandomResizedCrop, RandomHorizontalFlip, Solarization, VideoToTensor, VideoNormalize)
from torchvision.transforms import Compose


class Local2GlobalVFSAugmentation(object):
    def __init__(self,
                 global_crop_scale,
                 local_crop_scale,
                 global_crop_size=224,
                 local_crop_size=96,
                 local_crop_num=2):
        self.local_crop_num = local_crop_num
        self.global_spatial_aug = Compose([
            RandomResizedCrop(size=global_crop_size, scale=global_crop_scale),
            RandomHorizontalFlip(),
        ])
        self.global_color_aug = Compose([
            ColorJitter(brightness=0.4, contrast=0.4,
                        saturation=0.2, hue=0.1, p=0.4, all_same=True),
            RandomGrayscale(p=0.1, all_same=True),
            GaussianBlur(p=0.1, all_same=True),
            Solarization(p=0.1, all_same=True),
        ])
        self.local_aug = Compose([
            RandomResizedCrop(size=local_crop_size, scale=local_crop_scale, crop_num_per_img=local_crop_num),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.4, contrast=0.4,
                        saturation=0.2, hue=0.1, p=0.8),
            GaussianBlur(),
        ])
        self.normalize = Compose([
            VideoToTensor(),
            VideoNormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.228, 0.224, 0.225])
            ])

    def __call__(self, imgs):
        ## global augs
        transformed_global_views = self.global_color_aug(self.global_spatial_aug(imgs))
        transformed_local_views = self.local_aug(imgs)
        return self.normalize(transformed_global_views), self.normalize(transformed_local_views)


class VFSAugmentation(object):
    def __init__(self, transforms=None):
        super().__init__()
        if transforms is None:
            self.transforms = Compose([
                RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
                RandomHorizontalFlip(),
                VideoToTensor(),
                VideoNormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.228, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms
    
    def __call__(self, imgs):
        length = len(imgs)
        online_frames = imgs[:length//2]
        target_frames = imgs[length//2:]
        return self.transforms(online_frames), self.transforms(target_frames)


class Kinetics(Dataset):
    def __init__(self,
                 root='/data/datasets/Kinetics-400/train_256',
                 transforms=None,
                 n_frames=8,
                 video_loader_threads=4):
        super().__init__()
        self.root = root
        self.n_frames = n_frames
        self.video_loader = VideoLoader(n_frames, video_loader_threads)
        if transforms is None:
            self.transforms = VFSAugmentation()
        else:
            self.transforms = transforms
        # load all intact video paths
        with open(os.path.join(root, 'intact_video_paths.txt'), 'r') as f:
            lines = f.readlines()
        self.all_video_paths = [i.rstrip('\n') for i in lines]

    def __getitem__(self, index: int):
        video_path = self.all_video_paths[index]
        # Note that here we use distant sampling discribed in "
        # Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective"
        frames = self.video_loader(video_path)
        random.shuffle(frames)
        return self.transforms(frames)

    def __len__(self) -> int:
        return len(self.all_video_paths)


##
if __name__=="__main__":
    transforms = Local2GlobalVFSAugmentation(global_crop_scale=(0.6, 1.0), local_crop_scale=(0.2, 0.6))
    dataset = Kinetics(transforms=transforms, n_frames=4)
    global_views, local_views = dataset[0]
    for i in global_views:
        print(i.shape)
    print("----------------------------------------------")
    for j in local_views:
        print(j.shape)
