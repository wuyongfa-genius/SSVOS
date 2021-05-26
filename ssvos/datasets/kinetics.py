"""A dataset class to load kinetics-400 datasets."""
from torch.utils.data import Dataset
from .utils import VideoLoader
import os
import random
from .transform import RandomResizedCrop, RandomHorizontalFlip, VideoToTensor, VideoNormalize
from torchvision.transforms import Compose


class Kinetics(Dataset):
    def __init__(self, root='/data/datasets/Kinetics-400/train_256', transforms=None, n_frames=8, video_loader_threads=4):
        super().__init__()
        self.root = root
        self.n_frames = n_frames
        self.video_loader = VideoLoader(n_frames, video_loader_threads)
        if transforms is None:
            self.transforms = Compose([
                RandomResizedCrop(size=(224, 224), scale=(0.6, 1.0)),
                RandomHorizontalFlip(),
                VideoToTensor(),
                VideoNormalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms
        ## load all intact video paths
        with open(os.path.join(root, 'intact_video_paths.txt'), 'r') as f:
            lines = f.readlines()
        self.all_video_paths = [i.rstrip('\n') for i in lines]
    
    def __getitem__(self, index: int):
        video_path = self.all_video_paths[index]
        # Note that here we use distant sampling discribed in "
        # Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective"
        frames = self.video_loader(video_path)
        random.shuffle(frames)
        online_frames = frames[:self.n_frames//2]
        target_frames = frames[self.n_frames//2:]
        return self.transforms(online_frames), self.transforms(target_frames)
    
    def __len__(self) -> int:
        return len(self.all_video_paths)
    