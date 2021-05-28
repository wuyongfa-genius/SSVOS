import os
import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image


class DAVIS_VAL(Dataset):
    def __init__(self, root='/data/datasets/DAVIS', transforms=None, out_stride=8):
        super().__init__()
        self.root = root
        if transforms is None:
            self.transforms = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms
        self.out_stride = out_stride
        with open(os.path.join(root, 'ImageSets/2017/val.txt'), 'r') as f:
            lines = f.readlines()
        self.seq_names = [line.strip() for line in lines]
    
    def _load_frame(self, frame_path, scale_size=[480], return_h_w=False):
        """
        read a single frame & preprocess
        """
        img = Image.open(frame_path)
        ori_w, ori_h = img.size
        if len(scale_size) == 1:
            if(ori_h > ori_w):
                tw = scale_size[0]
                th = (tw * ori_h) / ori_w
                th = int((th // 64) * 64)
            else:
                th = scale_size[0]
                tw = (th * ori_w) / ori_h
                tw = int((tw // 64) * 64)
        else:
            th, tw = scale_size
        img = img.resize((tw, th))
        if return_h_w:
            return self.transforms(img), ori_h, ori_w
        else:
            return self.transforms(img)
    
    def _read_seg(self, seg_path, factor, scale_size=[480]):
        seg = Image.open(seg_path)
        _w, _h = seg.size # note PIL.Image.Image's size is (w, h)
        if len(scale_size) == 1:
            if(_w > _h):
                _th = scale_size[0]
                _tw = (_th * _w) / _h
                _tw = int((_tw // 64) * 64)
            else:
                _tw = scale_size[0]
                _th = (_tw * _h) / _w
                _th = int((_th // 64) * 64)
        else:
            _th = scale_size[1]
            _tw = scale_size[0]
        small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
        small_seg = F.one_hot(torch.from_numpy(small_seg).type(torch.LongTensor), num_classes=int(np.max(small_seg))+1) # HWC
        return small_seg.permute(2,0,1).float(), torch.from_numpy(np.array(seg))

    def __getitem__(self, index: int):
        seq_name = self.seq_names[index]
        seq_dir = os.path.join(self.root, "JPEGImages/480p/", seq_name)
        frame_names = sorted(os.listdir(seq_dir))
        ori_h, ori_w = 0, 0
        frames = []
        seg_path = ''
        for i in range(len(frame_names)):
            frame_path = os.path.join(seq_dir, frame_names[i])
            if i==0:
                frame, ori_h, ori_w = self._load_frame(frame_path, return_h_w=True)
                seg_path = frame_path.replace("JPEGImages", "Annotations").replace("jpg", "png")
            else:
                frame = self._load_frame(frame_path)
            frames.append(frame)
        ## read seg
        first_seg, seg_ori = self._read_seg(seg_path, self.out_stride)

        return index, frames, ori_h, ori_w, first_seg, seg_ori
    
    def __len__(self) -> int:
        return len(self.seq_names)


## test
# if __name__=="__main__":
#     davis = DAVIS_VAL()
#     idx, frames, h, w, small_seg, seg_ori = davis[0]
#     print(f'index: {idx}')
#     for frame in frames:
#         print(f'frame.shape: {frame.shape}')
#     print(f'h: {h}, w: {w}')
#     print(f'small_seg.shape: {small_seg.shape}')
#     print(f'seg_ori.shape: {seg_ori.shape}')