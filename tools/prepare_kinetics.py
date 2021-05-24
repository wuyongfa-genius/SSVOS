import os
from glob import glob
from decord import VideoReader
from tqdm import tqdm


def check_postfix(dataset_root):
    all_video_paths = glob(os.path.join(dataset_root, '*', '*'))
    print(f"Total video numbers:{len(all_video_paths)}")
    video_paths_ends_with_mp4 = [i for i in all_video_paths if i.endswith('mp4')]
    print(f"Mp4 video numbers:{len(video_paths_ends_with_mp4)}")

def find_intact_videos(dataset_root):
    all_video_paths = glob(os.path.join(dataset_root, '*', '*.mp4'))
    intact_video_paths = []
    for video_path in tqdm(all_video_paths):
        try:
            with open(video_path, 'rb') as f:
                vr = VideoReader(f)
        except:
                pass
        else:
            if len(vr)>=8:
                intact_video_paths.append(video_path)
            del vr
    print(f"Total video numbers:{len(all_video_paths)}")
    print(f"There are {len(intact_video_paths)} intact videos in total.")
    intact_video_paths_txt = os.path.join(dataset_root, 'intact_video_paths.txt')
    with open(intact_video_paths_txt, 'w') as f:
        f.write('\n'.join(intact_video_paths))
    print(f"Have written intact video paths into file {intact_video_paths_txt}.")


if __name__=="__main__":
    # check_postfix('/data/datasets/Kinetics-400/train_256')
    find_intact_videos('/data/datasets/Kinetics-400/train_256')