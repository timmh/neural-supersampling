import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

from utils import noop, warp, transform_rgb, transform_depth
from config import (
    history_length,
    data_root,
    device,
    source_resolution,
    target_resolution,
    color_file_extension,
    depth_file_extension,
    motion_file_extension,
    train_scenes,
    test_scenes,
    batch_size_train,
    batch_size_test,
    num_workers,
    train_random_crop_size,
)

class NeuralSupersamplingDataset(Dataset):
    def __init__(
        self,
        source_root_color,
        source_root_depth,
        source_root_motion,
        target_root_color,
        superscene,
        rendering_engine,
        source_resolution=source_resolution,
        target_resolution=target_resolution,
        color_file_extension=color_file_extension,
        depth_file_extension=depth_file_extension,
        motion_file_extension=motion_file_extension,
        history_length=history_length,
        device=torch.device("cpu"),  # it's complicated to use cuda in worker processes
        transform_rgb=transform_rgb,
        transform_depth=transform_depth,
        transform_motion=noop,
        random_crop_size=None,
    ):
        assert os.path.isdir(source_root_color) and os.path.isdir(source_root_depth) and os.path.isdir(source_root_motion)
        self.source_resolution = source_resolution
        self.target_resolution = target_resolution
        self.history_length = history_length
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_depth
        self.transform_motion = transform_motion
        self.device = device
        self.random_crop_size = random_crop_size

        self.source_files_depth =  sorted(glob.glob(os.path.join(source_root_depth, f"{superscene}*_{rendering_engine}_{source_resolution[0]}_{source_resolution[1]}_*.{depth_file_extension}")))
        self.source_files_motion = sorted(glob.glob(os.path.join(source_root_motion, f"{superscene}*_{rendering_engine}_{source_resolution[0]}_{source_resolution[1]}_*.{motion_file_extension}")))
        assert all([os.path.basename(a) == os.path.basename(b) for a, b in zip(self.source_files_depth, self.source_files_motion)])
        
        self.source_files_color = []
        self.target_files_color = []
        for source_file_depth in self.source_files_depth:
            subscene, rendering_engine, _, _, t = os.path.splitext(os.path.basename(source_file_depth))[0].split("_")
            source_file_color = os.path.join(source_root_color, f"{subscene}_{rendering_engine}_{source_resolution[0]}_{source_resolution[1]}_{t}.{color_file_extension}")
            target_file_color = os.path.join(target_root_color, f"{subscene}_{rendering_engine}_{target_resolution[0]}_{target_resolution[1]}_{t}.{color_file_extension}")
            assert os.path.exists(source_file_color)
            assert os.path.exists(target_file_color)
            self.source_files_color.append(source_file_color)
            self.target_files_color.append(target_file_color)

        assert len(self.source_files_depth) >= self.history_length

    def __len__(self):
        return len(self.source_files_depth) - self.history_length

    def __getitem__(self, idx):
        sample = {"source_rgb": [], "source_depth": []}

        source_motion = torch.zeros(self.history_length, 2, self.source_resolution[1], self.source_resolution[0], dtype=torch.float32, device=self.device)
        sample["source_rgb"] = [None] * self.history_length
        sample["source_depth"] = [None] * self.history_length

        x1 = np.random.randint(0, self.source_resolution[0] - self.random_crop_size[0]) if self.random_crop_size is not None else 0
        y1 = np.random.randint(0, self.source_resolution[1] - self.random_crop_size[1]) if self.random_crop_size is not None else 0
        x2 = (x1 + self.random_crop_size[0]) if self.random_crop_size is not None else self.source_resolution[0]
        y2 = (y1 + self.random_crop_size[1]) if self.random_crop_size is not None else self.source_resolution[1]

        prev_motion = None
        for i in range(self.history_length - 1, -1, -1):
            source_idx = idx + i

            sample["source_rgb"][i] = self.transform_rgb(cv2.imread(self.source_files_color[source_idx])[y1:y2, x1:x2, ::-1])
            sample["source_depth"][i] = self.transform_depth(cv2.imread(self.source_files_depth[source_idx], cv2.IMREAD_UNCHANGED)[y1:y2, x1:x2, None])
            motion = self.transform_motion(cv2.imread(self.source_files_motion[source_idx], cv2.IMREAD_UNCHANGED)[..., 0:2])

            motion[..., 1] = -motion[..., 1]
            motion = motion * -1
            motion = torch.from_numpy(motion.transpose(2, 0, 1)[None, ::-1, ...].copy()).to(self.device)

            # accumulate motion vectors
            if i == self.history_length - 1:
                source_motion[i] = 0
                prev_motion = motion
            else:
                source_motion[i] = prev_motion
                prev_motion = warp(motion, source_motion[i + 1:i + 2]) + source_motion[i + 1:i + 2]
            
        for k in sample:
            sample[k] = torch.from_numpy(np.array(sample[k]).transpose(0, 3, 1, 2).astype(np.float32))
        sample["source_motion"] = source_motion[:, :, y1:y2, x1:x2]
        upsampling_factor_horizontal = self.target_resolution[0] // self.source_resolution[0]
        upsampling_factor_vertical = self.target_resolution[1] // self.source_resolution[1]
        sample["target_rgb"] = torch.from_numpy(self.transform_rgb(cv2.imread(self.target_files_color[i])[y1 * upsampling_factor_vertical:y2 * upsampling_factor_vertical, x1 * upsampling_factor_horizontal:x2 * upsampling_factor_horizontal, ::-1]).transpose(2, 0, 1).copy().astype(np.float32))
        return sample


trainset = torch.utils.data.ConcatDataset([
    NeuralSupersamplingDataset(
        os.path.join(data_root, "color"),
        os.path.join(data_root, "depth"),
        os.path.join(data_root, "motion"),
        os.path.join(data_root, "color"),
        scene,
        "cycles",
        random_crop_size=train_random_crop_size,
    )
    for scene in train_scenes
])
trainset_len = int(0.9 * len(trainset))
trainset, valset = torch.utils.data.random_split(trainset, [trainset_len, len(trainset) - trainset_len], generator=torch.Generator().manual_seed(42))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)
testset = torch.utils.data.ConcatDataset([
    NeuralSupersamplingDataset(
        os.path.join(data_root, "color"),
        os.path.join(data_root, "depth"),
        os.path.join(data_root, "motion"),
        os.path.join(data_root, "color"),
        scene,
        "cycles",
    )
    for scene in test_scenes
])
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=num_workers)


def visualize_warping():
    trainset = NeuralSupersamplingDataset(
        os.path.join(data_root, "color"),
        os.path.join(data_root, "depth"),
        os.path.join(data_root, "motion"),
        os.path.join(data_root, "color"),
        "coffeerun",
        "cycles",
    )
    
    for sample in iter(trainset):
        imgs = []
        for i in range(len(sample["source_rgb"])):
            img = warp(sample["source_rgb"][i:i+1].to(sample["source_motion"].device).to(torch.float32), sample["source_motion"][i:i+1])
            imgs.append(img[0].to(torch.uint8).detach().cpu().numpy().transpose(1, 2, 0))
        
        cv2.imshow("Warped Images", np.hstack(imgs))
        cv2.waitKey(-1)