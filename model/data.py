import os
import glob
import numpy as np
import OpenEXR
import Imath
import torch
from torch.utils.data import Dataset

from utils import noop

class NeuralSupersamplingDataset(Dataset):
    def __init__(self, root, scene, rendering_engine, source_resolution, target_resolution, image_file_extension, history_length, transform_rgb=noop, transform_depth=noop, transform_motion=noop):
        assert os.path.isdir(root)
        self.source_resolution = source_resolution
        self.target_resolution = target_resolution
        self.history_length = history_length
        self.transform_rgb = transform_rgb
        self.transform_depth = transform_rgb
        self.transform_motion = transform_rgb
        self.source_files = sorted(glob.glob(os.path.join(root, f"{scene}_{rendering_engine}_{source_resolution[0]}_{source_resolution[1]}_*.{image_file_extension}")))
        self.target_files = sorted(glob.glob(os.path.join(root, f"{scene}_{rendering_engine}_{target_resolution[0]}_{target_resolution[1]}_*.{image_file_extension}")))
        self.num_image_pairs = min(len(self.source_files), len(self.target_files))
        assert self.num_image_pairs >= self.history_length

    def __len__(self):
        return self.num_image_pairs - self.history_length

    def __getitem__(self, idx):
        sample = {"source_rgb": [], "source_depth": [], "source_motion": []}
        for i in range(idx + self.history_length, idx, -1):
            source = OpenEXR.InputFile(self.source_files[i])
            target = OpenEXR.InputFile(self.target_files[i])
            sample["source_rgb"] += [self.transform_rgb(np.dstack([np.frombuffer(source.channel(f"RenderLayer.Combined.{c}", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(self.source_resolution[1], self.source_resolution[0]) for c in ["R", "G", "B"]]))]
            sample["source_depth"] += [self.transform_depth(np.dstack([np.frombuffer(source.channel(f"RenderLayer.Depth.Z", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(self.source_resolution[1], self.source_resolution[0])]))]
            sample["source_motion"] += [self.transform_motion(np.dstack([np.frombuffer(source.channel(f"RenderLayer.Vector.{c}", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(self.source_resolution[1], self.source_resolution[0]) for c in ["X", "Y"]]))]
        for k in sample:
            sample[k] = torch.from_numpy(np.array(sample[k]).transpose(0, 3, 1, 2))
        sample["target_rgb"] = torch.from_numpy(self.transform_rgb(np.dstack([np.frombuffer(target.channel(f"RenderLayer.Combined.{c}", Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32).reshape(self.target_resolution[1], self.target_resolution[0]) for c in ["R", "G", "B"]])).transpose(2, 0, 1))
        return sample