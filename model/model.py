import unittest
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb

from utils import warp


class ZeroUpsampling(nn.Module):
    def __init__(self, channels: int, upsampling_factor: Tuple[int]):
        super().__init__()
        self.channels = channels
        self.upsampling_factor = upsampling_factor
        kernel = torch.zeros((channels, 1, upsampling_factor[1], upsampling_factor[0]), dtype=torch.float32, requires_grad=False)
        kernel[:, 0, 0, 0] = 1
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor):
        return F.conv_transpose2d(x, self.kernel, stride=self.upsampling_factor, groups=self.channels)


class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 8, (3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.relu1(self.conv1(input))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return torch.cat((input, x), dim=1)


class FeatureReweighting(nn.Module):
    def __init__(self, history_length: int, summand=1., factor=10.):
        super().__init__()
        self.conv1 = nn.Conv2d(history_length * 4, 32, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 4, (3, 3), padding=(1, 1))
        self.tanh3 = nn.Tanh()
        self.register_buffer("summand", torch.Tensor([summand]))
        self.register_buffer("factor", torch.Tensor([factor]))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.tanh3(self.conv3(x))

        x = (x + self.summand) * self.factor  # scaling operation as described in Xiao et al. 2020

        x = torch.cat((x, torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)), dim=1)

        return x


class Reconstruction(nn.Module):
    def __init__(self, history_length: int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(history_length * 12, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d((2, 2))  # assume max pooling, exact pooling not described in Xiao et al. 2020

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d((2, 2))  # assume max pooling, exact pooling not described in Xiao et al. 2020

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1)),  # assume this is meant by Xiao et al. 2020 by the "Upsize" term
            nn.ReplicationPad2d((0, 1, 0, 1)),  # TODO: normally, output_padding=(1, 1) could be used in nn.ConvTranspose2d above. This is however not yet supported by trtorch, see: https://github.com/NVIDIA/TRTorch/issues/290#issuecomment-767213863
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block8 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1)),  # assume this is meant by Xiao et al. 2020 by the "Upsize" term
            nn.ReplicationPad2d((0, 1, 0, 1)),  # TODO: normally, output_padding=(1, 1) could be used in nn.ConvTranspose2d above. This is however not yet supported by trtorch, see: https://github.com/NVIDIA/TRTorch/issues/290#issuecomment-767213863
        )

        self.block9 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block10 = nn.Sequential(
            nn.Conv2d(32, 3, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        x1 = self.block1(input)
        x2 = self.block2(x1)
        x3 = self.block3(self.pool2(x2))
        x4 = self.block4(x3)
        x5 = self.block5(self.pool4(x4))
        x6 = self.block6(x5)
        x7 = self.block7(torch.cat((x4, x6), dim=1))
        x8 = self.block8(x7)
        x9 = self.block9(torch.cat((x2, x8), dim=1))
        x10 = self.block10(x9)
        return x10


class NeuralSupersamplingModel(nn.Module):
    def __init__(self, history_length: int, source_resolution: Tuple[int, int], target_resolution: Tuple[int, int]):
        super().__init__()
        assert history_length >= 1
        self.history_length = history_length

        self.source_resolution = source_resolution
        self.target_resolution = target_resolution
        self.upsampling_factor = (
            self.target_resolution[0] // self.source_resolution[0],
            self.target_resolution[1] // self.source_resolution[1],
        )
        self.register_buffer("upsampling_factor_tensor", torch.Tensor([
            self.upsampling_factor[1],
            self.upsampling_factor[0],
        ]).reshape((1, 2, 1, 1)))

        self.feature_zero_upsampling = ZeroUpsampling(12, self.upsampling_factor)
        self.rgbd_zero_upsampling = ZeroUpsampling(4, self.upsampling_factor)
        self.feature_extraction = FeatureExtraction()
        self.feature_reweighting = FeatureReweighting(history_length)
        self.reconstruction = Reconstruction(history_length)

    def forward(self, rgb, depth, accumulated_motion):
        B, I, _, H, W = rgb.shape
        assert I == self.history_length
        W_t = W * self.upsampling_factor[0]
        H_t = H * self.upsampling_factor[1]

        rgb = rgb.reshape((B * I, 3, H, W))
        depth = depth.reshape((B * I, 1, H, W))
        accumulated_motion = accumulated_motion.reshape((B * I, 2, H, W))

        rgb = rgb_to_ycbcr(rgb)
        rgbd = torch.cat((rgb, depth), dim=1)

        features = self.feature_extraction(rgbd)
        features = self.feature_zero_upsampling(features)
        accumulated_motion = F.interpolate(
            accumulated_motion,
            scale_factor=(self.upsampling_factor[1], self.upsampling_factor[0]),
            mode="bilinear",
            align_corners=False,
        )
        accumulated_motion = accumulated_motion * self.upsampling_factor_tensor
        features = warp(features, accumulated_motion)

        rgbd_zero_upsampled = self.rgbd_zero_upsampling(rgbd)
        rgbd_zero_upsampled = rgbd_zero_upsampled.reshape((B, I * 4, H_t, W_t))
        reweighting_map = self.feature_reweighting(rgbd_zero_upsampled)
        reweighting_map = reweighting_map.reshape((B * I, 1, H_t, W_t))
        features = features * reweighting_map

        reconstructed = self.reconstruction(features.reshape((B, I * 12, H_t, W_t)))
        reconstructed = ycbcr_to_rgb(reconstructed)
        
        return reconstructed


class TestModel(unittest.TestCase):
    def test_zero_upsampling(self):

        # initialize zero upsampling
        zu = ZeroUpsampling(3, (2, 2))

        # create some image
        x = torch.Tensor([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ])

        # add batch dimension
        x = torch.unsqueeze(x, 0)

        # apply zero upsampling
        y = zu(x)

        # define target
        y_target = torch.Tensor([
            [
                [1, 0, 2, 0, 3, 0],
                [0, 0, 0, 0, 0, 0],
                [4, 0, 5, 0, 6, 0],
                [0, 0, 0, 0, 0, 0],
                [7, 0, 8, 0, 9, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
        ])

        self.assertTrue(torch.allclose(y, y_target))


    def test_warp(self):
        # create some image
        x = torch.Tensor([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])

        # add batch and channel dimensions
        x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)

        # create some flow
        flow = torch.Tensor([
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ])

        # add batch dimension
        flow = torch.unsqueeze(flow, 0)

        # warp
        y = warp(x, flow)

        # define target
        y_target = torch.Tensor([
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ])

        self.assertTrue(torch.allclose(y, y_target))


if __name__ == '__main__':
    unittest.main()