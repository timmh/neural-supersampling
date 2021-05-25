
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb


class ZeroUpsampling(nn.Module):
    def __init__(self, channels: int, upsampling_factor: int):
        super().__init__()
        self.channels = channels
        self.upsampling_factor = upsampling_factor
        kernel = torch.zeros((channels, 1, upsampling_factor, upsampling_factor), dtype=torch.float32, requires_grad=False)
        kernel[:, 0, 0, 0] = 1
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor):
        # TODO: switch to real implementation
        return F.upsample(x, scale_factor=self.upsampling_factor, mode="bilinear", align_corners=False)
        # return F.conv_transpose2d(x, self.kernel, stride=self.upsampling_factor, groups=self.channels)


def zero_upsample(image: torch.Tensor, upsampling_factor: int):
    assert upsampling_factor >= 1
    assert len(image.shape) == 4
    B, C, H, W = image.shape
    k = torch.zeros((C, 1, upsampling_factor, upsampling_factor), dtype=image.dtype, device=image.device)
    k[:, 0, 0, 0] = torch.ones((1, ), dtype=image.dtype, device=image.device)
    return F.conv_transpose2d(image, k, stride=upsampling_factor, groups=C)


def warp(image: torch.Tensor, flow: torch.Tensor):
    B, _, H, W = image.size()
    xx = torch.arange(0, W, dtype=image.dtype, device=image.device).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H, dtype=image.dtype, device=image.device).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx, yy),1)
    vgrid = grid + flow
    # TODO: is this right?
    # vgrid[:, 0, :, :] = 2 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1
    # vgrid[:, 1, :, :] = 2 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1
    return F.grid_sample(image, vgrid.permute(0, 2, 3, 1).to(image.dtype), mode="bilinear", align_corners=False)


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
    def __init__(self, history_length: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4 * history_length, 32, (3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, (3, 3), padding=(1, 1))  # TODO: according to Xiao et al. 2020, out_channels should be 4, but this is incompatible with the channel count of prev_features
        self.tanh3 = nn.Tanh()
        self.register_buffer("add", torch.FloatTensor([1]))
        self.register_buffer("multiply", torch.FloatTensor([10]))
        

    def forward(self, input, prev_features):
        x = self.relu1(self.conv1(input))
        x = self.relu2(self.conv2(x))
        x = self.tanh3(self.conv3(x))

        x = (x + self.add) * self.multiply  # scaling operation as described in Xiao et al. 2020

        return x * prev_features


class Reconstruction(nn.Module):
    def __init__(self, history_length: int):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d((history_length + 1) * 12, 64, (3, 3), padding=(1, 1)),
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
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # TODO: replace with transposed convolution as defined below?
            # nn.ConvTranspose2d(128, 128, (3, 3), stride=(2, 2), padding=(1, 1)),  # assume this is meant by Xiao et al. 2020 by the "Upsize" term
            # nn.ReplicationPad2d((0, 1, 0, 1)),  # TODO: normally, output_padding=(1, 1) could be used in nn.ConvTranspose2d above. This is however not yet supported by trtorch, see: https://github.com/NVIDIA/TRTorch/issues/290#issuecomment-767213863
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block8 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # TODO: replace with transposed convolution as defined below?
            # nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1)),  # assume this is meant by Xiao et al. 2020 by the "Upsize" term
            # nn.ReplicationPad2d((0, 1, 0, 1)),  # TODO: normally, output_padding=(1, 1) could be used in nn.ConvTranspose2d above. This is however not yet supported by trtorch, see: https://github.com/NVIDIA/TRTorch/issues/290#issuecomment-767213863
        )

        self.block9 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.block10 = nn.Sequential(
            nn.Conv2d(32, 3, (3, 3), padding=(1, 1)),
            nn.Sigmoid(),
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
    def __init__(self, upsampling_factor: int, batch_size: int, history_length: int, source_resolution: Tuple[int, int], target_resolution: Tuple[int, int]):
        super().__init__()
        assert upsampling_factor > 1
        self.upsampling_factor = upsampling_factor
        assert batch_size >= 1
        self.batch_size = batch_size
        assert history_length >= 1
        self.history_length = history_length

        self.source_resolution = source_resolution
        self.target_resolution = target_resolution

        self.feature_zero_upsampling = ZeroUpsampling(12, self.upsampling_factor)
        self.rgbd_zero_upsampling = ZeroUpsampling(4, self.upsampling_factor)
        self.feature_extraction = FeatureExtraction()
        self.feature_reweighting = FeatureReweighting(history_length)
        self.reconstruction = Reconstruction(history_length)

    def forward(self, rgb, depth, motion):
        B = self.batch_size
        I = self.history_length
        W, H = self.source_resolution

        rgb = rgb.reshape((B * I, 3, H, W))
        depth = depth.reshape((B * I, 1, H, W))
        motion = motion.reshape((B * I, 2, H, W))

        # TODO: remove
        depth = depth * torch.zeros((1,), dtype=depth.dtype, device=depth.device)
        motion = motion * torch.zeros((1,), dtype=motion.dtype, device=motion.device)

        # rgb = rgb_to_ycbcr(rgb)
        rgbd = torch.cat((rgb, depth), dim=1)

        features = self.feature_extraction(rgbd)

        features = self.feature_zero_upsampling(features)
        rgbd = self.rgbd_zero_upsampling(rgbd)

        # TODO: undo
        # rgbd_warped = warp(rgbd, F.interpolate(motion, scale_factor=self.upsampling_factor, mode="bilinear", align_corners=False))
        rgbd_warped = rgbd

        reweighted_features = self.feature_reweighting(
            rgbd_warped.view(B, I * 4, H * self.upsampling_factor, W * self.upsampling_factor),
            features.view(B, -1, H * self.upsampling_factor, W * self.upsampling_factor),
        )

        reconstructed = self.reconstruction(torch.cat((features[::I], reweighted_features), dim=1))
        # reconstructed = ycbcr_to_rgb(reconstructed)
        # reconstructed = reconstructed.clip(0, 1)
        
        return reconstructed