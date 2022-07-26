import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import NeuralSupersamplingModel
from config import (
    batch_size,
    history_length,
    upsampling_factor,
    device,
    source_resolution,
    target_resolution,
)


class WrappedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("alpha", torch.ones((1, 1, target_resolution[1], target_resolution[0]), dtype=torch.float32))
        self.register_buffer("rgb_max_value", torch.full((1,), 255, dtype=torch.float32))
        self.register_buffer("rgb_min_value", torch.full((1,), 0, dtype=torch.float32))
    
    def forward(self, source: torch.Tensor):
        source = source.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        source_rgb, source_depth, source_motion = source[:, :, 0:3, :, :], source[:, :, 3:4, :, :], source[:, :, 4:6, :, :]

        source_rgb = source_rgb / self.rgb_max_value

        # TODO: use .to(memory_format=torch.channels_last)
        # TODO: ensure depth has well-defined range

        reconstructed = self.model(source_rgb, source_depth, source_motion)
        reconstructed = torch.cat((reconstructed, self.alpha), dim=1)
        reconstructed = F.interpolate(reconstructed, scale_factor=1/4, mode="bilinear", align_corners=False)
        reconstructed = reconstructed[0]
        reconstructed = (reconstructed * self.rgb_max_value).permute(1, 2, 0).clip(self.rgb_min_value.item(), self.rgb_max_value.item())

        return reconstructed


def main():
    model = NeuralSupersamplingModel(upsampling_factor, batch_size, history_length, source_resolution, target_resolution)
    model.load_state_dict(torch.load(os.path.join("weights", "model_final.pt"), map_location=device))
    model = model.to(device)
    model.eval()
    wrapped_model = WrappedModel(model).to(device)

    source = torch.zeros((source_resolution[1], source_resolution[0], 6), dtype=torch.float32, device=device)

    traced_model = torch.jit.trace(wrapped_model, source)
    torch.jit.save(traced_model, os.path.join("..", "inference", "model_final_traced.pt"))


if __name__ == "__main__":
    main()