import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import trtorch

from model import NeuralSupersamplingModel
from config import (
    history_length,
    upsampling_factor,
    max_depth,
    device,
    inference_dtype,
    source_resolution,
    target_resolution,
)


class WrappedModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    
    def forward(self, source_rgb: torch.Tensor, source_depth: torch.Tensor, source_motion: torch.Tensor):
        source_rgb = source_rgb[..., 0:3].flip(0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).type(inference_dtype) / 255
        source_depth = source_depth.flip(0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).clip(0, max_depth) / max_depth
        source_motion = source_motion.flip(0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        reconstructed = self.model(source_rgb, source_depth, source_motion)
        reconstructed = F.interpolate(reconstructed, scale_factor=1/4, mode="bilinear", align_corners=False)
        reconstructed = (reconstructed[0] * 255).permute(1, 2, 0).clip(0, 255).type(torch.uint8)
        alpha = torch.full((reconstructed.shape[0], reconstructed.shape[1], 1), 255, dtype=reconstructed.dtype, device=reconstructed.device)
        reconstructed = torch.cat((reconstructed, alpha), dim=2)
        reconstructed = reconstructed.flip(0)

        return reconstructed


model = NeuralSupersamplingModel(upsampling_factor, history_length, source_resolution, target_resolution)
model.load_state_dict(torch.load(os.path.join("weights", "model_final.pt")))
model = model.to(device)
model.eval()
wrapped_model = WrappedModel(model)

source_rgb = torch.zeros((source_resolution[1], source_resolution[0], 4), dtype=torch.uint8, device=device)
source_depth = torch.zeros((source_resolution[1], source_resolution[0], 1), dtype=torch.float32, device=device)
source_motion = torch.zeros((source_resolution[1], source_resolution[0], 2), dtype=torch.float32, device=device)

traced_model = torch.jit.trace(wrapped_model, (source_rgb, source_depth, source_motion))

# trt_ts_module = trtorch.compile(traced_model, {
#     "input_shapes": [
#         [480, 270, 3],
#         [480, 270, 1],
#         [480, 270, 2],
#     ],
#     "op_precision": torch.half,
# })

# predicted_rgb = traced_model(source_rgb, source_depth, source_motion)
# cv2.imshow("Result", (predicted_rgb[0].detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255).clip(0, 255).astype(np.uint8))
# cv2.waitKey(0)

torch.jit.save(traced_model, os.path.join("..", "inference", "model_final_traced.pt"))
# torch.jit.save(trt_ts_module, os.path.join("..", "inference", "model_final_traced.ts"))