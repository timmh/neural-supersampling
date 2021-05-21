import os
import cv2
import torch

from config import (
    history_length,
    upsampling_factor,
    max_depth,
    device,
    inference_dtype,
    source_resolution,
    target_resolution,
)

traced_model = torch.jit.load(os.path.join("..", "inference", "model_final_traced.pt"))

source_rgb = cv2.imread(os.path.join(os.path.expanduser("~"), "Downloads", "Screenshot.png"))[..., 0:3][..., ::-1].transpose(1, 0, 2).copy()
source_rgb = torch.from_numpy(source_rgb).to(device)
source_depth = torch.zeros((source_resolution[1], source_resolution[0], 1), dtype=torch.float32, device=device)
source_motion = torch.zeros((source_resolution[1], source_resolution[0], 2), dtype=torch.float32, device=device)

predicted_rgb = traced_model(source_rgb, source_depth, source_motion)
cv2.imshow("Result", predicted_rgb.detach().cpu().numpy().transpose(1, 0, 2)[..., ::-1])
cv2.waitKey(0)