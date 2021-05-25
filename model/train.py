import os
import numpy as np
import cv2
import torch
from kornia.losses import SSIMLoss
from torch.utils.data import DataLoader

from model import NeuralSupersamplingModel
from data import NeuralSupersamplingDataset
from perceptual_loss import PerceptualLoss
from utils import noop_context
from config import (
    history_length,
    upsampling_factor,
    batch_size,
    ssim_window_size,
    n_epochs,
    log_interval,
    save_interval,
    enable_amp,
    enable_anomaly_detection,
    perceptual_loss_weight,
    num_workers,
    weight_decay,
    max_depth,
    data_root,
    device,
    source_resolution,
    target_resolution,
)


trainset = NeuralSupersamplingDataset(data_root, "dweebs", "cycles", source_resolution, target_resolution, "exr", history_length, transform_depth=lambda depth: np.clip(depth, 0, max_depth) / max_depth)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = NeuralSupersamplingDataset(data_root, "dweebs", "cycles", source_resolution, target_resolution, "exr", history_length, transform_depth=lambda depth: np.clip(depth, 0, max_depth) / max_depth)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = NeuralSupersamplingModel(upsampling_factor, batch_size, history_length, source_resolution, target_resolution)
model = model.to(device)

structural_loss = SSIMLoss(ssim_window_size, reduction="none")
if perceptual_loss_weight != 0:
    perceptual_loss = PerceptualLoss(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler()


for epoch in range(n_epochs):
    running_loss = 0
    running_loss_items = 0

    for i, data in enumerate(trainloader, 0):
        with torch.autograd.detect_anomaly() if enable_anomaly_detection else noop_context():
            model.train()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            with torch.cuda.amp.autocast(enabled=enable_amp):
                source_rgb, source_depth, source_motion, target_rgb = data["source_rgb"].to(device), data["source_depth"].to(device), data["source_motion"].to(device), data["target_rgb"].to(device)
                
                # TODO flip inpurt along height dimension
                predicted_rgb = model(source_rgb.flip(-2).to(device), source_depth.flip(-2).to(device), source_motion.flip(-2).to(device)).flip(-2)

                # TODO: without the NaN-masking and amp enabled, the loss becomes NaN
                sl = structural_loss(predicted_rgb, target_rgb)
                sl = torch.mean(sl[~sl.isnan()])

                pl =  perceptual_loss(predicted_rgb, target_rgb) if perceptual_loss_weight != 0 else 0

                loss = sl + perceptual_loss_weight * pl
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cv2.imshow("Result", (predicted_rgb[0].detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1] * 255).clip(0, 255).astype(np.uint8))
            cv2.waitKey(1)

            running_loss += loss.item()
            running_loss_items += 1

    if epoch % log_interval == log_interval - 1:
        print(f"[{epoch + 1:05d}] loss: {running_loss / running_loss_items:.3f}")
        running_loss = 0
        running_loss_items = 0

    if epoch % save_interval == save_interval - 1:
        torch.save(model.state_dict(), os.path.join("weights", f"model_{epoch + 1:05d}.pt"))


torch.save(model.state_dict(), os.path.join("weights", f"model_final.pt"))
print("Finished Training")