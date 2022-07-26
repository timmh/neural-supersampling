from argparse import ArgumentParser
import os
import glob
import av
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .train import NeuralSupersampling
from .data import testloader
from config import (
    learning_rate,
    history_length,
    upsampling_factor,
    ssim_window_size,
    n_epochs,
    enable_amp,
    perceptual_loss_weight,
    fps,
    device,
    weight_decay,
    tensorboard_root,
    source_resolution,
    target_resolution,
)
from utils import torch_to_numpy


def main(args):
    pl.seed_everything(42)
    checkpoint_path = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.ckpt")))[-1]  # load latest checkpoint
    neural_supersampling = NeuralSupersampling().load_from_checkpoint(checkpoint_path)
    neural_supersampling.to(device)
    neural_supersampling.eval()

    with av.open(args.output_path, mode="w") as container:
        
        length_acc = 0
        stream = None

        with torch.no_grad():  # dont need gradients in evaluation
            for batch in testloader:

                source = batch["source_rgb"][:, -1, ...]
                target = batch["target_rgb"]

                # upsample with nearest neighbour strategy for later visualization
                source_upsampled = F.interpolate(source, size=target.shape[2:4], mode="nearest")

                pred = neural_supersampling({k: v.to(device) for k, v in batch.items()})

                target = torch_to_numpy(target)
                source_upsampled = torch_to_numpy(source_upsampled)
                pred = torch_to_numpy(pred)

                for imgs in zip(source_upsampled, pred, target):
                    img = np.hstack(imgs)

                    if stream is None:
                        # setup stream with correct dimensions
                        stream = container.add_stream("h264", rate=fps)
                        stream.width = img.shape[1]
                        stream.height = img.shape[0]
                        stream.pix_fmt = "yuv420p"

                    # add frame to video
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

                    # early termination
                    length_acc += 1
                    if length_acc / fps >= args.length:
                        break
                if length_acc / fps >= args.length:
                    break
        
        # finalize stream
        for packet in stream.encode():
            container.mux(packet)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=os.getcwd(), help="where to load checkpoints from")
    parser.add_argument("--output_path", type=str, default="visualization.mp4", help="where to render output visualization to")
    parser.add_argument("--length", type=float, default=10., help="length of visualization in seconds")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)