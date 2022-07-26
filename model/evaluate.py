from argparse import ArgumentParser
import os
import torch
from kornia.losses import SSIMLoss
from kornia.metrics import ssim as compute_ssim
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
    weight_decay,
    tensorboard_root,
    source_resolution,
    target_resolution,
)


def main(args):
    pl.seed_everything(42)
    neural_supersampling = NeuralSupersampling()
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, every_n_epochs=1)
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        precision=16 if enable_amp else 32,
        callbacks=[checkpoint_callback],
    )
    trainer.test(neural_supersampling, testloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=os.getcwd(), help="where to load checkpoints from")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)