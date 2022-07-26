from argparse import ArgumentParser
import os
import torch
from kornia.losses import SSIMLoss
from kornia.metrics import ssim as compute_ssim
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .model import NeuralSupersamplingModel
from .data import trainloader, valloader
from .perceptual_loss import PerceptualLoss
from config import (
    learning_rate,
    history_length,
    ssim_window_size,
    n_epochs,
    enable_amp,
    perceptual_loss_weight,
    weight_decay,
    tensorboard_root,
    source_resolution,
    target_resolution,
)


class NeuralSupersampling(pl.LightningModule):
    def __init__(self, learning_rate=learning_rate, weight_decay=weight_decay):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = NeuralSupersamplingModel(history_length, source_resolution, target_resolution)
        self.structural_loss = SSIMLoss(ssim_window_size)
        self.perceptual_loss_weight = perceptual_loss_weight
        if perceptual_loss_weight != 0:
            self.perceptual_loss_container = dict(loss=PerceptualLoss().to(self.device))
        self.save_hyperparameters()

    def _compute_perceptual_loss(self, pred, gt):
        if self.perceptual_loss_weight == 0:
            return 0
        else:
            if next(self.perceptual_loss_container["loss"].parameters()).device != pred.device:
                self.perceptual_loss_container["loss"].backbone.to(pred.device)
            return self.perceptual_loss_container["loss"](pred, gt)

    def forward(self, batch):
        return self.model(batch["source_rgb"], batch["source_depth"], batch["source_motion"])

    def training_step(self, batch, batch_idx):
        pred = self.model(batch["source_rgb"], batch["source_depth"], batch["source_motion"])
        gt = batch["target_rgb"]
        sl = self.structural_loss(pred, gt)
        pl = self._compute_perceptual_loss(pred, gt)
        loss = sl + perceptual_loss_weight * pl
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch["source_rgb"], batch["source_depth"], batch["source_motion"])
        gt = batch["target_rgb"]
        sl = self.structural_loss(pred, gt)
        pl = self._compute_perceptual_loss(pred, gt)
        loss = sl + perceptual_loss_weight * pl
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        ssim = torch.mean(compute_ssim(pred, gt, ssim_window_size))
        self.log("val_ssim", ssim, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        pred = self.model(batch["source_rgb"], batch["source_depth"], batch["source_motion"])
        gt = batch["target_rgb"]
        sl = self.structural_loss(pred, gt)
        pl = self._compute_perceptual_loss(pred, gt)
        loss = sl + perceptual_loss_weight * pl
        self.log("test_loss", loss)
        ssim = torch.mean(compute_ssim(pred, gt, ssim_window_size))
        self.log("test_ssim", ssim)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


def main(args):
    pl.seed_everything(42)
    neural_supersampling = NeuralSupersampling()
    os.makedirs(tensorboard_root, exist_ok=True)
    logger = TensorBoardLogger(save_dir=tensorboard_root)
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, every_n_epochs=1)
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        max_epochs=n_epochs,
        logger=logger,
        precision=16 if enable_amp else 32,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(neural_supersampling, trainloader, valloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=os.getcwd(), help="where to store checkpoints")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)