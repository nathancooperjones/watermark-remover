from typing import Any, Dict

import lightning as pl
import torch
import torchvision
from typing_extensions import Self


class DoubleConv(torch.nn.Module):
    """Double convolution block with batch normalization and ReLU."""
    def __init__(self: Self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the double convolution block."""
        return self.double_conv(x)


class WatermarkRemovalModel(pl.LightningModule):
    """
    PyTorch Lightning module for watermark removal using a standard UNet architecture.

    Parameters
    ----------
    learning_rate: float
        Learning rate for training

    """
    def __init__(
        self: Self,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # encoder blocks
        self.enc_1 = DoubleConv(in_channels=3, out_channels=64)
        self.enc_2 = DoubleConv(in_channels=64, out_channels=128)
        self.enc_3 = DoubleConv(in_channels=128, out_channels=256)
        self.enc_4 = DoubleConv(in_channels=256, out_channels=512)
        self.enc_5 = DoubleConv(in_channels=512, out_channels=1024)

        # decoder blocks with updated input channels for concatenation
        self.dec_5 = DoubleConv(in_channels=1024, out_channels=512)
        # 512 from dec5 + 512 from enc4
        self.dec_4 = DoubleConv(in_channels=512+512, out_channels=256)
        # 256 from dec4 + 256 from enc3
        self.dec_3 = DoubleConv(in_channels=256+256, out_channels=128)
        # 128 from dec3 + 128 from enc2
        self.dec_2 = DoubleConv(in_channels=128+128, out_channels=64)
        # 64 from dec2 + 64 from enc1
        self.dec1 = DoubleConv(in_channels=64+64, out_channels=64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.upsample = torch.nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=True,
        )

        self.final_conv = torch.nn.Conv2d(
            in_channels=64,
            out_channels=3,
            kernel_size=1,
        )
        self.activation = torch.nn.Sigmoid()

        # initialize VGG for perceptual loss
        self.vgg = (
            torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            .features[:16]
        )
        self.vgg.eval()

        # freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(B, C, H, W)``

        Returns
        -------
        output: torch.Tensor
            Output tensor of shape ``(B, C, H, W)``

        """
        # encoder blocks
        enc1 = self.enc_1(x)
        enc2 = self.enc_2(self.pool(enc1))
        enc3 = self.enc_3(self.pool(enc2))
        enc4 = self.enc_4(self.pool(enc3))
        enc5 = self.enc_5(self.pool(enc4))

        # decoder blocks with skip connections
        dec5 = self.dec_5(enc5)
        # for each upsampling step, ensure the dimensions match before adding
        dec4_up = self.upsample(dec5)
        dec4 = self.dec_4(torch.cat([dec4_up, enc4], dim=1))

        dec3_up = self.upsample(dec4)
        dec3 = self.dec_3(torch.cat([dec3_up, enc3], dim=1))

        dec2_up = self.upsample(dec3)
        dec2 = self.dec_2(torch.cat([dec2_up, enc2], dim=1))

        dec1_up = self.upsample(dec2)
        dec1 = self.dec1(torch.cat([dec1_up, enc1], dim=1))

        # final output
        return self.activation(self.final_conv(dec1))

    def _compute_loss(
        self: Self,
        output: torch.Tensor,
        target: torch.Tensor,
        prefix: str = '',
    ) -> torch.Tensor:
        """
        Compute the total loss for a batch.

        Parameters
        ----------
        output: torch.Tensor
            Model output tensor
        target: torch.Tensor
            Target tensor
        prefix: str
            Prefix for logging (e.g., 'train_' or 'val_')

        Returns
        -------
        loss: torch.Tensor
            Total loss for the batch
        """
        # compute L1 loss
        l1_loss = torch.nn.functional.l1_loss(input=output, target=target)

        # compute perceptual loss
        perceptual_loss = self._compute_perceptual_loss(input=output, target=target)

        # compute total loss
        total_loss = l1_loss + 0.1 * perceptual_loss

        # log losses
        self.log(
            name=f'{prefix}_l1_loss',
            value=l1_loss,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            name=f'{prefix}_perceptual_loss',
            value=perceptual_loss,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            name=f'{prefix}_total_loss',
            value=total_loss,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def _compute_perceptual_loss(
        self: Self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute perceptual loss using VGG features.

        Parameters
        ----------
        input: torch.Tensor
            First image tensor
        target: torch.Tensor
            Second image tensor

        Returns
        -------
        loss: torch.Tensor
            Perceptual loss value

        """
        # VGG expects 224x224 images
        vgg_size = 224

        # resize images for VGG while preserving aspect ratio
        common_interpolate_kwargs = {
            'size': (vgg_size, vgg_size),
            'mode': 'bilinear',
            'align_corners': True,
        }

        input_resized = torch.nn.functional.interpolate(
            input=input,
            **common_interpolate_kwargs,
        )
        target_resized = torch.nn.functional.interpolate(
            input=target,
            **common_interpolate_kwargs,
        )

        with torch.no_grad():
            x_features = self.vgg(input_resized)
            y_features = self.vgg(target_resized)

        return torch.nn.functional.l1_loss(input=x_features, target=y_features)

    def training_step(
        self: Self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Training step.

        Parameters
        ----------
        batch: tuple of torch.Tensor
            Tuple containing the watermarked image and the target image

        Returns
        -------
        loss: torch.Tensor
            Total loss for the batch
        """
        watermarked, target = batch

        output = self(watermarked)

        return self._compute_loss(output=output, target=target, prefix='train')

    def validation_step(
        self: Self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch: tuple of torch.Tensor
            Tuple containing the watermarked image and the target image

        Returns
        -------
        loss: torch.Tensor
            Total loss for the batch
        """
        watermarked, target = batch

        output = self(watermarked)

        return self._compute_loss(output=output, target=target, prefix='val')

    def configure_optimizers(self: Self) -> Dict[str, Any]:
        """
        Configure optimizers and learning rate schedulers.

        Returns
        -------
        config: dict
            Dictionary containing optimizer and scheduler configurations

        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


class DifficultyScheduler(pl.Callback):
    """
    Callback that increases the dataset difficulty after each epoch.

    Parameters
    ----------
    initial_difficulty: float
        Initial difficulty level between ``0`` and ``1``
    max_difficulty: float
        Maximum difficulty level to reach
    step_size: float
        Amount to increase difficulty by each epoch

    """
    def __init__(
        self: Self,
        initial_difficulty: float = 0.5,
        max_difficulty: float = 0.9,
        step_size: float = 0.05,
    ) -> None:
        super().__init__()
        self.initial_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.step_size = step_size
        self.current_difficulty = initial_difficulty

    def on_train_epoch_end(self: Self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Increase difficulty at the end of each training epoch."""
        # get the dataset from the first dataloader
        dataset = trainer.train_dataloader.dataset

        # increase difficulty
        self.current_difficulty = min(
            (self.current_difficulty + self.step_size),
            self.max_difficulty,
        )

        # update dataset difficulty
        dataset.difficulty = self.current_difficulty

        # log the new difficulty
        trainer.logger.log_metrics(
            metrics={'watermark_difficulty': self.current_difficulty},
            step=trainer.current_epoch
        )
