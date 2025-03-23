from pathlib import Path

import lightning as pl
from torch.utils.data import DataLoader

from dataset import WatermarkedDataset
from trainer import DifficultyScheduler, WatermarkRemovalModel


def calculate_difficulty_step(
    initial_difficulty: float,
    max_difficulty: float,
    max_epochs: int,
) -> float:
    """
    Calculate the step size for difficulty increase based on epochs.

    Parameters
    ----------
    initial_difficulty: float
        Starting difficulty level between ``0`` and ``1``
    max_difficulty: float
        Target difficulty level between ``0`` and ``1``
    max_epochs: int
        Number of epochs to train for

    Returns
    -------
    step_size: float
        Amount to increase difficulty by each epoch

    """
    total_difficulty_increase = max_difficulty - initial_difficulty
    return total_difficulty_increase / max_epochs


def train(
    data_dir: str,
    output_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    initial_difficulty: float = 0.3,
    max_difficulty: float = 0.9,
    accelerator: str = 'gpu',
    devices: int = 1,
    precision: str = '16-mixed',
    image_size: int = 512,
) -> None:
    """
    Train the watermark removal model with automatic difficulty scheduling.

    Parameters
    ----------
    data_dir: str
        Directory containing training images
    output_dir: str
        Directory to save model checkpoints
    batch_size: int
        Batch size for training
    num_workers: int
        Number of data loading workers
    max_epochs: int
        Maximum number of training epochs
    learning_rate: float
        Initial learning rate
    initial_difficulty: float
        Initial watermark difficulty level between ``0`` and ``1``
    max_difficulty: float
        Maximum watermark difficulty level to reach between ``0`` and ``1``
    accelerator: str
        Training accelerator (e.g., ``gpu``, ``cpu``, etc.)
    devices: int
        Number of devices to use
    precision: str
        Training precision (e.g., ``16-mixed``, ``32``, etc.)
    image_size: int
        Size to resize all images to

    """
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # calculate difficulty step size
    difficulty_step = calculate_difficulty_step(
        initial_difficulty=initial_difficulty,
        max_difficulty=max_difficulty,
        max_epochs=max_epochs,
    )

    # create datasets
    train_dataset = WatermarkedDataset(
        root_dir=data_dir,
        difficulty=initial_difficulty,
        image_size=image_size,
    )

    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    # create model
    model = WatermarkRemovalModel(
        learning_rate=learning_rate,
    )

    # create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        precision=precision,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=output_path,
                filename='watermark-removal-{epoch:02d}-{val_total_loss:.2f}',
                monitor='val_total_loss',
                mode='min',
                save_top_k=3,
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_total_loss',
                patience=10,
                mode='min',
            ),
            DifficultyScheduler(
                initial_difficulty=initial_difficulty,
                max_difficulty=max_difficulty,
                step_size=difficulty_step,
            ),
        ],
    )

    # train the model
    trainer.fit(model, train_loader)


# def main() -> None:
#     """Parse command line arguments and run training."""
#     parser = argparse.ArgumentParser(description="Train watermark removal model")
#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         required=True,
#         help="Directory containing training images"
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         required=True,
#         help="Directory to save model checkpoints"
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=4,
#         help="Batch size for training"
#     )
#     parser.add_argument(
#         "--num_workers",
#         type=int,
#         default=4,
#         help="Number of data loading workers"
#     )
#     parser.add_argument(
#         "--max_epochs",
#         type=int,
#         default=100,
#         help="Maximum number of training epochs"
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=1e-4,
#         help="Initial learning rate"
#     )
#     parser.add_argument(
#         "--image_size",
#         type=int,
#         default=512,
#         help="Size to resize images to"
#     )
#     parser.add_argument(
#         "--initial_difficulty",
#         type=float,
#         default=0.3,
#         help="Initial watermark difficulty level (0-1)"
#     )
#     parser.add_argument(
#         "--max_difficulty",
#         type=float,
#         default=0.9,
#         help="Maximum watermark difficulty level to reach"
#     )
#     parser.add_argument(
#         "--accelerator",
#         type=str,
#         default="gpu",
#         help="Training accelerator (e.g., 'gpu', 'cpu')"
#     )
#     parser.add_argument(
#         "--devices",
#         type=int,
#         default=1,
#         help="Number of devices to use"
#     )
#     parser.add_argument(
#         "--precision",
#         type=str,
#         default="16-mixed",
#         help="Training precision (e.g., '16-mixed', '32')"
#     )

#     args = parser.parse_args()

#     # Validate difficulty values
#     if not 0 <= args.initial_difficulty <= 1:
#         raise ValueError("Initial difficulty must be between ``0`` and ``1``")
#     if not 0 <= args.max_difficulty <= 1:
#         raise ValueError("Max difficulty must be between ``0`` and ``1``")
#     if args.initial_difficulty >= args.max_difficulty:
#         raise ValueError("Initial difficulty must be less than max difficulty")

#     train(**vars(args))


# if __name__ == "__main__":
#     main()
