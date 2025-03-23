import os
from typing import Any, Dict, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import Self

from watermark import watermark_image


class WatermarkedDataset(Dataset):
    """
    Dataset that returns pairs of original and images augmented with a watermark.

    Parameters
    ----------
    root_dir: str
        Directory containing the images to watermark, where images are expected to be in the
        ``.png``, ``.jpg``, or ``.jpeg`` format
    difficulty: float
        Difficulty level between ``0`` and ``1``, where higher values create harder watermarks
    image_size: int
        Standard size to resize all images to

    """
    def __init__(
        self: Self,
        root_dir: str,
        difficulty: float = 0.5,
        image_size: int = 512,
    ) -> None:
        self.root_dir = root_dir
        self.difficulty = difficulty
        self.image_size = image_size

        # get all image files
        self.image_files = [
            f for f in os.listdir(path=root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        # transform that resizes and normalizes to ``[0, 1]``
        self.transform = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
        ])

        # base watermark parameters that will be adjusted based on difficulty
        self.base_params = {
            'font_size': (24, 48),
            'color': [(200, 200, 200, 128)],
            'angle': (30, 60),
            'text_padding': (10, 30),
            'opacity_range': (0.5, 0.9),
            'logo_scale_range': (0.8, 1.2),
            'logo_rotation_range': (-15, 15),
            'randomize_spacing': True,
            'randomize_angle': True,
            'randomize_size': True,
            'randomize_color': True,
            'multiple_colors': False,
            'draw_grid': False,
        }

        # parameters for maximum difficulty
        self.hard_params = {
            'font_size': (12, 72),
            'color': [(200, 200, 200, 128), (150, 150, 150, 128), (100, 100, 100, 128)],
            'angle': (0, 90),
            'text_padding': (5, 50),
            'opacity_range': (0.3, 0.95),
            'logo_scale_range': (0.5, 1.5),
            'logo_rotation_range': (-45, 45),
            'randomize_spacing': True,
            'randomize_angle': True,
            'randomize_size': True,
            'randomize_color': True,
            'multiple_colors': True,
            'draw_grid': True,
            'grid_spacing': (20, 100),
            'line_thickness': (1, 3),
        }

    @property
    def difficulty(self: Self) -> float:
        """
        Get the current difficulty level.

        Returns
        -------
        difficulty: float
            Current difficulty level between ``0`` and ``1``

        """
        return self._difficulty

    @difficulty.setter
    def difficulty(self: Self, value: float) -> None:
        """
        Set the difficulty level.

        Parameters
        ----------
        value: float
            New difficulty level between ``0`` and ``1``

        Raises
        ------
        ValueError
            If value is not between ``0`` and ``1``

        """
        if not 0 <= value <= 1:
            raise ValueError(f'Difficulty must be between ``0`` and ``1``, not ``{value}``!')

        self._difficulty = value

    def __len__(self: Self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def _interpolate_params(self: Self) -> Dict[str, Any]:
        """Interpolate between base and hard watermarking parameters based on difficulty level."""
        params = {}

        for key in self.base_params:
            if isinstance(self.base_params[key], tuple):
                # for numeric ranges, interpolate between min and max
                base_min, base_max = self.base_params[key]
                hard_min, hard_max = self.hard_params[key]

                min_val = base_min + (hard_min - base_min) * self._difficulty
                max_val = base_max + (hard_max - base_max) * self._difficulty

                params[key] = (min_val, max_val)
            else:
                # for boolean values, use hard params if difficulty is high enough
                params[key] = (
                    self.hard_params[key]
                    if self._difficulty > 0.7 and key in self.hard_params
                    else self.base_params[key]
                )

        return params

    def __getitem__(self: Self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of original and watermarked images.

        Parameters
        ----------
        idx: int
            Index of the image to load

        Returns
        -------
        original: torch.Tensor
            Original image as a tensor
        watermarked: torch.Tensor
            Watermarked image as a tensor

        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        original = Image.open(fp=img_path).convert(mode='RGB')

        watermarked = watermark_image(image=original, **self._interpolate_params())
        watermarked = watermarked.convert(mode='RGB')

        # convert to tensors and normalize to ``[0, 1]``
        original = self.transform(original)
        watermarked = self.transform(watermarked)

        return original, watermarked
