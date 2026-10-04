"""Load trained generators and run them on PIL images."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
from PIL import Image, ImageOps

# The generator downsamples twice, so spatial dims must be divisible by 4.
SIZE_MULTIPLE = 4


def round_to_multiple(value: float, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def center_crop_square(image: Image.Image) -> Image.Image:
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def preprocess(image: Image.Image, size: int, crop_square: bool = True) -> np.ndarray:
    """PIL image -> float32 batch of shape (1, H, W, 3) in [-1, 1].

    `size` is the output's longer side; each side is rounded to a multiple of 4.
    """
    image = ImageOps.exif_transpose(image).convert("RGB")
    if crop_square:
        image = center_crop_square(image)
    scale = size / max(image.size)
    target = (round_to_multiple(image.width * scale), round_to_multiple(image.height * scale))
    image = image.resize(target, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return array[np.newaxis, ...]


def postprocess(batch) -> Image.Image:
    """First image of a [-1, 1] batch (or a single HWC image) -> PIL image."""
    array = np.asarray(batch)
    if array.ndim == 4:
        array = array[0]
    array = np.clip((array + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
    return Image.fromarray(array)


def load_generator(path: Path | str) -> keras.Model:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}. Train one first with `python -m comicgan.train`."
        )
    return keras.models.load_model(path, compile=False)


def translate(
    model: keras.Model, image: Image.Image, size: int = 256, crop_square: bool = True
) -> Image.Image:
    batch = preprocess(image, size, crop_square)
    return postprocess(model(batch, training=False))
