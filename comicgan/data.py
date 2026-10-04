"""Dataset discovery and tf.data input pipelines.

CycleGAN learns from *unpaired* data, so the photo and comic streams are shuffled
independently and zipped together.
"""

from __future__ import annotations

import random
from pathlib import Path

import tensorflow as tf

from .config import COMIC_SUBDIR, PHOTO_SUBDIR, TrainConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


def _list_images(folder: Path) -> list[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    files = sorted(
        str(p) for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return files


def list_domain_files(data_dir: Path | str) -> tuple[list[str], list[str]]:
    """Return sorted (photo_paths, comic_paths) found under `data_dir`."""
    data_dir = Path(data_dir)
    return _list_images(data_dir / PHOTO_SUBDIR), _list_images(data_dir / COMIC_SUBDIR)


def split_paths(
    paths: list[str], test_size: float, seed: int
) -> tuple[list[str], list[str]]:
    """Shuffle deterministically and split into (train, test)."""
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    n_test = int(round(len(paths) * test_size))
    if test_size > 0 and len(paths) > 1:
        n_test = min(max(n_test, 1), len(paths) - 1)
    return paths[n_test:], paths[:n_test]


def normalize(image: tf.Tensor) -> tf.Tensor:
    """Map uint8 [0, 255] to float32 [-1, 1] (the generator's tanh range)."""
    return tf.cast(image, tf.float32) / 127.5 - 1.0


def _load_image(path: tf.Tensor, size: int) -> tf.Tensor:
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image = tf.image.resize(image, [size, size], method="bilinear", antialias=True)
    # Keep uint8 so the cache stays small (4x smaller than float32).
    return tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)


def _augment(image: tf.Tensor, crop_size: int) -> tf.Tensor:
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    return tf.image.random_flip_left_right(image)


def _domain_dataset(
    paths: list[str], cfg: TrainConfig, training: bool, seed: int
) -> tf.data.Dataset:
    size = cfg.load_size if training else cfg.image_size
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: _load_image(p, size), num_parallel_calls=AUTOTUNE)
    if cfg.cache:
        ds = ds.cache()
    ds = ds.shuffle(
        min(cfg.shuffle_buffer, len(paths)), seed=seed, reshuffle_each_iteration=True
    ).repeat()
    if training:
        ds = ds.map(lambda x: _augment(x, cfg.image_size), num_parallel_calls=AUTOTUNE)
    return ds.map(normalize, num_parallel_calls=AUTOTUNE)


def make_dataset(
    photo_paths: list[str],
    comic_paths: list[str],
    cfg: TrainConfig,
    training: bool,
) -> tf.data.Dataset:
    """Infinite dataset of `(photo_batch, comic_batch)` tuples in [-1, 1]."""
    photos = _domain_dataset(photo_paths, cfg, training, cfg.seed)
    comics = _domain_dataset(comic_paths, cfg, training, cfg.seed + 1)
    ds = tf.data.Dataset.zip((photos, comics))
    return ds.batch(cfg.batch_size, drop_remainder=True).prefetch(AUTOTUNE)
