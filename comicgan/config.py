"""Project paths and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Face2Comics v2 by Sxela. `faces/` holds real photos, `comics/` the comic versions.
DEFAULT_DATA_DIR = (
    PROJECT_ROOT
    / "comic-faces-paired-synthetic-v2"
    / "face2comics_v2.0.0_by_Sxela"
    / "face2comics_v2.0.0_by_Sxela"
)
PHOTO_SUBDIR = "faces"
COMIC_SUBDIR = "comics"

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

PHOTO2COMIC_FILE = "photo2comic.keras"
COMIC2PHOTO_FILE = "comic2photo.keras"


@dataclass
class TrainConfig:
    """Training settings. Defaults match the original Kaggle notebook."""

    data_dir: Path = DEFAULT_DATA_DIR
    model_dir: Path = DEFAULT_MODEL_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR

    # Images are resized to `load_size` and randomly cropped to `image_size`.
    image_size: int = 128
    load_size: int = 142
    batch_size: int = 16
    epochs: int = 50
    test_size: float = 0.01
    shuffle_buffer: int = 1000
    # Cache resized images in RAM (~1.2 GB for the full dataset at load_size=142).
    cache: bool = True
    # Use only the first N images per domain (handy for quick smoke runs).
    limit: int | None = None
    # Defaults to min(#photos, #comics) // batch_size.
    steps_per_epoch: int | None = None
    seed: int = 420

    transformer_blocks: int = 6
    gen_lr: float = 1e-4
    disc_lr: float = 4e-4
    beta_1: float = 0.5
    lambda_cycle: float = 10.0
    label_smoothing: float = 0.1

    sample_every: int = 4
    save_every: int = 5

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.model_dir = Path(self.model_dir)
        self.output_dir = Path(self.output_dir)
        if self.load_size < self.image_size:
            raise ValueError("load_size must be >= image_size")
        if self.image_size % 4 != 0:
            raise ValueError("image_size must be a multiple of 4")
