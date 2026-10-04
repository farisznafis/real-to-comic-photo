import numpy as np
import pytest
from PIL import Image

from comicgan.config import COMIC_SUBDIR, PHOTO_SUBDIR


@pytest.fixture
def data_dir(tmp_path):
    """Tiny dataset with 6 random images per domain."""
    root = tmp_path / "data"
    rng = np.random.default_rng(0)
    for subdir in (PHOTO_SUBDIR, COMIC_SUBDIR):
        (root / subdir).mkdir(parents=True)
        for i in range(6):
            pixels = rng.integers(0, 256, (80, 80, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / subdir / f"{i}.jpg")
    return root
