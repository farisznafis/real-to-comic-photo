import numpy as np
import pytest
from PIL import Image

from comicgan.inference import (
    center_crop_square,
    load_generator,
    postprocess,
    preprocess,
    round_to_multiple,
    translate,
)
from comicgan.model import build_generator


def test_round_to_multiple():
    assert round_to_multiple(127) == 128
    assert round_to_multiple(1) == 4


def test_center_crop_square():
    assert center_crop_square(Image.new("RGB", (300, 200))).size == (200, 200)


@pytest.mark.parametrize(
    ("crop", "expected"), [(True, (1, 256, 256, 3)), (False, (1, 168, 256, 3))]
)
def test_preprocess_shape_and_range(crop, expected):
    batch = preprocess(Image.new("RGB", (300, 197), "white"), 256, crop_square=crop)
    assert batch.shape == expected
    assert batch.dtype == np.float32
    assert batch.max() == pytest.approx(1.0)


def test_postprocess_inverts_preprocess():
    pixels = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    restored = postprocess(preprocess(Image.fromarray(pixels), 64))
    np.testing.assert_array_equal(np.asarray(restored), pixels)


def test_translate_returns_image():
    result = translate(build_generator(1), Image.new("RGB", (50, 70)), size=64)
    assert isinstance(result, Image.Image) and result.size == (64, 64)


def test_load_generator_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one first"):
        load_generator(tmp_path / "missing.keras")
