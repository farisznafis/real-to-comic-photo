import numpy as np
import pytest
import tensorflow as tf

from comicgan.config import TrainConfig
from comicgan.model import build_discriminator, build_generator
from comicgan.train import build_cyclegan


@pytest.fixture(scope="module")
def generator():
    return build_generator(transformer_blocks=1)


@pytest.mark.parametrize("shape", [(1, 64, 64, 3), (2, 32, 48, 3)])
def test_generator_preserves_shape_and_range(generator, shape):
    out = generator(np.random.uniform(-1, 1, shape).astype("float32"))
    assert out.shape == shape
    assert float(tf.reduce_min(out)) >= -1.0 and float(tf.reduce_max(out)) <= 1.0


def test_discriminator_outputs_patch_logits():
    out = build_discriminator()(np.zeros((2, 128, 128, 3), "float32"))
    assert out.shape == (2, 14, 14, 1)


def test_generator_roundtrips_through_keras_file(generator, tmp_path):
    import keras

    path = tmp_path / "gen.keras"
    generator.save(path)
    loaded = keras.models.load_model(path, compile=False)  # no custom_objects needed
    x = np.random.uniform(-1, 1, (1, 32, 32, 3)).astype("float32")
    np.testing.assert_allclose(generator(x), loaded(x), atol=1e-5)


def test_train_step_returns_finite_losses():
    model = build_cyclegan(TrainConfig(transformer_blocks=1))
    batch = (
        tf.random.uniform((2, 32, 32, 3), -1, 1),
        tf.random.uniform((2, 32, 32, 3), -1, 1),
    )
    logs = model.train_step(batch)
    assert set(logs) == {"p2c_gen_loss", "c2p_gen_loss", "comic_disc_loss", "photo_disc_loss"}
    assert all(np.isfinite(float(v)) for v in logs.values())
