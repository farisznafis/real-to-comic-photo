"""CycleGAN loss functions. All return scalars."""

from __future__ import annotations

import keras
import tensorflow as tf


def make_discriminator_loss(label_smoothing: float = 0.1):
    bce = keras.losses.BinaryCrossentropy(
        from_logits=True, label_smoothing=label_smoothing
    )

    def discriminator_loss(real_logits, fake_logits):
        real_loss = bce(tf.ones_like(real_logits), real_logits)
        fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
        return 0.5 * (real_loss + fake_loss)

    return discriminator_loss


_generator_bce = keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_logits):
    """Adversarial loss: the generator wants fakes classified as real."""
    return _generator_bce(tf.ones_like(fake_logits), fake_logits)


def cycle_loss(real, cycled, weight):
    return weight * tf.reduce_mean(tf.abs(real - cycled))


def identity_loss(real, same, weight):
    return 0.5 * weight * tf.reduce_mean(tf.abs(real - same))
