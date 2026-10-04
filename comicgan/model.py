"""Generator (ResNet-style encoder/decoder with skips) and PatchGAN discriminator.

Only built-in Keras layers are used, so saved `.keras` models load without
`custom_objects`. Instance normalization is `GroupNormalization(groups=-1)`.
"""

from __future__ import annotations

import keras
from keras import layers

CHANNELS = 3


def _conv_init() -> keras.initializers.Initializer:
    return keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def _instance_norm() -> layers.Layer:
    return layers.GroupNormalization(
        groups=-1, epsilon=1e-3, gamma_initializer=_conv_init()
    )


def encoder_block(x, filters, size=3, strides=2, instance_norm=True, name="block"):
    x = layers.Conv2D(
        filters,
        size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=_conv_init(),
        name=f"encoder_{name}",
    )(x)
    if instance_norm:
        x = _instance_norm()(x)
    return layers.LeakyReLU(0.2)(x)


def residual_block(x, size=3, name="block"):
    filters = x.shape[-1]
    y = layers.Conv2D(
        filters, size, padding="same", use_bias=False,
        kernel_initializer=_conv_init(), name=f"transformer_{name}_1",
    )(x)
    y = layers.LeakyReLU(0.2)(y)
    y = layers.Conv2D(
        filters, size, padding="same", use_bias=False,
        kernel_initializer=_conv_init(), name=f"transformer_{name}_2",
    )(y)
    return layers.Add()([y, x])


def decoder_block(x, filters, size=3, strides=2, name="block"):
    x = layers.Conv2DTranspose(
        filters,
        size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=_conv_init(),
        name=f"decoder_{name}",
    )(x)
    x = _instance_norm()(x)
    return layers.LeakyReLU(0.3)(x)


def build_generator(
    transformer_blocks: int = 6,
    height: int | None = None,
    width: int | None = None,
    name: str = "generator",
) -> keras.Model:
    """Image-to-image generator; input/output in [-1, 1].

    With `height=width=None` it accepts any size divisible by 4.
    """
    inputs = layers.Input(shape=(height, width, CHANNELS), name="input_image")

    enc_1 = encoder_block(inputs, 64, 7, 1, instance_norm=False, name="block_1")
    enc_2 = encoder_block(enc_1, 128, 3, 2, name="block_2")  # 1/2
    enc_3 = encoder_block(enc_2, 256, 3, 2, name="block_3")  # 1/4

    x = enc_3
    for n in range(transformer_blocks):
        x = residual_block(x, name=f"block_{n + 1}")

    x = layers.Concatenate(name="enc_dec_skip_1")([x, enc_3])
    x = decoder_block(x, 128, name="block_1")  # 1/2
    x = layers.Concatenate(name="enc_dec_skip_2")([x, enc_2])
    x = decoder_block(x, 64, name="block_2")  # 1/1
    x = layers.Concatenate(name="enc_dec_skip_3")([x, enc_1])

    outputs = layers.Conv2D(
        CHANNELS,
        7,
        padding="same",
        use_bias=False,
        activation="tanh",
        kernel_initializer=_conv_init(),
        name="decoder_output_block",
    )(x)
    return keras.Model(inputs, outputs, name=name)


def build_discriminator(name: str = "discriminator") -> keras.Model:
    """PatchGAN discriminator returning a grid of real/fake logits."""
    inputs = layers.Input(shape=(None, None, CHANNELS), name="input_image")

    x = encoder_block(inputs, 64, 4, 2, instance_norm=False, name="block_1")
    x = encoder_block(x, 128, 4, 2, name="block_2")
    x = encoder_block(x, 256, 4, 2, name="block_3")

    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, use_bias=False, kernel_initializer=_conv_init())(x)
    x = _instance_norm()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.ZeroPadding2D()(x)
    outputs = layers.Conv2D(1, 4, kernel_initializer=_conv_init())(x)
    return keras.Model(inputs, outputs, name=name)
