"""CycleGAN training wrapper around two generators and two discriminators."""

from __future__ import annotations

import keras
import tensorflow as tf


class CycleGAN(keras.Model):
    """Trains `photo2comic` / `comic2photo` with adversarial, cycle and identity losses.

    Batches are `(real_photo, real_comic)` tuples in [-1, 1].
    """

    def __init__(
        self,
        photo2comic: keras.Model,
        comic2photo: keras.Model,
        comic_discriminator: keras.Model,
        photo_discriminator: keras.Model,
        lambda_cycle: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.photo2comic = photo2comic
        self.comic2photo = comic2photo
        self.comic_disc = comic_discriminator
        self.photo_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

    def compile(
        self,
        p2c_optimizer,
        c2p_optimizer,
        comic_disc_optimizer,
        photo_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn,
    ):
        super().compile()
        self.p2c_optimizer = p2c_optimizer
        self.c2p_optimizer = c2p_optimizer
        self.comic_disc_optimizer = comic_disc_optimizer
        self.photo_disc_optimizer = photo_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

        # Create optimizer slots up front instead of inside the traced train step.
        for optimizer, model in self._optimizer_pairs():
            optimizer.build(model.trainable_variables)

    def _optimizer_pairs(self):
        return (
            (self.p2c_optimizer, self.photo2comic),
            (self.c2p_optimizer, self.comic2photo),
            (self.comic_disc_optimizer, self.comic_disc),
            (self.photo_disc_optimizer, self.photo_disc),
        )

    def call(self, inputs, training=False):
        return self.photo2comic(inputs, training=training)

    def train_step(self, data):
        real_photo, real_comic = data

        with tf.GradientTape(persistent=True) as tape:
            # photo -> comic -> photo
            fake_comic = self.photo2comic(real_photo, training=True)
            cycled_photo = self.comic2photo(fake_comic, training=True)

            # comic -> photo -> comic
            fake_photo = self.comic2photo(real_comic, training=True)
            cycled_comic = self.photo2comic(fake_photo, training=True)

            # identity mapping: a generator fed its own target domain should change nothing
            same_comic = self.photo2comic(real_comic, training=True)
            same_photo = self.comic2photo(real_photo, training=True)

            disc_real_comic = self.comic_disc(real_comic, training=True)
            disc_fake_comic = self.comic_disc(fake_comic, training=True)
            disc_real_photo = self.photo_disc(real_photo, training=True)
            disc_fake_photo = self.photo_disc(fake_photo, training=True)

            total_cycle_loss = self.cycle_loss_fn(
                real_photo, cycled_photo, self.lambda_cycle
            ) + self.cycle_loss_fn(real_comic, cycled_comic, self.lambda_cycle)

            p2c_loss = (
                self.gen_loss_fn(disc_fake_comic)
                + total_cycle_loss
                + self.identity_loss_fn(real_comic, same_comic, self.lambda_cycle)
            )
            c2p_loss = (
                self.gen_loss_fn(disc_fake_photo)
                + total_cycle_loss
                + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            )

            comic_disc_loss = self.disc_loss_fn(disc_real_comic, disc_fake_comic)
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)

        losses = (p2c_loss, c2p_loss, comic_disc_loss, photo_disc_loss)
        for loss, (optimizer, model) in zip(losses, self._optimizer_pairs()):
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        del tape

        return {
            "p2c_gen_loss": p2c_loss,
            "c2p_gen_loss": c2p_loss,
            "comic_disc_loss": comic_disc_loss,
            "photo_disc_loss": photo_disc_loss,
        }
