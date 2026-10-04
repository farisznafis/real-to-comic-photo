"""Train the CycleGAN.

Usage:
    python -m comicgan.train --epochs 50
    python -m comicgan.train --epochs 1 --limit 64 --batch-size 4   # quick smoke run
    python -m comicgan.train --resume                               # continue an interrupted run
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from . import losses
from .config import COMIC2PHOTO_FILE, PHOTO2COMIC_FILE, TrainConfig
from .cyclegan import CycleGAN
from .data import list_domain_files, make_dataset, split_paths
from .inference import postprocess
from .model import build_discriminator, build_generator


class SampleSaver(keras.callbacks.Callback):
    """Every `every` epochs, write a PNG grid: [photo | fake comic] and [comic | fake photo]."""

    def __init__(self, sample_batch, out_dir: Path, every: int, num_images: int = 4):
        super().__init__()
        photos, comics = sample_batch
        self.photos = np.asarray(photos)[:num_images]
        self.comics = np.asarray(comics)[:num_images]
        self.out_dir = Path(out_dir)
        self.every = max(1, every)

    def on_epoch_end(self, epoch, logs=None):
        last_epoch = epoch + 1 == self.params.get("epochs")
        if (epoch + 1) % self.every and not last_epoch:
            return
        fake_comics = self.model.photo2comic(self.photos, training=False)
        fake_photos = self.model.comic2photo(self.comics, training=False)
        rows = [
            np.concatenate([self.photos, fake_comics, self.comics, fake_photos], axis=2)[i]
            for i in range(len(self.photos))
        ]
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"epoch_{epoch + 1:03d}.png"
        postprocess(np.concatenate(rows, axis=0)).save(path)


class GeneratorExporter(keras.callbacks.Callback):
    """Save both generators every `every` epochs and to `model_dir` when training ends."""

    def __init__(self, checkpoint_dir: Path, model_dir: Path, every: int):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = Path(model_dir)
        self.every = max(1, every)

    def _save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.model.photo2comic.save(folder / PHOTO2COMIC_FILE)
        self.model.comic2photo.save(folder / COMIC2PHOTO_FILE)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0:
            self._save(self.checkpoint_dir / f"epoch_{epoch + 1:03d}")

    def on_train_end(self, logs=None):
        self._save(self.model_dir)
        print(f"Saved generators to {self.model_dir.resolve()}")


class TrainingState(keras.callbacks.Callback):
    """Checkpoint everything needed to resume: all four networks, optimizers and the epoch."""

    def __init__(self, model: CycleGAN, directory: Path, every: int):
        super().__init__()
        self.checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(0, dtype=tf.int64),
            photo2comic=model.photo2comic,
            comic2photo=model.comic2photo,
            comic_disc=model.comic_disc,
            photo_disc=model.photo_disc,
            p2c_optimizer=model.p2c_optimizer,
            c2p_optimizer=model.c2p_optimizer,
            comic_disc_optimizer=model.comic_disc_optimizer,
            photo_disc_optimizer=model.photo_disc_optimizer,
        )
        # Only the latest state is kept; it is ~250 MB with the default model size.
        self.manager = tf.train.CheckpointManager(self.checkpoint, str(directory), max_to_keep=1)
        self.every = max(1, every)

    def restore(self, source: Path | None = None) -> int | None:
        """Load the latest state from `source` (default: own directory).

        Returns the number of completed epochs, or None if no state was found.
        """
        path = tf.train.latest_checkpoint(str(source)) if source else self.manager.latest_checkpoint
        if path is None:
            return None
        self.checkpoint.restore(path).assert_existing_objects_matched()
        print(f"Resumed from {path}")
        return int(self.checkpoint.epoch)

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint.epoch.assign(epoch + 1)
        if (epoch + 1) % self.every == 0 or epoch + 1 == self.params.get("epochs"):
            self.manager.save(checkpoint_number=epoch + 1)


class HistoryCSV(keras.callbacks.Callback):
    """Append each epoch's losses to a CSV file (one row per epoch)."""

    def __init__(self, path: Path, append: bool = False):
        super().__init__()
        self.path = Path(path)
        self.append = append

    def on_train_begin(self, logs=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.append:
            self.path.unlink(missing_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        keys = sorted(logs)
        write_header = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", *keys])
            writer.writerow([epoch + 1, *(float(logs[k]) for k in keys)])


def configure_gpus() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def build_cyclegan(cfg: TrainConfig) -> CycleGAN:
    model = CycleGAN(
        photo2comic=build_generator(cfg.transformer_blocks, name="photo2comic"),
        comic2photo=build_generator(cfg.transformer_blocks, name="comic2photo"),
        comic_discriminator=build_discriminator(name="comic_discriminator"),
        photo_discriminator=build_discriminator(name="photo_discriminator"),
        lambda_cycle=cfg.lambda_cycle,
    )
    model.compile(
        p2c_optimizer=keras.optimizers.Adam(cfg.gen_lr, beta_1=cfg.beta_1),
        c2p_optimizer=keras.optimizers.Adam(cfg.gen_lr, beta_1=cfg.beta_1),
        comic_disc_optimizer=keras.optimizers.Adam(cfg.disc_lr, beta_1=cfg.beta_1),
        photo_disc_optimizer=keras.optimizers.Adam(cfg.disc_lr, beta_1=cfg.beta_1),
        gen_loss_fn=losses.generator_loss,
        disc_loss_fn=losses.make_discriminator_loss(cfg.label_smoothing),
        cycle_loss_fn=losses.cycle_loss,
        identity_loss_fn=losses.identity_loss,
    )
    return model


def train(cfg: TrainConfig) -> keras.callbacks.History:
    configure_gpus()
    keras.utils.set_random_seed(cfg.seed)

    photos, comics = list_domain_files(cfg.data_dir)
    if cfg.limit:
        photos, comics = photos[: cfg.limit], comics[: cfg.limit]
    photo_train, photo_test = split_paths(photos, cfg.test_size, cfg.seed)
    comic_train, comic_test = split_paths(comics, cfg.test_size, cfg.seed + 1)
    print(
        f"Photos: {len(photo_train)} train / {len(photo_test)} test | "
        f"Comics: {len(comic_train)} train / {len(comic_test)} test"
    )

    steps = cfg.steps_per_epoch or min(len(photo_train), len(comic_train)) // cfg.batch_size
    if steps < 1:
        raise ValueError("Not enough images for one batch; lower --batch-size.")

    train_ds = make_dataset(photo_train, comic_train, cfg, training=True)
    # Fall back to training images for samples if the test split is empty.
    test_ds = make_dataset(
        photo_test or photo_train, comic_test or comic_train, cfg, training=False
    )
    sample_batch = next(iter(test_ds))

    model = build_cyclegan(cfg)
    state = TrainingState(model, cfg.state_dir, cfg.checkpoint_every)
    initial_epoch = _restore_state(state, cfg)

    callbacks = [
        state,
        SampleSaver(sample_batch, cfg.output_dir / "samples", cfg.sample_every),
        GeneratorExporter(cfg.output_dir / "checkpoints", cfg.model_dir, cfg.save_every),
        HistoryCSV(cfg.output_dir / "history.csv", append=initial_epoch > 0),
    ]
    return model.fit(
        train_ds,
        epochs=cfg.epochs,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps,
        callbacks=callbacks,
        shuffle=False,  # the tf.data pipeline already shuffles
    )


def _restore_state(state: TrainingState, cfg: TrainConfig) -> int:
    """Return the epoch to start from (0 for a fresh run)."""
    if cfg.resume_from:
        epoch = state.restore(cfg.resume_from)
        if epoch is None:
            raise FileNotFoundError(f"No training state found in {cfg.resume_from}")
    elif cfg.resume:
        epoch = state.restore()
        if epoch is None:
            print(f"No training state in {cfg.state_dir}; starting from scratch.")
            return 0
    else:
        return 0
    if epoch >= cfg.epochs:
        print(f"State is already at epoch {epoch} (>= --epochs {cfg.epochs}); exporting only.")
    else:
        print(f"Continuing from epoch {epoch + 1}/{cfg.epochs}")
    return epoch


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the photo <-> comic CycleGAN.")
    defaults = TrainConfig()
    for f in dataclasses.fields(TrainConfig):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, Path):
            parser.add_argument(flag, type=Path, default=default)
        elif default is None:
            parser.add_argument(flag, type=f.metadata.get("type", int), default=None)
        else:
            parser.add_argument(flag, type=type(default), default=default)
    return TrainConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
