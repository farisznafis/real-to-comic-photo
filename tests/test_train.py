import csv

import numpy as np
import pytest

from comicgan.config import COMIC2PHOTO_FILE, PHOTO2COMIC_FILE, TrainConfig
from comicgan.train import parse_args, train


def _config(data_dir, tmp_path, **overrides):
    return TrainConfig(
        data_dir=data_dir,
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "models",
        image_size=32,
        load_size=36,
        batch_size=2,
        steps_per_epoch=1,
        transformer_blocks=1,
        **overrides,
    )


def _history_epochs(cfg):
    with (cfg.output_dir / "history.csv").open() as f:
        return [int(row["epoch"]) for row in csv.DictReader(f)]


def test_train_writes_models_samples_and_state(data_dir, tmp_path):
    cfg = _config(data_dir, tmp_path, epochs=1)
    train(cfg)
    assert (cfg.model_dir / PHOTO2COMIC_FILE).is_file()
    assert (cfg.model_dir / COMIC2PHOTO_FILE).is_file()
    assert (cfg.output_dir / "samples" / "epoch_001.png").is_file()
    assert (cfg.state_dir / "checkpoint").is_file()
    assert _history_epochs(cfg) == [1]


def test_resume_continues_from_saved_epoch(data_dir, tmp_path):
    train(_config(data_dir, tmp_path, epochs=2))

    cfg = _config(data_dir, tmp_path, epochs=3, resume=True)
    history = train(cfg)
    assert len(history.history["p2c_gen_loss"]) == 1  # only epoch 3 ran
    assert _history_epochs(cfg) == [1, 2, 3]
    assert (cfg.output_dir / "samples" / "epoch_003.png").is_file()


def test_resume_from_other_folder(data_dir, tmp_path):
    first = _config(data_dir, tmp_path / "run1", epochs=1)
    train(first)

    second = _config(data_dir, tmp_path / "run2", epochs=2, resume_from=first.state_dir)
    history = train(second)
    assert len(history.history["p2c_gen_loss"]) == 1
    assert np.isfinite(history.history["p2c_gen_loss"][0])


def test_resume_from_missing_folder_fails(data_dir, tmp_path):
    cfg = _config(data_dir, tmp_path, epochs=1, resume_from=tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="No training state"):
        train(cfg)


def test_parse_args_types(tmp_path):
    cfg = parse_args(["--epochs", "3", "--resume-from", str(tmp_path), "--no-cache"])
    assert cfg.epochs == 3 and cfg.resume_from == tmp_path and cfg.cache is False
    assert parse_args([]).resume_from is None
