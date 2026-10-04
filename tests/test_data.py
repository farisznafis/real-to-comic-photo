import numpy as np
import pytest

from comicgan.config import COMIC_SUBDIR, PHOTO_SUBDIR, TrainConfig
from comicgan.data import list_domain_files, make_dataset, split_paths


def test_list_domain_files(data_dir):
    photos, comics = list_domain_files(data_dir)
    assert len(photos) == len(comics) == 6
    assert all(PHOTO_SUBDIR in p for p in photos)
    assert all(COMIC_SUBDIR in c for c in comics)


def test_list_domain_files_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_domain_files(tmp_path)


def test_split_is_deterministic_and_disjoint():
    paths = [f"{i}.jpg" for i in range(100)]
    train, test = split_paths(paths, 0.1, seed=1)
    assert (train, test) == split_paths(paths, 0.1, seed=1)
    assert len(test) == 10 and not set(train) & set(test)


def test_split_keeps_one_test_item_for_tiny_sets():
    train, test = split_paths(["a", "b", "c"], 0.01, seed=0)
    assert len(test) == 1 and len(train) == 2


@pytest.mark.parametrize("training", [True, False])
def test_make_dataset_batches(data_dir, training):
    cfg = TrainConfig(data_dir=data_dir, image_size=32, load_size=40, batch_size=2)
    photos, comics = list_domain_files(data_dir)
    photo_batch, comic_batch = next(iter(make_dataset(photos, comics, cfg, training)))
    for batch in (photo_batch, comic_batch):
        assert batch.shape == (2, 32, 32, 3)
        assert -1.0 <= float(np.min(batch)) and float(np.max(batch)) <= 1.0
