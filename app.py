"""Streamlit web app: turn a face photo into a comic portrait (or back).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import random
from pathlib import Path

import streamlit as st
from PIL import Image

from comicgan.config import (
    COMIC2PHOTO_FILE,
    COMIC_SUBDIR,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_DIR,
    PHOTO2COMIC_FILE,
    PHOTO_SUBDIR,
)

MODEL_DIR = Path(os.environ.get("COMICGAN_MODEL_DIR", DEFAULT_MODEL_DIR))
DATA_DIR = Path(os.environ.get("COMICGAN_DATA_DIR", DEFAULT_DATA_DIR))

DIRECTIONS = {
    "Photo → Comic": (PHOTO2COMIC_FILE, PHOTO_SUBDIR),
    "Comic → Photo": (COMIC2PHOTO_FILE, COMIC_SUBDIR),
}

st.set_page_config(page_title="Real to Comic Photo", page_icon="🎨", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def get_generator(path: str, modified: float):
    # `modified` is part of the cache key so a retrained model is picked up.
    from comicgan.inference import load_generator

    return load_generator(path)


def list_examples(subdir: str, count: int = 200) -> list[Path]:
    folder = DATA_DIR / subdir
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.jpg"))[:count]


def pick_input(example_subdir: str) -> Image.Image | None:
    upload_tab, camera_tab, example_tab = st.tabs(["Upload", "Camera", "Example"])

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"]
        )
    with camera_tab:
        snapshot = st.camera_input("Take a photo")
    with example_tab:
        examples = list_examples(example_subdir)
        if not examples:
            st.info(f"No example images found in `{DATA_DIR / example_subdir}`.")
        elif st.button("Pick a random example"):
            st.session_state["example"] = str(random.choice(examples))

    source = uploaded or snapshot
    if source is not None:
        return Image.open(source)
    example = st.session_state.get("example")
    if example and Path(example).parent.name == example_subdir:
        return Image.open(example)
    return None


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.title("🎨 Real to Comic Photo")
    st.caption(
        "A CycleGAN trained on the Face2Comics dataset. Works best on a single, "
        "front-facing face that fills most of the frame."
    )

    with st.sidebar:
        st.header("Settings")
        direction = st.radio("Direction", list(DIRECTIONS))
        size = st.select_slider(
            "Output size (px)",
            options=[128, 256, 384, 512],
            value=256,
            help="The model was trained on 128 px crops; larger sizes are slower "
            "and may show more artifacts.",
        )
        crop_square = st.toggle(
            "Center-crop to square",
            value=True,
            help="Training images are square face crops.",
        )

    model_file, example_subdir = DIRECTIONS[direction]
    model_path = MODEL_DIR / model_file
    if not model_path.is_file():
        st.warning(
            f"Model file `{model_path}` not found. Train the model first:\n\n"
            "```bash\npython -m comicgan.train --epochs 50\n```\n\n"
            "or point `COMICGAN_MODEL_DIR` at a folder containing "
            f"`{PHOTO2COMIC_FILE}` and `{COMIC2PHOTO_FILE}`."
        )
        st.stop()

    image = pick_input(example_subdir)
    if image is None:
        st.info("Upload an image, take a photo, or pick an example to get started.")
        return

    from comicgan.inference import translate

    generator = get_generator(str(model_path), model_path.stat().st_mtime)
    with st.spinner("Generating…"):
        result = translate(generator, image, size=size, crop_square=crop_square)

    left, right = st.columns(2)
    left.image(image, caption="Input", width="stretch")
    right.image(result, caption=direction.split(" → ")[1], width="stretch")
    right.download_button(
        "Download result",
        data=to_png_bytes(result),
        file_name=f"{'comic' if model_file == PHOTO2COMIC_FILE else 'photo'}.png",
        mime="image/png",
    )


main()
