"""Embroidery creation logic.

This module exposes a function that accepts an image, creates an embroidery pattern from it, and saves this pattern as a PNG image.

## concepts
- https://edutechwiki.unige.ch/en/Embroidery_format_DST

## code
- https://github.com/EmbroidePy/pyembroidery"""

import cv2
import pathlib
import numpy as np
import pyembroidery
import io

from ember import utils

DEFAULT_PATTERN_COLOR = "blue"


def new_pattern(color: str) -> pyembroidery.EmbPattern:
    pattern = pyembroidery.EmbPattern()
    pattern.add_thread({"color": color})
    return pattern


def add_contour_to_pattern(
    pattern: pyembroidery.EmbPattern, contour: np.ndarray
) -> None:
    x, y = contour[0][0]
    pattern.add_stitch_absolute(pyembroidery.JUMP, x, y)

    for point in contour[1:]:
        x, y = point[0]
        pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)


def terminate_pattern(pattern: pyembroidery.EmbPattern, x: int, y: int) -> None:
    pattern.add_stitch_absolute(pyembroidery.END, x, y)


def save_pattern(
    pattern: pyembroidery.EmbPattern, output_buffer: io.BufferedWriter
) -> None:
    pyembroidery.write_png(pattern, output_buffer)


def create_embroidery_naive(
    image: io.BufferedReader | bytes,
    output_buffer: io.BufferedWriter,
) -> None:
    image = utils.opencv_img_from_buffer(image, cv2.IMREAD_ANYCOLOR)
    edges = utils.detect_edges(image)
    contours = utils.find_contours(edges)
    pattern = new_pattern(DEFAULT_PATTERN_COLOR)
    for contour in contours:
        add_contour_to_pattern(pattern, contour)

    last_x, last_y = contours[-1][-1][0] if contours else (0, 0)
    terminate_pattern(pattern, last_x, last_y)

    save_pattern(pattern, output_buffer)


def main():
    data_directory = pathlib.Path(os.getenv("DATA_DIRECTORY"))
    if not (path := data_directory / "image.jpg").exists():
        raise ValueError(
            f"Please provide an example image in the data directory at {path}.\n You can modify the image name in embroidery.py main()"
        )
    with (
        open(data_directory / "image.jpg", "rb") as input_buffer,
        open(data_directory / "output.png", "wb") as output_buffer,
    ):
        create_embroidery_naive(
            input_buffer,
            output_buffer,
        )


if __name__ == "__main__":
    import os
    import logging

    import dotenv

    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO)

    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    main()
