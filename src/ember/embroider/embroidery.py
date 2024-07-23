"""Embroidery creation logic.

This module exposes a function that accepts an image, creates an embroidery pattern from it, and saves this pattern as a PNG image.

## concepts
- https://edutechwiki.unige.ch/en/Embroidery_format_DST

## code
- https://github.com/EmbroidePy/pyembroidery"""

import io
import pathlib

import cv2
import numpy as np
import pyembroidery

from ember.embroider import utils

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
    pattern: pyembroidery.EmbPattern, output: io.BufferedWriter | pathlib.Path
) -> None:
    match output:
        case io.BufferedWriter():
            pyembroidery.write_png(pattern, output)
        case pathlib.Path():
            with open(output, "wb") as output:
                pyembroidery.write_png(pattern, output)


def create_embroidery_naive(
    data: io.BufferedReader | bytes,
    output_buffer: io.BufferedWriter,
) -> None:
    """
    First pass to create an embroidery pattern from an image.

    This function reads an image, detects its edges, finds contours, and creates an embroidery pattern from these contours.

    Args:
        data: The input image data.
        output_buffer: The output buffer to write the embroidery pattern to

    Returns:
        None
    """
    image = utils.opencv_img_from_buffer(data, cv2.IMREAD_ANYCOLOR)
    edges = utils.detect_edges(image)
    contours = utils.find_contours(edges)
    pattern = new_pattern(DEFAULT_PATTERN_COLOR)
    for contour in contours:
        add_contour_to_pattern(pattern, contour)

    last_x, last_y = contours[-1][-1][0] if contours else (0, 0)
    terminate_pattern(pattern, last_x, last_y)

    save_pattern(pattern, output_buffer)


import cv2
import numpy as np
import pyembroidery
from sklearn.cluster import KMeans
from ember.embroider import utils
import random


def create_embroidery_sweeping(
    data: io.BufferedReader | bytes,
    output_buffer: io.BufferedWriter,
    color_threshold: int = 30,
    stitch_length: int = 10,
    num_colors: int = 5,
) -> None:
    """
    Create an embroidery pattern from an image using a sweeping approach.

    Args:
        data: The input image data.
        output_buffer: The output buffer to write the embroidery pattern to.
        color_threshold: The threshold for color difference to start a new stitch.
        stitch_length: The maximum length of a single stitch.
        num_colors: The number of colors to use in the palette.

    Returns:
        None
    """
    image = utils.opencv_img_from_buffer(data, cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    pattern = pyembroidery.EmbPattern()

    # Extract the color palette from the image
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)

    # Convert all pixels in the image to the closest color in the palette
    flat_image = image.reshape(-1, 3)
    labels = kmeans.predict(flat_image)
    quantized_image = palette[labels].reshape(height, width, 3)
    utils.store_data_as_image(
        quantized_image, pathlib.Path("./data"), "quantized_image.png"
    )

    # Move from left to right, top to bottom on the image
    current_color = None
    for y in range(height):
        stitch_start = None

        for x in range(width):
            pixel_color = tuple(quantized_image[y, x])

            if current_color is None:
                # Start a new stitch with a new color
                if stitch_start is not None:
                    pattern.add_stitch_absolute(
                        pyembroidery.STITCH, stitch_start[0], stitch_start[1]
                    )

                current_color = pixel_color

                # Add a color change
                pattern += random.choice(["red", "green", "blue", "yellow", "black"])

                pattern.add_stitch_absolute(pyembroidery.JUMP, x, y)
                pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)
                stitch_start = (x, y)
            elif stitch_start and (x - stitch_start[0] >= stitch_length):
                # Continue the stitch
                pattern.add_stitch_absolute(pyembroidery.STITCH, x, y)
                stitch_start = (x, y)

        # Terminate the stitch at the end of each row
        if stitch_start is not None:
            pattern.add_stitch_absolute(pyembroidery.STITCH, width - 1, y)

    # Terminate the pattern
    pattern.add_stitch_absolute(pyembroidery.END, width - 1, height - 1)

    # Save the pattern
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
        create_embroidery_sweeping(
            input_buffer,
            output_buffer,
        )


if __name__ == "__main__":
    import logging
    import os

    import dotenv

    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO)

    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    main()
