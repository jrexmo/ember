import io
import pathlib
import random

import cv2
import pyembroidery

from ember.implementations import pipeline, utils


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
    pattern = pyembroidery.EmbPattern()
    image = utils.opencv_img_from_buffer(data, cv2.IMREAD_COLOR)
    shape = utils.image_shape(image)
    palette_image = utils.image_to_palette(image, num_colors)

    # Move from left to right, top to bottom on the image
    current_color = None
    pattern.add_block([(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)], "red")
    pattern.add_block([(0, 0), (0, 50), (50, 50), (50, 0), (0, 0)], "blue")
    pattern.add_stitchblock

    # Save the pattern
    utils.save_pattern(pattern, output_buffer)


if __name__ == "__main__":
    pipeline.run(create_embroidery_sweeping)
