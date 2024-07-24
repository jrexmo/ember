import io
import pathlib
import itertools as it
import random

import cv2
import pyembroidery

from ember.implementations import pipeline, utils


def create_embroidery_sweeping(
    data: io.BufferedReader | bytes,
    output_buffer: io.BufferedWriter,
    num_colors: int = 5,
) -> None:
    """
    Create an embroidery pattern from an image using a sweeping approach.

    Args:
        data: The input image data.
        output_buffer: The output buffer to write the embroidery pattern to.
        num_colors: The number of colors to use in the palette.

    Returns:
        None
    """
    image = utils.opencv_img_from_buffer(data, cv2.IMREAD_COLOR)
    shape = utils.image_shape(image)
    palette_image = utils.image_to_palette(image, num_colors)
    print(palette_image[0, 0])

    pattern = pyembroidery.EmbPattern()
    for y in range(0, shape.height, 2):
        groups = it.groupby(palette_image[y], key=lambda x: (x[0], x[1], x[2]))
        x_offset = 0
        for color, group in groups:
            hex = utils.rgb_to_hex(color)
            group = list(group)
            print(f"Drawing {hex} at {y=} with length {len(group)}")
            pattern.add_block(
                [(x + x_offset, y) for x in range(0, len(group), 3)],
                hex,
            )
            x_offset += len(group)

    utils.save_pattern(pattern, output_buffer)
    pyembroidery.write(pattern, "data/output.dst")


if __name__ == "__main__":
    pipeline.run(create_embroidery_sweeping)
