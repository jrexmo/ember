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
    Converts an image to an embroidery with a series of horizontal lines.

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

    blocks = []
    for y in range(0, shape.height, 3):
        groups = it.groupby(palette_image[y], key=lambda x: (x[0], x[1], x[2]))
        x_offset = 0
        for color, grouped_blocks in groups:
            hex = utils.rgb_to_hex(color)
            grouped_blocks = list(grouped_blocks)
            print(f"Drawing {hex} at {y=} with length {len(grouped_blocks)}")
            blocks.append(
                ([(x + x_offset, y) for x in range(0, len(grouped_blocks), 3)], hex)
            )
            x_offset += len(grouped_blocks)

    # sort by color, then by y, then by x
    # blocks[block_index][0] = block
    # blocks[block_index][0][stitch_index] = (x, y)
    # blocks[block_index][1] = color
    blocks = sorted(blocks, key=lambda x: (x[1], x[0][0][1], x[0][0][0]))

    pattern = pyembroidery.EmbPattern()
    for color, grouped_blocks in it.groupby(blocks, key=lambda x: x[1]):
        for block in grouped_blocks:
            # discard color since we have it as the key
            block, _ = block
            pattern.add_block(block, color)

    utils.save_pattern(pattern, output_buffer)


if __name__ == "__main__":
    pipeline.run(create_embroidery_sweeping)
