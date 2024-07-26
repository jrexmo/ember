import io
import itertools as it
import numpy as np

import cv2
import pyembroidery

from ember.implementations import pipeline, utils

def create_embroidery_sweeping(
    data: io.BufferedReader | bytes,
    output_buffer: io.BufferedWriter,
    num_colors: int = 3,
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
    color_masks = utils.create_color_masks(palette_image)

    blocks = {}
    max_stitch_length = 5
    for color in color_masks:
        masked_image = color_masks[color]
        block = []
        i = 0
        while (1 in masked_image.flatten()):
            print(f"color:{color}\tpass:{i}\tones remaining:{np.count_nonzero(masked_image.flatten() == 1)}")
            i += 1
            for y in range(0, shape.height):
                if not 1 in masked_image[y]:
                    continue
                left_to_right = y % 2 == 0
                stitches_count = 0
                last_stitch_x = 0
                for x in range(0, shape.width):
                    if (stitches_count > 0 and 
                        (x == (shape.width - 1) or masked_image[y, x] == 0)):
                        if left_to_right:
                            block.append((x, y))
                        else:
                            block.insert(len(block) - stitches_count, (x, y))
                        masked_image[y][last_stitch_x:x + 1] = 0
                        stitches_count += 1
                        last_stitch_x = x
                        break
                    elif (masked_image[y, x] == 1 and
                          (stitches_count == 0 or
                           x - last_stitch_x == max_stitch_length)):
                        if left_to_right:
                            block.append((x, y))
                        else:
                            block.insert(len(block) - stitches_count, (x, y))
                        masked_image[y][last_stitch_x:x + 1] = 0
                        stitches_count += 1
                        last_stitch_x = x
        blocks[color] = block

    pattern = pyembroidery.EmbPattern()
    sorted_colors = sorted((k for k, _ in blocks.items()), reverse=True)
    for color in sorted_colors:
        pattern.add_block(blocks[color], color)

    utils.save_pattern(pattern, output_buffer)


if __name__ == "__main__":
    pipeline.run(create_embroidery_sweeping)
