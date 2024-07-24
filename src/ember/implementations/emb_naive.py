"""Embroidery creation logic.

This module exposes a function that accepts an image, creates an embroidery pattern from it, and saves this pattern as a PNG image.

## concepts
- https://edutechwiki.unige.ch/en/Embroidery_format_DST

## code
- https://github.com/EmbroidePy/pyembroidery"""

import io

import cv2

from ember.implementations import pipeline, utils

DEFAULT_PATTERN_COLOR = "blue"


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
    pattern = utils.new_pattern(DEFAULT_PATTERN_COLOR)
    for contour in contours:
        utils.add_contour_to_pattern(pattern, contour)

    last_x, last_y = contours[-1][-1][0] if contours else (0, 0)
    utils.terminate_pattern(pattern, last_x, last_y)

    utils.save_pattern(pattern, output_buffer)


if __name__ == "__main__":
    pipeline.run(create_embroidery_naive)
