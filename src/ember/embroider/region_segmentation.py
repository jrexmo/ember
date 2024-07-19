"""Example implementation of region-based segmentation using watershed transform.

## concepts
- https://en.wikipedia.org/wiki/Watershed_(image_processing)
- https://en.wikipedia.org/wiki/Sobel_operator

## code
- https://scikit-image.org/docs/stable/auto_examples/applications/plot_coins_segmentation.html
- https://docs.scipy.org/doc/scipy/reference/ndimage.html

## test implementations
- label2rgb: https://chatgpt.com/share/81a7c671-70fc-49ca-b044-67268e42c4ae
- binary_fill_holes: https://chatgpt.com/share/dc4e5b51-519c-4ce0-86d8-fd4251df5674
"""

import io
import logging
import os
import pathlib

import dotenv
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import segmentation
from skimage.color import label2rgb
from skimage.filters import sobel

from ember.embroider import utils


class Segmentation:
    def __init__(self, segmented_image, labeled_image, overlay_image):
        self.segmented_image = segmented_image
        self.labeled_image = labeled_image
        self.overlay_image = overlay_image


def region_based_segmentation(image: bytes) -> Segmentation:
    # Convert bytes to numpy array
    pil_image = Image.open(io.BytesIO(image))
    coins = np.array(pil_image.convert("L"))  # Convert to grayscale

    # Create elevation map using Sobel gradient
    elevation_map = sobel(coins)

    # Find markers of the background and the coins
    markers = np.zeros_like(coins)
    markers[coins < 30] = 1
    markers[coins > 150] = 2

    # Use watershed transform for segmentation
    segmentation_coins = segmentation.watershed(elevation_map, markers)

    # Post-process the segmentation
    segmentation_coins = ndi.binary_fill_holes(segmentation_coins - 1)
    labeled_coins, _ = ndi.label(segmentation_coins)

    # Create color overlay
    image_label_overlay = label2rgb(labeled_coins, image=coins, bg_label=0)

    # Convert numpy arrays back to PIL images
    segmented_image = Image.fromarray((segmentation_coins * 255).astype(np.uint8))
    labeled_image = Image.fromarray(
        (labeled_coins * 255 / labeled_coins.max()).astype(np.uint8)
    )
    overlay_image = Image.fromarray((image_label_overlay * 255).astype(np.uint8))

    return Segmentation(segmented_image, labeled_image, overlay_image)


def main():
    data_directory = pathlib.Path(os.getenv("DATA_DIRECTORY"))
    if not (path := data_directory / "image.jpg").exists():
        raise ValueError(
            f"Please provide an example image in the data directory at {path}.\n You can modify the image name in the main() function"
        )

    # Read the input image
    with open(data_directory / "image.jpg", "rb") as input_buffer:
        input_bytes = input_buffer.read()

    # Perform region-based segmentation
    result: Segmentation = region_based_segmentation(input_bytes)

    # Save the output images
    result.segmented_image.save(data_directory / "segmented_output.png")
    result.labeled_image.save(data_directory / "labeled_output.png")
    result.overlay_image.save(data_directory / "overlay_output.png")

    logging.info(f"Segmentation results saved in {data_directory}")


if __name__ == "__main__":
    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO)

    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    main()
