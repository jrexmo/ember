import cv2
import pathlib
import numpy as np
import pyembroidery
from absl import logging

from ember import utils

type Contour = np.ndarray
type Image = np.ndarray


@utils.capture_function_output
def read_image(image_path: str) -> Image:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


@utils.capture_function_output
def detect_edges(image: Image, threshold1: int, threshold2: int) -> Image:
    return cv2.Canny(image, threshold1, threshold2)


@utils.capture_function_output
def find_contours(edges: np.ndarray) -> list[Contour]:
    # Discard the contour heiararchy
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def create_pattern(color: str) -> pyembroidery.EmbPattern:
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


def save_pattern(pattern: pyembroidery.EmbPattern, output_path: pathlib.Path) -> None:
    pyembroidery.write_png(pattern, str(output_path))
    logging.info(f"Pattern created and saved as {output_path}")


def create_embroidery_from_image(
    image_path: pathlib.Path,
    output_path: pathlib.Path,
    threshold1: int = 100,
    threshold2: int = 200,
    color: str = "red",
) -> None:
    image = read_image(image_path)
    edges = detect_edges(image, threshold1, threshold2)
    contours = find_contours(edges)
    pattern = create_pattern(color)
    for contour in contours:
        add_contour_to_pattern(pattern, contour)

    last_x, last_y = contours[-1][-1][0] if contours else (0, 0)
    terminate_pattern(pattern, last_x, last_y)

    save_pattern(pattern, output_path)


def main():

    data_directory = pathlib.Path(os.getenv("DATA_DIRECTORY"))
    create_embroidery_from_image(
        data_directory / "input_image.jpg",
        data_directory / "output.png",
        threshold1=100,
        threshold2=200,
        color="blue",
    )


if __name__ == "__main__":
    import os

    import dotenv

    dotenv.load_dotenv()

    logging.set_verbosity(logging.INFO)

    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    main()
