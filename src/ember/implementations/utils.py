"""Directory management and image processing utilities for Ember."""

import datetime
import pyembroidery
import io
import logging
import os
import pathlib
from sklearn.cluster import KMeans

import shutil
from collections.abc import Iterable
from functools import wraps

import cv2
from PIL import Image
import numpy as np

from dataclasses import dataclass

DEFAULT_IMAGE_HEIGHT = 2000
DEFAULT_IMAGE_WIDTH = 4000
CANNY_MIN_THRESHOLD = 100
CANNY_MAX_THRESHOLD = 200


logger = logging.getLogger(__name__)

type ContourData = np.ndarray
type ImageData = np.ndarray
DataType = ContourData | ImageData
type StorableData = DataType | Iterable[DataType]


@dataclass
class ImageShape:
    height: int
    width: int


def filename_safe_time() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")


def prefix_with_time(s: str) -> str:
    return f"{filename_safe_time()}-{s}"


def store_data_as_image(
    data: StorableData, path: pathlib.Path, filename: str
) -> list[pathlib.Path]:
    if isinstance(data, np.ndarray):
        if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in [1, 3, 4]):
            written_path = path / f"{filename}.png"
            cv2.imwrite(written_path, data)
        elif data.ndim == 3 and data.shape[2] == 2:
            img = image_from_contours(data, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
            written_path = path / f"{filename}_contour.png"
            cv2.imwrite(written_path, img)
        else:
            raise ValueError(f"Unsupported shape {data.shape} for {path}")
        return [written_path]
    if isinstance(data, Iterable):
        written_paths: list[pathlib.Path] = []
        for i, data_holder in enumerate(data):
            written_paths += store_data_as_image(data_holder, path, f"{filename}_{i}")
        return written_paths


def capture_function_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        directory = os.getenv("DEBUG_DIRECTORY")
        result = func(*args, **kwargs)
        if not os.path.exists(directory):
            os.makedirs(directory)
        target_path = pathlib.Path(directory)

        written_paths = store_data_as_image(
            result, target_path, prefix_with_time(f"{func.__module__}.{func.__name__}")
        )
        logger.info(
            f"Capturing output of {func.__name__} to {[str(p) for p in written_paths]}"
        )
        return result

    return wrapper


# IMAGE UTILS
def image_shape(image: np.ndarray) -> ImageShape:
    return ImageShape(*image.shape[:2])


@capture_function_output
def image_to_palette(image: ImageData, num_colors: int) -> ImageData:
    shape = image_shape(image)

    # Extract the color palette from the image
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42)
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_.astype(int)

    # Convert all pixels in the image to the closest color in the palette
    flat_image = image.reshape(-1, 3)
    labels = kmeans.predict(flat_image)
    return palette[labels].reshape(shape.height, shape.width, 3)


def reset_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.info(f"Cleared and recreated {directory}")


@capture_function_output
def read_image(image_path: str) -> ImageData:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


@capture_function_output
def opencv_img_from_buffer(
    buffer: io.BufferedReader | bytes, flags=cv2.IMREAD_COLOR
) -> ImageData:
    match buffer:
        case io.BufferedReader():
            bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
        case bytes():
            bytes_as_np_array = np.frombuffer(buffer, dtype=np.uint8)
        case _:
            raise ValueError("Invalid buffer type")
    return cv2.imdecode(bytes_as_np_array, flags)


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image to an OpenCV image (BGR format).

    Args:
    - pil_image (Image.Image): The PIL Image to convert.

    Returns:
    - np.ndarray: The converted OpenCV image (BGR format).
    """
    # Convert PIL image to RGB
    rgb_image = np.array(pil_image.convert("RGB"))

    # Convert RGB to BGR
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    return bgr_image


def cv2_to_pil(cv_image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV image (BGR format) to a PIL Image.

    Args:
    - cv_image (np.ndarray): The OpenCV image to convert (BGR format).

    Returns:
    - Image.Image: The converted PIL Image.
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)

    return pil_image


@capture_function_output
def detect_edges(image: ImageData) -> ImageData:
    return cv2.Canny(image, CANNY_MIN_THRESHOLD, CANNY_MAX_THRESHOLD)


@capture_function_output
def find_contours(edges: ContourData) -> list[ContourData]:
    # Discard the contour heiararchy
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def image_from_contours(
    contour_data: np.ndarray, height: int, width: int
) -> np.ndarray:
    img = np.zeros((height, width), dtype=contour_data.dtype)
    cv2.drawContours(img, [contour_data], 0, (255), 2)
    return img


# PATTERN UTILS
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
