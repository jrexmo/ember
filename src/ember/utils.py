"""Directory management and image processing utilities for Ember."""

import datetime
import io
import logging
import os
import pathlib
import shutil
from collections.abc import Iterable
from functools import wraps

import cv2
import numpy as np

from ember.ember_types import Contour, Image

DEFAULT_IMAGE_HEIGHT = 2000
DEFAULT_IMAGE_WIDTH = 4000
CANNY_MIN_THRESHOLD = 100
CANNY_MAX_THRESHOLD = 200


DataType = Contour | Image
type StorableData = DataType | Iterable[DataType]

logger = logging.getLogger(__name__)


def reset_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    logger.info(f"Cleared and recreated {directory}")


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


@capture_function_output
def read_image(image_path: str) -> Image:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


@capture_function_output
def opencv_img_from_buffer(buffer: io.BufferedReader | bytes, flags) -> Image:
    match buffer:
        case io.BufferedReader():
            bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
        case bytes():
            bytes_as_np_array = np.frombuffer(buffer, dtype=np.uint8)
        case _:
            raise ValueError("Invalid buffer type")
    return cv2.imdecode(bytes_as_np_array, flags)


@capture_function_output
def detect_edges(image: Image) -> Image:
    return cv2.Canny(image, CANNY_MIN_THRESHOLD, CANNY_MAX_THRESHOLD)


@capture_function_output
def find_contours(edges: Contour) -> list[Contour]:
    # Discard the contour heiararchy
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def image_from_contours(
    contour_data: np.ndarray, height: int, width: int
) -> np.ndarray:
    img = np.zeros((height, width), dtype=contour_data.dtype)
    cv2.drawContours(img, [contour_data], 0, (255), 2)
    return img
