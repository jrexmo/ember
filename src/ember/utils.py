import os
import pathlib
import shutil
import typing
from functools import wraps
from typing import Callable

import cv2
import numpy as np
from absl import logging

DEFAULT_IMAGE_HEIGHT = 2000
DEFAULT_IMAGE_WIDTH = 4000

T = typing.TypeVar("T")
StorableData = np.ndarray | list[np.ndarray] | tuple[np.ndarray]


def reset_directory(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    logging.info(f"Cleared and recreated {directory}")


def filename_safe_time() -> str:
    import datetime

    return datetime.datetime.now().strftime("%H:%M:%S")


def create_image_from_contour(
    contour_data: np.ndarray, height: int, width: int
) -> np.ndarray:
    img = np.zeros((height, width), dtype=contour_data.dtype)
    cv2.drawContours(img, [contour_data], 0, (255), 2)
    return img


def store_data_as_image(
    data: StorableData, path: pathlib.Path, filename: str
) -> list[pathlib.Path]:
    match data:
        case np.ndarray():
            if data.ndim == 2 or (data.ndim == 3 and data.shape[2] in [1, 3, 4]):
                written_path = path / f"{filename}.png"
                cv2.imwrite(written_path, data)
            elif data.ndim == 3 and data.shape[2] == 2:
                img = create_image_from_contour(
                    data, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH
                )
                written_path = path / f"{filename}_contour.png"
                cv2.imwrite(written_path, img)
            else:
                raise ValueError(f"Unsupported shape {data.shape} for {path}")
            return [written_path]
        case list() | tuple():
            written_paths = []
            for i, data_holder in enumerate(data):
                written_paths += store_data_as_image(
                    data_holder, path, f"{filename}_{i}"
                )
            return written_paths
        case _:
            raise ValueError(f"Unsupported type {type(data)} for {path=}, {filename=}")


def capture_function_output(
    func: Callable[[T], StorableData]
) -> Callable[[T], StorableData]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        directory = os.getenv("DEBUG_DIRECTORY")
        result = func(*args, **kwargs)
        if not os.path.exists(directory):
            os.makedirs(directory)
        target_path = pathlib.Path(directory)

        written_paths = store_data_as_image(
            result, target_path, f"{filename_safe_time()}-{func.__name__}"
        )
        logging.info(
            f"Capturing output of {func.__name__} to {[str(p) for p in written_paths]}"
        )
        return result

    return wrapper
