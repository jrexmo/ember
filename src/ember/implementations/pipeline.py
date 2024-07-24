import os
import pathlib
from ember.implementations import utils


def run(embroidery_fn):
    import logging
    import os

    import dotenv

    dotenv.load_dotenv()

    logging.basicConfig(level=logging.INFO)

    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    data_directory = pathlib.Path(os.getenv("DATA_DIRECTORY"))

    if not (path := data_directory / "image.jpg").exists():
        raise ValueError(
            f"Please provide an example image in the data directory at {path}.\n You can modify the image name in embroidery.py main()"
        )
    with (
        open(data_directory / "image.jpg", "rb") as input_buffer,
        open(data_directory / "output.png", "wb") as output_buffer,
    ):
        embroidery_fn(
            input_buffer,
            output_buffer,
        )
