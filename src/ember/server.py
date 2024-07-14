import logging
import os
import pathlib
import time
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi import BackgroundTasks

from ember import utils
from ember.embroidery import create_embroidery_naive

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEMP_DIR = pathlib.Path(os.getenv("TEMP_DIRECTORY"))
INPUT_DIR = TEMP_DIR / "input"
OUTPUT_DIR = TEMP_DIR / "output"
DEFAULT_DELETE_DELAY = 10


@asynccontextmanager
async def lifespan(_: FastAPI):
    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    INPUT_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    yield
    # TODO: This cleanup is not executed.
    logger.info("Cleaning up temporary directories")
    INPUT_DIR.unlink(missing_ok=True)
    OUTPUT_DIR.unlink(missing_ok=True)
    TEMP_DIR.unlink(missing_ok=True)


app = FastAPI(lifespan=lifespan)


def delete_files_after_delay(files: list[pathlib.Path], delay=DEFAULT_DELETE_DELAY):
    time.sleep(delay)
    for file in files:
        file.unlink(missing_ok=True)


@app.post("/embroidery/")
async def create_embroidery(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    input_path = INPUT_DIR / file.filename
    output_filename = f"{pathlib.Path(file.filename).stem}_embroidery.png"
    output_path = OUTPUT_DIR / output_filename

    with open(input_path, "wb") as input_buffer:
        input_buffer.write(await file.read())
    with open(output_path, "wb") as output_buffer:
        create_embroidery_naive(open(input_path, "rb"), output_buffer)

    background_tasks.add_task(delete_files_after_delay, [input_path, output_path])
    return FileResponse(
        str(output_path), media_type="image/png", filename=output_filename
    )


@app.get("/debug")
async def debug_info():
    temp_dir = pathlib.Path("/tmp/embroidery")
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"

    return JSONResponse(
        {
            "temp_dir_exists": temp_dir.exists(),
            "input_dir_exists": input_dir.exists(),
            "output_dir_exists": output_dir.exists(),
            "input_files": [str(f) for f in input_dir.glob("*") if f.is_file()],
            "output_files": [str(f) for f in output_dir.glob("*") if f.is_file()],
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
