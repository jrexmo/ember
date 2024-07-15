"""Ember web interface.

This webapp exposes the embroidery functionality defined elsewhere in the Ember project.
"""

import logging
import os
import pathlib
import time
import base64
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

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

# Serve static files
app.mount("/static", StaticFiles(directory="src/ember/webapp/templates"), name="static")


def delete_files_after_delay(files: list[pathlib.Path], delay=DEFAULT_DELETE_DELAY):
    time.sleep(delay)
    for file in files:
        file.unlink(missing_ok=True)


@app.post("/embroidery/")
async def create_embroidery(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    input_path = INPUT_DIR / file.filename
    output_filename = f"{pathlib.Path(file.filename).stem}_embroidery.png"
    output_path = OUTPUT_DIR / output_filename

    with open(input_path, "wb") as input_buffer:
        input_buffer.write(await file.read())
    with open(output_path, "wb") as output_buffer:
        create_embroidery_naive(open(input_path, "rb"), output_buffer)

    background_tasks.add_task(delete_files_after_delay, [input_path, output_path])

    # Check if it's an HTMX request
    if request.headers.get("HX-Request") == "true":
        # Read the image file and encode it as base64
        with open(output_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

        # Return an HTML response with the embedded image
        html_response = (
            f'<img src="data:image/png;base64,{img_data}" alt="Embroidered Image">'
        )
        return HTMLResponse(content=html_response)
    else:
        # Return the file as usual for non-HTMX requests
        return FileResponse(
            str(output_path), media_type="image/png", filename=output_filename
        )


@app.get("/debug")
async def debug_info():
    print(list(INPUT_DIR.glob("*")))
    print(list(OUTPUT_DIR.glob("*")))

    return JSONResponse(
        {
            "temp_dir_exists": TEMP_DIR.exists(),
            "input_dir_exists": INPUT_DIR.exists(),
            "output_dir_exists": OUTPUT_DIR.exists(),
            "input_files": [
                {"name": f.name, "url": f"/static/input/{f.name}"}
                for f in INPUT_DIR.glob("*")
                if f.is_file()
            ],
            "output_files": [
                {"name": f.name, "url": f"/static/output/{f.name}"}
                for f in OUTPUT_DIR.glob("*")
                if f.is_file()
            ],
        }
    )


# TODO: Convert this to a Jinja rendered template.
@app.get("/")
async def read_root():
    return FileResponse("src/ember/webapp/templates/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
