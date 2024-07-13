import logging
import os
import pathlib
from contextlib import asynccontextmanager

import dotenv
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ember import utils
from ember.embroidery import create_embroidery_from_image

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    utils.reset_directory(os.getenv("DEBUG_DIRECTORY"))
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/embroidery/")
async def create_embroidery(
    file: UploadFile = File(...),
):
    temp_dir = pathlib.Path("/tmp/embroidery")
    temp_dir.mkdir(exist_ok=True)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    input_path = input_dir / file.filename
    output_filename = f"{pathlib.Path(file.filename).stem}_embroidery.png"
    output_path = output_dir / output_filename

    try:
        with open(output_path, "wb") as output_buffer:
            create_embroidery_from_image(await file.read(), output_buffer)
        return FileResponse(
            str(output_path), media_type="image/png", filename=output_filename
        )
    finally:
        input_path.unlink(missing_ok=True)
        # output_path.unlink(missing_ok=True)


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
