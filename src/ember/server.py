import os
import pathlib
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from tempfile import NamedTemporaryFile
import uvicorn
from embroidery import create_embroidery_from_image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/embroidery/")
async def create_embroidery(
    file: UploadFile = File(...),
    threshold1: int = 100,
    threshold2: int = 200,
    color: str = "blue"
):
    # Create temporary directories for input and output
    temp_dir = pathlib.Path("/tmp/embroidery")
    temp_dir.mkdir(exist_ok=True)
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Save uploaded file
    input_path = input_dir / file.filename
    with open(input_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Generate output filename
    output_filename = f"{pathlib.Path(file.filename).stem}_embroidery.png"
    output_path = output_dir / output_filename

    logger.debug(f"Input path: {input_path}")
    logger.debug(f"Output path: {output_path}")

    try:
        # Create embroidery
        create_embroidery_from_image(
            input_path,
            output_path,
            threshold1=threshold1,
            threshold2=threshold2,
            color=color
        )

        # Check if the file was created
        if not output_path.exists():
            logger.error(f"Output file not created at {output_path}")
            raise FileNotFoundError(f"Output file not created at {output_path}")

        # Return the output file
        return FileResponse(str(output_path), media_type="image/png", filename=output_filename)
    except FileNotFoundError as e:
        logger.exception("File not found error")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error occurred")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        input_path.unlink(missing_ok=True)
        # Don't delete the output file here, as it might be needed for the response

@app.get("/debug")
async def debug_info():
    temp_dir = pathlib.Path("/tmp/embroidery")
    input_dir = temp_dir / "input"
    output_dir = temp_dir / "output"
    
    return JSONResponse({
        "temp_dir_exists": temp_dir.exists(),
        "input_dir_exists": input_dir.exists(),
        "output_dir_exists": output_dir.exists(),
        "input_files": [str(f) for f in input_dir.glob("*") if f.is_file()],
        "output_files": [str(f) for f in output_dir.glob("*") if f.is_file()],
    })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)