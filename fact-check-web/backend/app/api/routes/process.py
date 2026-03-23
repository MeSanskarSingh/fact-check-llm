from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.pipeline_service import run_pipeline

router = APIRouter()

@router.post("/")
async def process_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = run_pipeline(temp_path)

    os.remove(temp_path)

    return result