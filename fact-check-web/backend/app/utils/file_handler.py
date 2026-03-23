import os
import shutil
from uuid import uuid4

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_upload_file(upload_file):
    file_id = str(uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{upload_file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)