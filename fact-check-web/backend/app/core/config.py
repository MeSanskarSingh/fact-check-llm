import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "FactCheckLLM"
    DEBUG = True

    # API Keys
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    # File upload
    UPLOAD_DIR = "uploads"
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

settings = Settings()