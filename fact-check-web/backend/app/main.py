from fastapi import FastAPI
from app.api.routes.upload import router as upload_router

app = FastAPI()

app.include_router(upload_router, prefix="/process", tags=["Process"])