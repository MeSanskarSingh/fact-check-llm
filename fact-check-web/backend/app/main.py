from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.upload import router as upload_router
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)

app.include_router(upload_router, prefix="/process", tags=["Process"])