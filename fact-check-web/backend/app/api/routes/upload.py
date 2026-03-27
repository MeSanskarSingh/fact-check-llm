from fastapi import APIRouter, UploadFile, File, HTTPException
from app.utils.file_handler import save_upload_file, delete_file
from app.services.preprocessing_service import preprocess_input
from app.services.validation_service import get_validation_service
from app.services.clustering_service import run_clustering
from app.models.schemas import ProcessResponse

router = APIRouter()


@router.post("/", response_model=ProcessResponse)
async def upload_and_process(file: UploadFile = File(...)):
    file_path = None

    try:
        # Save file
        file_path = save_upload_file(file)

        # Preprocess
        text = preprocess_input(file_path)

        # Validate
        validation = get_validation_service().validate_text(text)

        # Cluster
        cluster_id = run_clustering(text)

        return ProcessResponse(
            verdict=validation["verdict"],
            confidence=validation["confidence"],
            extractedText=text,
            explanation=validation["explanation"],
            cluster=cluster_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if file_path:
            delete_file(file_path)