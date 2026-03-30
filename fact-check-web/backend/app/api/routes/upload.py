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
        if not text or not str(text).strip():
            text = "No text could be extracted from the image."

        # Validate
        validation = get_validation_service().validate_text(text)

        # Cluster
        cluster_id = run_clustering(text)

        safe_text = text.strip() if text else ""
        if not safe_text:
            safe_text = "No text could be extracted from the image."

        safe_explanation = validation.get("explanation", "").strip()
        if not safe_explanation:
            safe_explanation = "No explanation provided."

        safe_verdict = validation.get("verdict", "Uncertain")
        safe_confidence = float(validation.get("confidence", 0.5))

        return ProcessResponse(
            verdict=safe_verdict,
            confidence=safe_confidence,
            extractedText=safe_text,
            explanation=safe_explanation,
            cluster=cluster_id
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if file_path:
            delete_file(file_path)