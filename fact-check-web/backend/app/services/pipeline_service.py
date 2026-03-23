from ml_pipeline.preprocessing.PreprocessingScript import EnhancedFactCheckPreprocessor
from ml_pipeline.clustering.cluster_manager import update_clusters

# you’ll import your validation logic properly here

def run_pipeline(input_data):
    preprocessor = EnhancedFactCheckPreprocessor()

    # Step 1: preprocess
    processed = preprocessor.process_input(input_data)

    if processed.status != "success":
        return {"error": processed.error_message}

    text = processed.text

    # Step 2: validation (TEMP mock if notebook not converted)
    validation_result = {
        "verdict": "contradicted",
        "confidence": 0.82,
        "explanation": "Claim not supported by evidence"
    }

    # Step 3: clustering (optional for now)
    chain_result = {
        "plain_json": [
            {
                "pass_rest": {
                    "claim": {
                        "canonical_text": text,
                        "embedding": [0.1, 0.2]  # TEMP
                    }
                }
            }
        ],
        "validation": None
    }

    clusters, summary = update_clusters(chain_result)

    # FINAL OUTPUT (what frontend needs)
    return {
        "verdict": validation_result["verdict"],
        "confidence": validation_result["confidence"],
        "extractedText": text,
        "explanation": validation_result["explanation"],
        "cluster": summary[0]["cluster_id"] if summary else None
    }