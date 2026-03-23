from ml_pipeline.clustering.cluster_manager import update_clusters


def run_clustering(text):
    # minimal structure required by your cluster manager
    chain_result = {
        "plain_json": [
            {
                "pass_rest": {
                    "claim": {
                        "canonical_text": text,
                        "embedding": [0.1, 0.2]  # TEMP placeholder
                    }
                }
            }
        ],
        "validation": None
    }

    clusters, summary = update_clusters(chain_result)

    if summary:
        return summary[0]["cluster_id"]

    return None