import numpy as np


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0

    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def similarity_check(claim, clusters, threshold=0.75):
    """
    Compare claim embedding with all cluster centroids.

    Returns:
        {
            "score": best_score,
            "cluster_id": best_cluster_id (0 if no match)
        }
    """

    claim_embedding = claim.get("embedding")

    if not claim_embedding:
        return {
            "score": 0.0,
            "cluster_id": 0
        }

    best_score = -1.0
    best_cluster_id = 0

    for cluster in clusters:
        cluster_embedding = cluster.get("embedding_parent")

        if not cluster_embedding:
            continue

        score = cosine_similarity(claim_embedding, cluster_embedding)

        if score > best_score:
            best_score = score
            best_cluster_id = cluster["id"]

    # Apply threshold
    if best_score >= threshold:
        return {
            "score": best_score,
            "cluster_id": best_cluster_id
        }
    else:
        return {
            "score": best_score,
            "cluster_id": 0  # no match
        }