"""
Module 3 - Embedding + Similarity
Two functions, both wrapped as RunnableLambda for LangChain pipeline use.

Install:
    pip install sentence-transformers torch langchain-core anthropic

Usage:
    from module3 import embed_chain, similarity_chain, module3_chain
"""

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnableLambda

# ── Model (loaded once) ───────────────────────────────────────────────────────
_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: Embed canonical_text in a dict
# ─────────────────────────────────────────────────────────────────────────────
def embed_canonical_text(input_dict: dict) -> dict:
    """
    Input:
        {
            "canonical_text": "Drinking bleach cures COVID-19.",
            ... any other fields (passed through unchanged) ...
        }

    Output:
        {
            "canonical_text": "Drinking bleach cures COVID-19.",
            "embedding": np.ndarray of shape [384],
            ... original fields ...
        }
    """
    if "canonical_text" not in input_dict:
        raise ValueError("Input dict must contain a 'canonical_text' field.")

    text = input_dict["canonical_text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("'canonical_text' must be a non-empty string.")

    embedding = _EMBED_MODEL.encode(text, normalize_embeddings=True)  # L2-normalized

    return {**input_dict, "embedding": embedding}


# RunnableLambda wrapper
embed_chain = RunnableLambda(embed_canonical_text)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: Similarity check between two embedded dicts
# ─────────────────────────────────────────────────────────────────────────────
def compute_similarity(input_dict: dict) -> dict:
    """
    Input:
        {
            "rumor":  { "canonical_text": "...", "embedding": np.ndarray },
            "source": { "canonical_text": "...", "embedding": np.ndarray }
        }

    Output:
        {
            "rumor":            { ... },
            "source":           { ... },
            "similarity_score": float (0.0 – 1.0),
            "similarity_label": "HIGH" | "MEDIUM" | "LOW"
        }
    """
    rumor_dict  = input_dict.get("rumor")
    source_dict = input_dict.get("source")

    if rumor_dict is None or source_dict is None:
        raise ValueError("Input must contain both 'rumor' and 'source' keys.")

    if "embedding" not in rumor_dict or "embedding" not in source_dict:
        raise ValueError(
            "Both dicts must have an 'embedding' field. "
            "Run embed_canonical_text() on them first."
        )

    def _to_tensor(emb):
        if isinstance(emb, torch.Tensor):
            return emb.float()
        return torch.tensor(np.array(emb), dtype=torch.float32)

    v1 = F.normalize(_to_tensor(rumor_dict["embedding"]).unsqueeze(0),  dim=1)
    v2 = F.normalize(_to_tensor(source_dict["embedding"]).unsqueeze(0), dim=1)

    score = round(F.cosine_similarity(v1, v2).item(), 4)

    if score >= 0.75:
        label = "HIGH"
    elif score >= 0.45:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "rumor":            rumor_dict,
        "source":           source_dict,
        "similarity_score": score,
        "similarity_label": label,
    }


# RunnableLambda wrapper
similarity_chain = RunnableLambda(compute_similarity)


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED: embed both + similarity in one shot
# ─────────────────────────────────────────────────────────────────────────────
def embed_both_and_compare(input_dict: dict) -> dict:
    """
    Convenience function — embeds both dicts then computes similarity.

    Input:
        {
            "rumor":  { "canonical_text": "..." },
            "source": { "canonical_text": "..." }
        }

    Output: same as compute_similarity()
    """
    embedded = {
        "rumor":  embed_canonical_text(input_dict["rumor"]),
        "source": embed_canonical_text(input_dict["source"]),
    }
    return compute_similarity(embedded)


module3_chain = RunnableLambda(embed_both_and_compare)


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Test 1: embed a single dict ──
    print("=" * 50)
    print("TEST 1: Single embed")
    result = embed_chain.invoke({
        "canonical_text": "Drinking hot water every hour kills coronavirus.",
        "source": "whatsapp",
    })
    print("Keys    :", list(result.keys()))
    print("Shape   :", result["embedding"].shape)
    print("Norm    :", round(np.linalg.norm(result["embedding"]), 4))

    # ── Test 2: similarity between two dicts ──
    print("\n" + "=" * 50)
    print("TEST 2: Similarity (similar claims)")
    out = module3_chain.invoke({
        "rumor":  {"canonical_text": "Garlic cures COVID-19."},
        "source": {"canonical_text": "Eating garlic prevents coronavirus infection."},
    })
    print("Score   :", out["similarity_score"])
    print("Label   :", out["similarity_label"])

    print("\n" + "=" * 50)
    print("TEST 3: Similarity (unrelated claims)")
    out2 = module3_chain.invoke({
        "rumor":  {"canonical_text": "5G towers spread COVID-19."},
        "source": {"canonical_text": "Vaccines are safe and effective per WHO guidelines."},
    })
    print("Score   :", out2["similarity_score"])
    print("Label   :", out2["similarity_label"])
