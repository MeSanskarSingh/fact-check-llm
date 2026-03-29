from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Literal, Optional, TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

import numpy as np
import pandas as pd
import ast
import asyncio
import json
import os

load_dotenv()

# ── tuneable constants ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85
CLUSTER_CSV_PATH     = "cluster.csv"

# ── models & stores ───────────────────────────────────────────────────────────
embed_model  = MistralAIEmbeddings()
json_model   = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
jsonmodel    = ChatMistralAI(model="mistral-small-latest", temperature=0.1)

vector_store = Chroma(
    embedding_function=embed_model,
    persist_directory="rumor_detection",
    collection_name="facts"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ── pydantic schemas ──────────────────────────────────────────────────────────
class Claim(BaseModel):
    claim_id: int = Field(
        description="Unique index of this claim inside the rumor"
    )
    claim: str = Field(
        description=(
            "Atomic factual proposition containing exactly ONE subject and ONE object. "
            "Do not copy the full rumor sentence. Split conjunctions into separate claims."
        )
    )
    claim_type: Literal[
        "health", "death", "policy", "event",
        "statistic", "relationship", "other"
    ] = Field(description="Strict category label")
    entities: list[str] = Field(
        min_length=2, max_length=2,
        description="Exactly two entities: subject and object"
    )
    time:     Optional[str] = Field(default=None, description="Explicit time reference if present")
    location: Optional[str] = Field(default=None, description="Explicit location reference if present")
    canonical_text: str = Field(
        description=(
            "Controlled identity sentence: <subject> <relation> <object> [context]. "
            "Must represent only this claim."
        )
    )


class ClaimLabel(BaseModel):
    verdict: Literal["supported", "contradicted", "conflicting", "insufficient"] = Field(
        description="Label for the claim at the same index"
    )

class ValidationOutput(BaseModel):
    results: list[ClaimLabel] = Field(
        description="List index corresponds exactly to claims list index"
    )

# ── state ─────────────────────────────────────────────────────────────────────
class RumorState(TypedDict):
    # input
    rumor:               str

    # after extract_claim
    claim:               Optional[dict]

    # after attach_embedding  (lives on state, NOT inside claim dict)
    embedding:           Optional[List[float]]

    # after similarity_check
    sim_score:           Optional[float]
    matched_cluster_id:  Optional[int]         # int on HIT, None on MISS

    # after rag_retrieve  (MISS path only)
    rag_docs:            Optional[List[Document]]

    # after validate  (MISS path only)
    plain_json:          Optional[dict]
    validation:          Optional[ValidationOutput]

    # after create_cluster  (MISS path only)
    new_cluster_id:      Optional[int]

# ── prompts ───────────────────────────────────────────────────────────────────
json_extract_prompt = ChatPromptTemplate.from_template("""
You are a structured fact extraction system.

Return ONLY valid JSON matching the schema exactly.
Do not add commentary.

--------------------------------------------------
TASK
From a rumor, extract ONE atomic factual claim suitable for verification and clustering.

Each claim must represent ONE real-world relationship:

    subject → relation → object

--------------------------------------------------
ATOMICITY RULE (CRITICAL)

Split conjunctions:

"X and Y cause Z"
→ X causes Z
→ Y causes Z

"X causes Y and Z"
→ X causes Y
→ X causes Z

Never keep conjunctions inside a claim.
Each claim must stand independently.

--------------------------------------------------
CLAIM RULES

The claim field:
- minimal factual statement
- no "and/or/with"
- no explanation
- no context phrases

--------------------------------------------------
ENTITY RULES

entities MUST contain exactly two items:
[subject, object]

Do NOT include time/location/conditions.

--------------------------------------------------
CLAIM TYPE (STRICT ENUM)

Choose one:
health | death | policy | event | statistic | relationship | other

--------------------------------------------------
TIME & LOCATION RULE

If missing → return null (NOT "NAN")

--------------------------------------------------
CANONICAL TEXT CONTRACT

Format:
<subject> <relation> <object> [context]

Allowed relations:
prevents | causes | cures | treats | increases | decreases | kills |
contains | leads_to | results_in | died_from | implemented | occurred_in | affects

Rules:
- lowercase except proper nouns
- remove modal words
- ≤ 12 words
- health claims end with "in humans"
- must match the claim meaning exactly

--------------------------------------------------
OUTPUT FORMAT (STRICT — single object, NOT a list)

{{
    "claim_id": integer (start at 0),
    "claim": string,
    "claim_type": one of enum,
    "entities": [subject, object],
    "time": string or null,
    "location": string or null,
    "canonical_text": string
}}

Return STRICT JSON only.

--------------------------------------------------
Rumor: {rumor}
""")

parser = PydanticOutputParser(pydantic_object=ValidationOutput)

validation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a strict factual verification classifier.

You will receive:

1) A list of CLAIMS
2) A list of DOCUMENT GROUPS

Each claim at index i must ONLY be evaluated using the
documents at index i.

Never mix indices.

Labels:

supported:
documents clearly confirm the claim

contradicted:
documents clearly deny the claim

conflicting:
documents contain both support and contradiction

insufficient:
documents related but no proof

Rules:
- No outside knowledge
- No guessing
- Output labels must match claim count exactly
- Order must be preserved

Return JSON only:

{format_instructions}
"""
    ),
    (
        "human",
        """
CLAIMS:
{claims}

DOCUMENTS:
{documents}
"""
    )
]).partial(format_instructions=parser.get_format_instructions())

# ── CSV helpers ───────────────────────────────────────────────────────────────
def _parse_emb(s: str) -> list:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(s)


def _load_cluster_df(csv_path: str) -> pd.DataFrame:
    """Load cluster.csv; return empty DataFrame with correct columns if missing."""
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["cluster_id", "embedding_representation", "participants"])
    return pd.read_csv(csv_path)


def _load_cluster_matrix(df: pd.DataFrame):
    """Return (cluster_ids list, L2-normalised embedding matrix) from a loaded df."""
    cluster_ids = df["cluster_id"].tolist()
    matrix = np.array(
        df["embedding_representation"].apply(_parse_emb).tolist(),
        dtype=np.float32
    )
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-10, None)
    return cluster_ids, matrix


def _next_cluster_id(df: pd.DataFrame) -> int:
    """Auto-increment: max existing cluster_id + 1, or 1 if CSV is empty."""
    if df.empty:
        return 1
    return int(df["cluster_id"].max()) + 1

# ── node functions ────────────────────────────────────────────────────────────

structured_llm = json_model.with_structured_output(Claim)

def extract_claim(state: RumorState) -> dict:
    """LLM extracts one atomic Claim from the rumor text."""
    result: Claim = (json_extract_prompt | structured_llm).invoke({"rumor": state["rumor"]})
    return {"claim": result.model_dump()}


def attach_embedding(state: RumorState) -> dict:
    """Embed canonical_text; stored on state separately — NOT inside claim dict."""
    embedding = embed_model.embed_documents([state["claim"]["canonical_text"]])[0]
    return {"embedding": embedding}


def similarity_check(state: RumorState) -> dict:
    """
    Cosine similarity of claim embedding vs all cluster centroids in cluster.csv.
    HIT  → matched_cluster_id is set (int), sim_score >= THRESHOLD
    MISS → matched_cluster_id is None,    sim_score <  THRESHOLD
    """
    claim_vec  = np.array(state["embedding"], dtype=np.float32)
    claim_norm = claim_vec / np.clip(np.linalg.norm(claim_vec), 1e-10, None)

    df = _load_cluster_df(CLUSTER_CSV_PATH)

    if df.empty:
        return {"sim_score": 0.0, "matched_cluster_id": None}

    cluster_ids, matrix = _load_cluster_matrix(df)
    scores       = matrix @ claim_norm
    best_idx     = int(np.argmax(scores))
    best_score   = float(scores[best_idx])
    best_cluster = cluster_ids[best_idx]

    return {
        "sim_score":          best_score,
        "matched_cluster_id": best_cluster if best_score >= SIMILARITY_THRESHOLD else None,
    }


def append_to_cluster(state: RumorState) -> dict:
    """
    HIT path — append claim dict as a new participant to the matched cluster row.
    Reads cluster.csv → finds matched row → appends claim dict → writes back.
    """
    df         = _load_cluster_df(CLUSTER_CSV_PATH)
    cluster_id = state["matched_cluster_id"]
    claim      = state["claim"]

    row_mask = df["cluster_id"] == cluster_id
    if not row_mask.any():
        print(f"[WARN] cluster_id={cluster_id} not found in CSV; skipping append.")
        return {}

    existing_raw = df.loc[row_mask, "participants"].values[0]
    participants = json.loads(existing_raw) if isinstance(existing_raw, str) else []
    participants.append(claim)

    df.loc[row_mask, "participants"] = json.dumps(participants)
    df.to_csv(CLUSTER_CSV_PATH, index=False)

    return {}


def rag_retrieve(state: RumorState) -> dict:
    """MISS path — retrieve supporting docs from Chroma for canonical_text."""
    docs = retriever.invoke(state["claim"]["canonical_text"])
    return {"rag_docs": docs}


def validate(state: RumorState) -> dict:
    """MISS path — LLM verdict against retrieved docs."""
    claim     = state["claim"]
    doc_texts = [
        d.page_content if isinstance(d, Document) else str(d)
        for d in (state.get("rag_docs") or [])
    ]

    validation_result: ValidationOutput = (
        validation_prompt | jsonmodel | parser
    ).invoke({
        "claims":    [claim["canonical_text"]],
        "documents": [doc_texts],
    })

    return {
        "plain_json": claim,
        "validation": validation_result,
    }


def create_cluster(state: RumorState) -> dict:
    """
    MISS path — write a brand new cluster row to cluster.csv.

    cluster_id               : auto-incremented int (max existing + 1, starts at 1)
    embedding_representation : JSON-serialised claim embedding
    participants             : JSON list with claim dict as first entry
    """
    df        = _load_cluster_df(CLUSTER_CSV_PATH)
    new_id    = _next_cluster_id(df)
    claim     = state["claim"]
    embedding = state["embedding"]

    new_row = pd.DataFrame([{
        "cluster_id":               new_id,
        "embedding_representation": json.dumps(embedding),
        "participants":             json.dumps([claim]),
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CLUSTER_CSV_PATH, index=False)

    return {"new_cluster_id": new_id}

# ── routing ───────────────────────────────────────────────────────────────────
def route_after_similarity(state: RumorState) -> str:
    """
    HIT  (score >= THRESHOLD) → append_to_cluster → END
    MISS (score <  THRESHOLD) → rag_retrieve → validate → create_cluster → END
    """
    score = state.get("sim_score") or 0.0
    return "append_to_cluster" if score >= SIMILARITY_THRESHOLD else "rag_retrieve"

# ── graph assembly ────────────────────────────────────────────────────────────
builder = StateGraph(RumorState)

builder.add_node("extract_claim",     extract_claim)
builder.add_node("attach_embedding",  attach_embedding)
builder.add_node("similarity_check",  similarity_check)
builder.add_node("append_to_cluster", append_to_cluster)   # HIT path
builder.add_node("rag_retrieve",      rag_retrieve)        # MISS path
builder.add_node("validate",          validate)            # MISS path
builder.add_node("create_cluster",    create_cluster)      # MISS path

builder.set_entry_point("extract_claim")
builder.add_edge("extract_claim",    "attach_embedding")
builder.add_edge("attach_embedding", "similarity_check")

builder.add_conditional_edges(
    "similarity_check",
    route_after_similarity,
    {
        "append_to_cluster": "append_to_cluster",
        "rag_retrieve":      "rag_retrieve",
    }
)

builder.add_edge("append_to_cluster", END)
builder.add_edge("rag_retrieve",      "validate")
builder.add_edge("validate",          "create_cluster")
builder.add_edge("create_cluster",    END)

graph = builder.compile()
print("Graph compiled ✓")

# ── result printer (API-ready) ────────────────────────────────────────────────
def print_result(result: dict) -> None:
    """
    Human-readable summary of a pipeline result.
    Same fields can be returned directly from a FastAPI endpoint.
    """
    sep = "=" * 60
    print(f"\n{sep}")

    if result.get("matched_cluster_id") is not None:
        # ── HIT ──────────────────────────────────────────────────
        print(f"PATH            : HIT")
        print(f"cluster_id      : {result['matched_cluster_id']}")
        print(f"sim_score       : {result['sim_score']:.4f}")
        print(f"canonical_text  : {result['claim']['canonical_text']}")
        print(f"claim_type      : {result['claim']['claim_type']}")
        print(f"entities        : {result['claim']['entities']}")
        print(f"location        : {result['claim'].get('location')}")
        print(f"time            : {result['claim'].get('time')}")

    else:
        # ── MISS ─────────────────────────────────────────────────
        claim    = result.get("plain_json") or {}
        rag_docs = result.get("rag_docs") or []
        verdicts = result.get("validation")

        print(f"PATH            : MISS")
        print(f"sim_score       : {result['sim_score']:.4f}")
        print(f"new_cluster_id  : {result.get('new_cluster_id')}")
        print(f"canonical_text  : {claim.get('canonical_text')}")
        print(f"claim_type      : {claim.get('claim_type')}")
        print(f"entities        : {claim.get('entities')}")
        print(f"location        : {claim.get('location')}")
        print(f"time            : {claim.get('time')}")

        print(f"\nVERDICT")
        if verdicts:
            for i, label in enumerate(verdicts.results):
                print(f"  [{i}] {label.verdict}")
        else:
            print("  (none)")

        print(f"\nRAG DOCS  ({len(rag_docs)} retrieved)")
        for i, doc in enumerate(rag_docs):
            content = doc.page_content if isinstance(doc, Document) else str(doc)
            meta    = doc.metadata    if isinstance(doc, Document) else {}
            print(f"  [{i}] {content}")
            if meta:
                src  = meta.get("source", "")
                year = meta.get("year", "")
                cred = meta.get("credibility", "")
                print(f"       source={src}  year={year}  credibility={cred}")

    print(f"{sep}\n")


# ── entry point ───────────────────────────────────────────────────────────────
def run_pipeline(rumor: str) -> dict:
    """Run the rumor validation graph for a single claim."""
    return graph.invoke({
        "rumor":              rumor,
        "claim":              None,
        "embedding":          None,
        "sim_score":          None,
        "matched_cluster_id": None,
        "rag_docs":           None,
        "plain_json":         None,
        "validation":         None,
        "new_cluster_id":     None,
    })



# ── run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_pipeline(
        "From whatsapp doctors say drinking cold water after meals causes stomach cancer in India"
    )
    print_result(result)
    