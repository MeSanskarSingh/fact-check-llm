import os
import ast
import json

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from typing import Literal, Optional, TypedDict, List

from pydantic import BaseModel, Field

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, END

load_dotenv()

# ── tuneable constants ────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.85
CLUSTER_CSV_PATH     = "cluster.csv"

# ── models & stores ───────────────────────────────────────────────────────────
embed_model = MistralAIEmbeddings()
json_model  = ChatMistralAI(model="mistral-small-latest", temperature=0.1)
jsonmodel   = ChatMistralAI(model="mistral-small-latest", temperature=0.1)

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


# ── graph state ───────────────────────────────────────────────────────────────
class RumorState(TypedDict):
    rumor:               str
    claim:               Optional[dict]
    embedding:           Optional[List[float]]
    sim_score:           Optional[float]
    matched_cluster_id:  Optional[int]
    rag_docs:            Optional[List[Document]]
    plain_json:          Optional[dict]
    validation:          Optional[ValidationOutput]
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

    subject -> relation -> object

--------------------------------------------------
ATOMICITY RULE (CRITICAL)

Split conjunctions:

"X and Y cause Z"
-> X causes Z
-> Y causes Z

"X causes Y and Z"
-> X causes Y
-> X causes Z

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

If missing -> return null (NOT "NAN")

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
- 12 words max
- health claims end with "in humans"
- must match the claim meaning exactly

--------------------------------------------------
OUTPUT FORMAT (STRICT - single object, NOT a list)

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

_parser = PydanticOutputParser(pydantic_object=ValidationOutput)

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
]).partial(format_instructions=_parser.get_format_instructions())


# ── CSV helpers ───────────────────────────────────────────────────────────────
def _parse_emb(s: str) -> list:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return ast.literal_eval(s)


def _load_cluster_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["cluster_id", "embedding_representation", "participants"])
    return pd.read_csv(csv_path)


def _load_cluster_matrix(df: pd.DataFrame):
    cluster_ids = df["cluster_id"].tolist()
    matrix = np.array(
        df["embedding_representation"].apply(_parse_emb).tolist(),
        dtype=np.float32
    )
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-10, None)
    return cluster_ids, matrix


def _next_cluster_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    return int(df["cluster_id"].max()) + 1


# ── graph node functions ──────────────────────────────────────────────────────
_structured_llm = json_model.with_structured_output(Claim)


def extract_claim(state: RumorState) -> dict:
    """LLM extracts one atomic Claim from the rumor text."""
    result: Claim = (json_extract_prompt | _structured_llm).invoke({"rumor": state["rumor"]})
    return {"claim": result.model_dump()}


def attach_embedding(state: RumorState) -> dict:
    """Embed canonical_text; stored on state separately — NOT inside claim dict."""
    embedding = embed_model.embed_documents([state["claim"]["canonical_text"]])[0]
    return {"embedding": embedding}


def similarity_check(state: RumorState) -> dict:
    """
    Cosine similarity of claim embedding vs all cluster centroids in cluster.csv.
    HIT  -> matched_cluster_id is set (int), sim_score >= THRESHOLD
    MISS -> matched_cluster_id is None,    sim_score <  THRESHOLD
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
    """HIT path — append claim dict as a new participant to the matched cluster row."""
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
        validation_prompt | jsonmodel | _parser
    ).invoke({
        "claims":    [claim["canonical_text"]],
        "documents": [doc_texts],
    })

    return {
        "plain_json": claim,
        "validation": validation_result,
    }


def create_cluster(state: RumorState) -> dict:
    """MISS path — write a brand new cluster row to cluster.csv."""
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
    score = state.get("sim_score") or 0.0
    return "append_to_cluster" if score >= SIMILARITY_THRESHOLD else "rag_retrieve"


# ── graph assembly ────────────────────────────────────────────────────────────
_builder = StateGraph(RumorState)

_builder.add_node("extract_claim",     extract_claim)
_builder.add_node("attach_embedding",  attach_embedding)
_builder.add_node("similarity_check",  similarity_check)
_builder.add_node("append_to_cluster", append_to_cluster)
_builder.add_node("rag_retrieve",      rag_retrieve)
_builder.add_node("validate",          validate)
_builder.add_node("create_cluster",    create_cluster)

_builder.set_entry_point("extract_claim")
_builder.add_edge("extract_claim",    "attach_embedding")
_builder.add_edge("attach_embedding", "similarity_check")

_builder.add_conditional_edges(
    "similarity_check",
    route_after_similarity,
    {
        "append_to_cluster": "append_to_cluster",
        "rag_retrieve":      "rag_retrieve",
    }
)

_builder.add_edge("append_to_cluster", END)
_builder.add_edge("rag_retrieve",      "validate")
_builder.add_edge("validate",          "create_cluster")
_builder.add_edge("create_cluster",    END)

_graph = _builder.compile()


# ── internal pipeline runner ──────────────────────────────────────────────────
def _run_pipeline(rumor: str) -> dict:
    """Run the full LangGraph rumor pipeline and return the final RumorState."""
    return _graph.invoke({
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


# ── verdict / confidence mapping ──────────────────────────────────────────────
_VERDICT_MAP: dict[str, str] = {
    "supported":    "Real",
    "contradicted": "Fake",
    "conflicting":  "Uncertain",
    "insufficient": "Uncertain",
}

_CONFIDENCE_MAP: dict[str, float] = {
    "supported":    0.90,
    "contradicted": 0.85,
    "conflicting":  0.50,
    "insufficient": 0.40,
}


# ── ValidationService ─────────────────────────────────────────────────────────
class ValidationService:
    """
    Wraps the full LangGraph rumor-detection pipeline.

    Pipeline flow:
        extract_claim -> attach_embedding -> similarity_check
            HIT  (sim >= 0.85) -> append_to_cluster -> END
            MISS (sim <  0.85) -> rag_retrieve -> validate -> create_cluster -> END

    Public interface (unchanged from original):
        validate_text(text) -> { "verdict", "confidence", "explanation" }

    Verdict mapping (MISS path):
        supported    -> Real,      confidence 0.90
        contradicted -> Fake,      confidence 0.85
        conflicting  -> Uncertain, confidence 0.50
        insufficient -> Uncertain, confidence 0.40

    HIT path: verdict is always "Real", confidence = cosine similarity score.
    """

    def validate_text(self, text: str) -> dict:
        try:
            result = _run_pipeline(text)
            return self._build_response(result)
        except Exception as exc:
            print("ValidationService error:", exc)
            return {
                "verdict":     "Uncertain",
                "confidence":  0.5,
                "explanation": "Validation failed, fallback used",
            }

    # ── response builder ──────────────────────────────────────────────────────
    def _build_response(self, result: dict) -> dict:
        matched_cluster_id = result.get("matched_cluster_id")
        sim_score: float   = result.get("sim_score") or 0.0

        # ── HIT path ──────────────────────────────────────────────────────────
        if matched_cluster_id is not None:
            claim = result.get("claim") or {}
            return {
                "verdict":     "Real",
                "confidence":  round(sim_score, 4),
                "explanation": (
                    f"Matched existing cluster #{matched_cluster_id} "
                    f"(similarity {sim_score:.2%}). "
                    f"Canonical: \"{claim.get('canonical_text', 'N/A')}\". "
                    f"Type: {claim.get('claim_type', 'N/A')}. "
                    f"Entities: {claim.get('entities', [])}. "
                    f"Location: {claim.get('location')}. "
                    f"Time: {claim.get('time')}."
                ),
            }

        # ── MISS path ─────────────────────────────────────────────────────────
        validation: ValidationOutput | None = result.get("validation")
        claim       = result.get("plain_json") or {}
        new_cluster = result.get("new_cluster_id")
        rag_docs    = result.get("rag_docs") or []

        doc_sources = [
            f"{d.metadata.get('source', 'unknown')} "
            f"(credibility={d.metadata.get('credibility', 'N/A')}, "
            f"year={d.metadata.get('year', 'N/A')})"
            for d in rag_docs if isinstance(d, Document)
        ]
        sources_str = ", ".join(doc_sources) if doc_sources else "no external sources retrieved"

        if validation and validation.results:
            label      = validation.results[0].verdict
            verdict    = _VERDICT_MAP.get(label, "Uncertain")
            confidence = _CONFIDENCE_MAP.get(label, 0.5)
            explanation = (
                f"LangGraph verdict: {label} -> {verdict}. "
                f"Canonical: \"{claim.get('canonical_text', 'N/A')}\". "
                f"Type: {claim.get('claim_type', 'N/A')}. "
                f"Entities: {claim.get('entities', [])}. "
                f"Location: {claim.get('location')}. "
                f"Time: {claim.get('time')}. "
                f"RAG docs ({len(rag_docs)}): {sources_str}. "
                f"Assigned to new cluster #{new_cluster}."
            )
        else:
            verdict     = "Uncertain"
            confidence  = 0.40
            explanation = (
                f"New claim (sim {sim_score:.2%}), no RAG docs available for verification. "
                f"Canonical: \"{claim.get('canonical_text', 'N/A')}\". "
                f"Type: {claim.get('claim_type', 'N/A')}. "
                f"Entities: {claim.get('entities', [])}. "
                f"Assigned to new cluster #{new_cluster}."
            )

        return {
            "verdict":     verdict,
            "confidence":  confidence,
            "explanation": explanation,
        }


# ── singleton accessor (identical to original interface) ──────────────────────
_validation_service: ValidationService | None = None


def get_validation_service() -> ValidationService:
    global _validation_service
    if _validation_service is None:
        _validation_service = ValidationService()
    return _validation_service
