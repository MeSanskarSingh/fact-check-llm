# Multimodal Rumor Detection & Verification System

## Overview

This project is an AI‑assisted rumor detection platform that extracts factual claims from user content, retrieves trusted knowledge using Retrieval Augmented Generation (RAG), evaluates credibility, and stores structured knowledge for future risk analysis.

The system supports text, image, audio, and video inputs and converts them into unified factual claims for verification.

Goal: Build a continuously improving knowledge intelligence system — not just a classifier — capable of reasoning, memory, clustering, and trend monitoring.

---

## Core Capabilities

### 1. Claim Understanding

The model converts raw user content into a single atomic factual claim using a structured LLM extraction prompt. The output is a validated Pydantic `Claim` object with the following fields:

- `claim_id` — unique index of the claim within the rumor
- `claim` — atomic factual proposition with exactly one subject and one object (conjunctions are split)
- `claim_type` — strict enum: `health | death | policy | event | statistic | relationship | other`
- `entities` — exactly two entries: `[subject, object]`
- `time` — explicit time reference, or `null`
- `location` — explicit location reference, or `null`
- `canonical_text` — controlled identity sentence in the form `<subject> <relation> <object> [context]`, max 12 words, using only allowed relations

Allowed canonical relations: `prevents | causes | cures | treats | increases | decreases | kills | contains | leads_to | results_in | died_from | implemented | occurred_in | affects`

This prevents semantic noise and ensures embedding similarity depends on meaning instead of wording style.

---

### 2. Multimodal Input Support

All media is converted into text claims before verification.

#### Text

Direct claim extraction via LLM structured output.

#### Image

Pipeline:

1. OCR extraction
2. Caption generation
3. Claim extraction

#### Audio

Pipeline:

1. Speech‑to‑text transcription
2. Claim extraction

#### Video

Pipeline:

1. Keyframe extraction
2. OCR + captioning per frame
3. Speech transcription
4. Merge → claim extraction

All modalities converge to the same canonical claim representation.

---

### 3. Knowledge Retrieval (RAG Layer)

Trusted factual knowledge is stored in a Chroma vector database (`collection_name="facts"`, `persist_directory="rumor_detection"`).

Stored format:

FACT ENTRY

- canonical_text
- source
- evidence
- topic

The retriever fetches the top 2 most semantically similar documents (`k=2`) for each incoming canonical claim.

Only the `canonical_text` field is embedded — not the raw rumor text, not the full JSON claim object.

Why canonical text is used:
Different wording → same meaning → same embedding neighborhood
Different topic → far apart embeddings

This prevents false matches based on writing style.

---

### 4. Dual Database Memory System

The system maintains two independent memories.

#### A. Truth Knowledge Store (Chroma Vector DB)

Contains verified facts from reliable sources embedded using `MistralAIEmbeddings`.
Used for RAG-grounded reasoning during claim validation.

#### B. Rumor Intelligence Store (Cluster CSV — `cluster.csv`)

Each row stores:

- `cluster_id` — auto-incremented integer (starts at 1)
- `embedding_representation` — JSON-serialised L2-normalised embedding of the canonical claim
- `participants` — JSON list of all claim dicts that have been assigned to this cluster

New claims are either appended to an existing cluster (HIT) or create a new row (MISS). Similarity threshold: **0.85 cosine similarity**.

---

### 5. Claim Evaluation

The LangGraph pipeline evaluates each claim through the following node sequence:

```
extract_claim → attach_embedding → similarity_check
    ├─ HIT  (sim ≥ 0.85) → append_to_cluster → END
    └─ MISS (sim < 0.85) → rag_retrieve → validate → create_cluster → END
```

**similarity_check** computes cosine similarity between the claim embedding and all cluster centroids stored in `cluster.csv`. The highest-scoring centroid determines HIT or MISS.

**validate** (MISS path only) passes the canonical claim and the top-2 retrieved Chroma documents to the LLM classifier. The classifier assigns one of four labels:

| Label | Meaning |
|---|---|
| `supported` | Documents clearly confirm the claim |
| `contradicted` | Documents clearly deny the claim |
| `conflicting` | Documents contain both support and contradiction |
| `insufficient` | Documents are related but provide no proof |

Rules: no outside knowledge, no guessing, label count must exactly match claim count, order preserved.

Final classification exposed by `ValidationService`:

| LangGraph label | Verdict | Confidence |
|---|---|---|
| `supported` | Real | 0.90 |
| `contradicted` | Fake | 0.85 |
| `conflicting` | Uncertain | 0.50 |
| `insufficient` | Uncertain | 0.40 |
| HIT path | Real | cosine sim score |

---

### 6. Continuous Learning

The system improves over time:

- New rumors are grouped into clusters in `cluster.csv` based on embedding similarity
- HIT claims are appended as participants to their matched cluster, growing the cluster's record of recurring misinformation
- MISS claims trigger full RAG + LLM validation and are stored as the seed of a new cluster
- Future claims are judged using historical cluster membership patterns

This transforms the project from verification → intelligence system.

---

## System Architecture

Input → Multimodal Processing → Claim Extraction → Canonicalization → Embedding
→ Similarity Check (cluster.csv) → HIT: append to cluster / MISS: RAG Retrieval (Chroma) → LLM Validation → New Cluster → END

---

## Embedding Strategy

Only `canonical_text` fields are embedded — using `MistralAIEmbeddings`.
Not raw text.
Not full JSON structure.

Reason:
Embeddings must represent meaning, not grammar or metadata.

All stored cluster centroids in `cluster.csv` are L2-normalised before cosine similarity comparison.

---

## Data Flow

1. User submits content
2. Convert to text (modality-specific pipeline)
3. Extract one atomic factual claim via LLM structured output (`mistral-small-latest`, temp 0.1)
4. Normalize claim into `canonical_text`
5. Embed `canonical_text` using `MistralAIEmbeddings`
6. Compute cosine similarity against all cluster centroids in `cluster.csv`
7. **HIT** (sim ≥ 0.85): append claim to matched cluster → END
8. **MISS** (sim < 0.85): retrieve top-2 docs from Chroma vector store
9. LLM classifier evaluates claim against retrieved docs → assigns `supported / contradicted / conflicting / insufficient`
10. Write new cluster row to `cluster.csv` with embedding and claim as first participant

---

## Planned Intelligent Features

### Trend Monitoring

Detect viral misinformation early

### Risk Forecasting

Predict which topics may spread next

### Source Reputation Tracking

Repeated false claims lower trust score

### Knowledge Gap Detection

Identify areas where verification data is insufficient

---

## Technology Stack

- **LLM reasoning engine** — `mistral-small-latest` via `ChatMistralAI` (temp 0.1 for structured output, claim validation)
- **Embedding model** — `MistralAIEmbeddings`
- **Vector database** — Chroma (`rumor_detection/`, collection `facts`), retriever `k=2`
- **Cluster database** — `cluster.csv` (flat file, cosine similarity threshold 0.85)
- **Orchestration** — LangGraph `StateGraph` with conditional routing
- **Schema validation** — Pydantic (`Claim`, `ClaimLabel`, `ValidationOutput`)
- **Speech recognition + OCR + captioning** — for multimodal ingestion

---

## Future Expansion Roadmap

Phase 1: Reliable claim verification system
Phase 2: Misinformation clustering intelligence
Phase 3: Predictive rumor spread detection
Phase 4: Automated public advisory generator
Phase 5: AI investigative assistant

Long‑term vision: A decision‑support AI that not only checks facts but anticipates misinformation behavior.

---

## Evaluation Metrics

- Retrieval accuracy
- Reasoning correctness
- False positive rate
- Risk prediction accuracy
- Cluster purity

---

## Key Design Principles

Meaning over wording
Memory over stateless prediction
Risk awareness over binary classification
Separation of knowledge and behavior data

---

## Project Outcome

A hybrid AI system combining:

- LLM reasoning
- Retrieval grounding
- Historical behavioral memory
- Multimodal understanding

Result: A practical real‑world misinformation intelligence platform rather than a simple classifier.
