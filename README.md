# Multimodal Rumor Detection & Verification System

## Overview

This project is an AI‑assisted rumor detection platform that extracts factual claims from user content, retrieves trusted knowledge using Retrieval Augmented Generation (RAG), evaluates credibility, and stores structured knowledge for future risk analysis.

The system supports text, image, audio, and video inputs and converts them into unified factual claims for verification.

Goal: Build a continuously improving knowledge intelligence system — not just a classifier — capable of reasoning, memory, clustering, and trend monitoring.

---

## Core Capabilities

### 1. Claim Understanding

The model converts raw user content into structured factual statements:

* Canonicalized claim (normalized wording)
* Claim type (health, event, statistic, policy, etc.)
* Entities involved
* Context (time / location if present)

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

Trusted factual knowledge is stored in a vector database.

Stored format:

FACT ENTRY

* canonical_text
* source
* evidence
* topic

User claims are embedded and matched semantically against verified knowledge.

Why canonical text is used:
Different wording → same meaning → same embedding neighborhood
Different topic → far apart embeddings

This prevents false matches based on writing style.

---

### 4. Dual Database Memory System

The system maintains two independent memories.

#### A. Truth Knowledge Store (Vector DB)

Contains verified facts from reliable sources.
Used for retrieval grounded reasoning.

#### B. Rumor Intelligence Store (Cluster DB)

Stores analyzed claims:

* claim
* risk score
* classification
* embedding
* topic cluster

Purpose:
Track recurring misinformation patterns instead of treating each claim independently.

---

### 5. Claim Evaluation

Each claim receives multiple scores:

Relevance Score → similarity with known facts
Confidence Score → model certainty
Risk Score → potential harmful impact
Topic Score → category importance weighting

Final classification:

* Verified
* Likely True
* Uncertain
* Likely False
* High Risk Rumor

---

### 6. Continuous Learning

The system improves over time:

* New rumors grouped into clusters
* Emerging misinformation trends detected
* Future claims judged using historical behavior patterns

This transforms the project from verification → intelligence system.

---

## System Architecture

Input → Multimodal Processing → Claim Extraction → Canonicalization → Embedding
→ Retrieval (Truth DB) → Reasoning → Scoring → Storage (Rumor DB + Clusters)

---

## Embedding Strategy

Only canonical claims are embedded.
Not raw text.
Not JSON structure.

Reason:
Embeddings must represent meaning, not grammar.

---

## Data Flow

1. User submits content
2. Convert to text
3. Extract factual claim
4. Normalize claim wording
5. Embed claim
6. Retrieve relevant verified facts
7. Evaluate credibility
8. Assign risk
9. Store in rumor intelligence database
10. Update clusters

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

LLM reasoning engine
Vector database for factual memory
Clustering database for rumor behavior memory
Embedding model for semantic similarity
Speech recognition + OCR + captioning for multimodal ingestion

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

* Retrieval accuracy
* Reasoning correctness
* False positive rate
* Risk prediction accuracy
* Cluster purity

---

## Key Design Principles

Meaning over wording
Memory over stateless prediction
Risk awareness over binary classification
Separation of knowledge and behavior data

---

## Project Outcome

A hybrid AI system combining:

* LLM reasoning
* Retrieval grounding
* Historical behavioral memory
* Multimodal understanding

Result: A practical real‑world misinformation intelligence platform rather than a simple classifier.
