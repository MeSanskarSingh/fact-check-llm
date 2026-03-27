"""
cluster_manager.py  (v2 — aligned to similarity_check.ipynb schema)
====================================================================

Cluster schema (matches your notebook exactly):
    {
        "id":               int,
        "embedding_parent": list[float],   # centroid of all member embeddings
        "participants":     list[int]       # claim_ids belonging to this cluster
    }

Two CSV files are maintained:
    ┌─────────────────────────────────────────────────────────────────┐
    │  claim_clusters.csv                                             │
    │  id | embedding_parent (JSON) | participants (JSON list of int) │
    ├─────────────────────────────────────────────────────────────────┤
    │  claim_records.csv                                              │
    │  claim_id | claim | canonical_text | claim_type | entities      │
    │           | time  | location       | verdict    | cluster_id    │
    └─────────────────────────────────────────────────────────────────┘

Key design decisions
    • cluster id=0 is reserved as the sentinel "no match" value,
      so real cluster IDs start at 1.
    • embedding_parent is updated as a running centroid every time
      a new claim is merged in.
    • The similarity_check function signature is kept identical to
      your notebook; only the RunnableLambda wrapper is fixed.
"""

import csv
import json
import os

from ml_pipeline.clustering.similarity import similarity_check

import numpy as np

# ─────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────
CLUSTER_CSV   = "claim_clusters.csv"
CLAIMS_CSV    = "claim_records.csv"
THRESHOLD     = 0.75          # must match your similarity_check notebook

CLUSTER_FIELDS = ["id", "embedding_parent", "participants"]

CLAIM_FIELDS   = [
    "claim_id", "claim", "canonical_text", "claim_type",
    "entities", "time", "location", "verdict", "cluster_id",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLUSTER CSV I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_clusters(path: str = CLUSTER_CSV) -> list[dict]:
    """
    Load clusters from CSV.
    Each row → {"id": int, "embedding_parent": list[float], "participants": list[int]}
    Returns [] when the file does not exist yet.
    """
    if not os.path.exists(path):
        return []
    clusters = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clusters.append({
                "id":               int(row["id"]),
                "embedding_parent": json.loads(row["embedding_parent"]),
                "participants":     json.loads(row["participants"]),
            })
    return clusters


def save_clusters(clusters: list[dict], path: str = CLUSTER_CSV) -> None:
    """Overwrite the cluster CSV with the current in-memory state."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLUSTER_FIELDS)
        w.writeheader()
        for c in clusters:
            w.writerow({
                "id":               c["id"],
                "embedding_parent": json.dumps(c["embedding_parent"]),
                "participants":     json.dumps(c["participants"]),
            })


def _next_cluster_id(clusters: list[dict]) -> int:
    """Next available cluster ID (starts at 1; 0 is reserved as sentinel)."""
    if not clusters:
        return 1
    return max(c["id"] for c in clusters) + 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CLAIM RECORDS CSV I/O
# ═══════════════════════════════════════════════════════════════════════════════

def load_claim_records(path: str = CLAIMS_CSV) -> list[dict]:
    """Load all stored claim records. Returns [] when file missing."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _next_claim_id(records: list[dict]) -> int:
    """Auto-increment claim_id."""
    if not records:
        return 0
    return max(int(r["claim_id"]) for r in records) + 1


def _append_claim_record(record: dict, path: str = CLAIMS_CSV) -> None:
    """Append a single claim record row (creates file + header if needed)."""
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLAIM_FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(record)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — UPDATE CLUSTERS  (call after chain.invoke())
# ═══════════════════════════════════════════════════════════════════════════════

def _running_centroid(
    old_centroid: list[float],
    new_embedding: list[float],
    old_count: int,
) -> list[float]:
    """
    Efficient running centroid — avoids re-embedding all members:
        new_centroid = (old_centroid * n + new_vec) / (n + 1)
    """
    old = np.array(old_centroid)
    new = np.array(new_embedding)
    return ((old * old_count + new) / (old_count + 1)).tolist()


def update_clusters(
    chain_result: dict,
    threshold: float     = THRESHOLD,
    cluster_path: str    = CLUSTER_CSV,
    claims_path:  str    = CLAIMS_CSV,
) -> tuple[list[dict], list[dict]]:
    """
    Persist cluster and claim data after chain.invoke().

    Flow for each claim:
        1. Run similarity_check against the CURRENT clusters list
           (so claims from the same rumour can form separate clusters
            or merge with each other if similar enough)
        2a. score >= threshold  →  merge claim into matched cluster,
                                   update embedding_parent (running centroid),
                                   append claim_id to participants
        2b. score <  threshold  →  create new cluster (id starts at 1)
        3.  Write claim record to claim_records.csv
        4.  After all claims processed, overwrite claim_clusters.csv

    Args:
        chain_result  – dict returned by chain.invoke()
        threshold     – cosine similarity cut-off
        cluster_path  – path to claim_clusters.csv
        claims_path   – path to claim_records.csv

    Returns:
        (clusters, summary_rows)
        clusters     – updated list (already saved to CSV)
        summary_rows – one dict per claim, consumed by print_result():
            {
                claim_id, canonical_text, claim_type, entities,
                time, location, verdict,
                action,      "merged" | "new_cluster"
                score,       best cosine score found
                cluster_id,  final cluster the claim belongs to
            }
    """
    clusters = load_clusters(cluster_path)
    records  = load_claim_records(claims_path)

    plain_json = chain_result.get("plain_json", [])
    validation = chain_result.get("validation")
    verdicts   = [r.verdict for r in validation.results] if validation else []

    summary_rows = []

    for idx, item in enumerate(plain_json):
        claim   = item["pass_rest"]["claim"]
        verdict = verdicts[idx] if idx < len(verdicts) else "insufficient"

        sim      = similarity_check(claim, clusters, threshold)
        score    = sim["score"]
        cid      = sim["cluster_id"]   # 0 = no match

        claim_id = _next_claim_id(records)

        if cid != 0:
            # ── MERGE into existing cluster ───────────────────────────────
            target    = next(c for c in clusters if c["id"] == cid)
            old_count = len(target["participants"])

            target["embedding_parent"] = _running_centroid(
                target["embedding_parent"],
                claim["embedding"],
                old_count,
            )
            target["participants"].append(claim_id)
            action = "merged"

        else:
            # ── CREATE new cluster ────────────────────────────────────────
            cid = _next_cluster_id(clusters)
            clusters.append({
                "id":               cid,
                "embedding_parent": claim["embedding"],
                "participants":     [claim_id],
            })
            action = "new_cluster"

        # ── Write claim record ────────────────────────────────────────────
        record = {
            "claim_id":       claim_id,
            "claim":          claim.get("claim", ""),
            "canonical_text": claim.get("canonical_text", ""),
            "claim_type":     claim.get("claim_type", "other"),
            "entities":       json.dumps(claim.get("entities", [])),
            "time":           claim.get("time") or "",
            "location":       claim.get("location") or "",
            "verdict":        verdict,
            "cluster_id":     cid,
        }
        records.append(record)
        _append_claim_record(record, claims_path)

        summary_rows.append({
            "claim_id":       claim_id,
            "canonical_text": claim.get("canonical_text", ""),
            "claim_type":     claim.get("claim_type", "other"),
            "entities":       claim.get("entities", []),
            "time":           claim.get("time"),
            "location":       claim.get("location"),
            "verdict":        verdict,
            "action":         action,
            "score":          score,
            "cluster_id":     cid,
        })

    save_clusters(clusters, cluster_path)
    return clusters, summary_rows


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULT PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

_C = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "grey":   "\033[90m",
    "cyan":   "\033[96m",
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
}

_VERDICT_COLOR = {
    "supported":    _C["green"],
    "contradicted": _C["red"],
    "conflicting":  _C["yellow"],
    "insufficient": _C["grey"],
}


def _badge(text: str, color: str) -> str:
    return f"{color}{_C['bold']}[ {text.upper():^13} ]{_C['reset']}"


def _wrap(text: str, width: int = 58, indent: str = "        ") -> list[str]:
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        lines.append(line)
    return [indent + l for l in lines] if lines else [indent]


def print_result(chain_result: dict, summary_rows: list[dict]) -> None:
    """
    Print a colour-coded terminal summary of the pipeline output.

    Args:
        chain_result  – output of chain.invoke()
        summary_rows  – second return value of update_clusters()
    """
    plain_json = chain_result.get("plain_json", [])
    B, R, D    = _C["bold"], _C["reset"], _C["dim"]

    print()
    print(f"{B}{'═' * 72}{R}")
    print(f"{B}   HEALTH RUMOUR FACT-CHECK REPORT{R}")
    print(f"{'═' * 72}")
    print(f"   Claims extracted : {B}{len(plain_json)}{R}")
    print(f"{'─' * 72}")

    for idx, item in enumerate(plain_json):
        claim   = item["pass_rest"]["claim"]
        row     = summary_rows[idx] if idx < len(summary_rows) else {}
        verdict = row.get("verdict", "insufficient")
        score   = row.get("score", -1.0)
        action  = row.get("action", "—")
        cid     = row.get("cluster_id", "—")
        docs    = item.get("rag_doc", [])

        print()
        print(
            f"  {B}Claim #{idx+1}  "
            f"(claim_id={row.get('claim_id','?')}){R}  "
            f"{_badge(verdict, _VERDICT_COLOR.get(verdict,''))}"
        )
        print(f"  {_C['cyan']}{claim.get('canonical_text', '')}{R}")
        print()
        print(f"    {D}type     :{R} {claim.get('claim_type', '—')}")
        print(f"    {D}entities :{R} {' → '.join(claim.get('entities', []))}")
        if claim.get("time"):
            print(f"    {D}time     :{R} {claim['time']}")
        if claim.get("location"):
            print(f"    {D}location :{R} {claim['location']}")
        print()

        # Evidence block
        if docs:
            print(f"    {B}Evidence:{R}")
            for d in docs:
                content = d.page_content if hasattr(d, "page_content") else str(d)
                lines   = _wrap(content)
                print(f"      • {lines[0].strip()}")
                for l in lines[1:]:
                    print(l)
        else:
            print(f"    {D}Evidence : (none retrieved){R}")

        print()

        # Cluster block
        if action == "merged":
            cluster_str = (
                f"{_C['green']}merged → cluster #{cid}{R}"
                f"  {D}(cosine {score:.3f}){R}"
            )
        else:
            cluster_str = (
                f"{_C['yellow']}new cluster #{cid} created{R}"
                f"  {D}(best score was {score:.3f}){R}"
            )

        print(f"    {B}Cluster  :{R} {cluster_str}")
        print(f"  {'─' * 68}")

    print()
    print(f"{'═' * 72}")
    print(f"  {B}Files updated:{R}")
    print(f"    • {CLUSTER_CSV:<22} cluster heads & participant IDs")
    print(f"    • {CLAIMS_CSV:<22} full claim records with verdicts")
    print(f"{'═' * 72}")
    print()