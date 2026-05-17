"""
retrieval.py — Hybrid BM25 + FAISS retrieval with Reciprocal Rank Fusion.

Pipeline per request:
  1. Build search query from conversation history
  2. BM25 search  → ranked list with scores
  3. FAISS search → ranked list with cosine scores (if index available)
  4. RRF merge   → unified ranking, top-10 candidates
  5. Pin explicitly mentioned product names (for comparison queries)

Why RRF (Reciprocal Rank Fusion)?
  - BM25 is great for exact keyword matches ("Java", "SVAR", "OPQ32r")
  - FAISS is great for semantic matches ("leadership" → OPQ Leadership Report)
  - RRF combines both without needing score calibration between the two systems
  - Formula: score(d) = Σ  1 / (k + rank_i(d))   where k=60 (standard)

Token budget impact:
  - Before: 28 items × ~47 tokens = ~1,316 tokens of catalog context
  - After:  10 items × ~80 tokens = ~800 tokens  (richer but fewer)
  - Net:    ~40% fewer tokens → measurably faster LLM response
"""
from __future__ import annotations

import re
from typing import List

from app.catalog import CatalogIndex, CatalogItem
from app.models import Message

# RRF rank constant (60 is standard; higher = smoother blend)
_RRF_K = 60

# Final candidate count sent to LLM
_FINAL_TOP_K = 10

# How many candidates to pull from each retriever before merging
_RETRIEVER_TOP_K = 25


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
def build_search_query(messages: List[Message]) -> str:
    """
    Build a unified search string from all user turns.
    Latest turn is repeated 2× to bias retrieval toward the current intent.
    """
    user_turns = [m.content for m in messages if m.role == "user"]
    if not user_turns:
        return ""
    latest = user_turns[-1]
    earlier = " ".join(user_turns[:-1])
    combined = f"{latest} {latest} {earlier}"
    return re.sub(r"[>#`*_\[\]]", " ", combined)


def extract_mentioned_product_names(messages: List[Message]) -> list[str]:
    """
    Extract explicitly mentioned product names from any turn.
    Critical for comparison queries: "What's the difference between OPQ and DSI?"
    both products must appear in the candidate set regardless of BM25/FAISS ranking.
    """
    all_text = " ".join(m.content for m in messages)
    candidates = re.findall(r'\b[A-Z][A-Za-z0-9 &+.\-]{2,40}\b', all_text)
    seen, out = set(), []
    for c in candidates:
        cs = c.strip()
        if cs not in seen and len(cs) > 3:
            seen.add(cs); out.append(cs)
    return out


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def _rrf_merge(
    bm25_results: list[tuple[CatalogItem, float]],
    faiss_results: list[tuple[CatalogItem, float]],
    k: int = _RRF_K,
) -> list[CatalogItem]:
    """
    Merge BM25 and FAISS ranked lists using RRF.
    Returns items ranked by combined RRF score.
    """
    scores: dict[str, float] = {}
    item_map: dict[str, CatalogItem] = {}

    for rank, (item, _) in enumerate(bm25_results):
        key = item.entity_id
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        item_map[key] = item

    for rank, (item, _) in enumerate(faiss_results):
        key = item.entity_id
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        item_map[key] = item

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [item_map[key] for key in ranked_keys]


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------
def retrieve_candidates(
    messages: List[Message],
    catalog: CatalogIndex,
    top_k: int = _FINAL_TOP_K,
) -> list[CatalogItem]:
    """
    Hybrid retrieval pipeline.

    Returns exactly top_k items for the LLM prompt,
    plus any pinned items (explicitly named in conversation).
    """
    query = build_search_query(messages)

    # ── BM25 leg ──────────────────────────────────────────────────────────
    bm25_results = catalog.bm25_search(query, top_k=_RETRIEVER_TOP_K)

    # ── Semantic leg (FAISS) — skipped gracefully if index not built ───────
    if catalog.has_semantic:
        faiss_results = catalog.faiss_search(query, top_k=_RETRIEVER_TOP_K)
        merged = _rrf_merge(bm25_results, faiss_results)
    else:
        # BM25-only fallback — still correct, just less semantic coverage
        merged = [item for item, _ in bm25_results]

    # ── Pin explicitly named products ────────────────────────────────────
    # These MUST be in the candidate set for comparison queries to work.
    mentioned = extract_mentioned_product_names(messages)
    named_items = catalog.find_named(mentioned)

    seen_ids = {item.entity_id for item in merged[:top_k]}
    pinned = [item for item in named_items if item.entity_id not in seen_ids]

    # Final list: top_k from RRF + any pinned extras
    final = merged[:top_k] + pinned

    return final