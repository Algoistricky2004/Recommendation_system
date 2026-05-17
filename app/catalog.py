"""
catalog.py — SHL catalog loader with BM25 + FAISS hybrid retrieval support.

CPU-safe:
  - rank_bm25:           pure Python, ~1ms search
  - faiss-cpu:           BLAS-only, ~1ms search on 377 items
  - sentence-transformers: CPU inference, ~50ms per query encode
"""
from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"

KEY_TO_CODE: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

_JL_ABBREV = {
    "Professional Individual Contributor": "IC",
    "Mid-Professional": "Mid",
    "Entry-Level": "Entry",
    "Front Line Manager": "FLM",
    "General Population": "GenPop",
}

_STOP = {
    "a", "an", "the", "and", "or", "of", "for", "in", "to", "is", "are",
    "be", "this", "that", "with", "on", "at", "by", "it", "its", "we",
    "our", "their", "has", "have", "as", "from", "can", "will", "not",
    "but", "also", "which", "who", "they", "you", "i", "me", "my",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return [t for t in text.split() if t and t not in _STOP and len(t) > 1]


def keys_to_type_code(keys: list[str]) -> str:
    codes, seen = [], set()
    for k in keys:
        code = KEY_TO_CODE.get(k)
        if code and code not in seen:
            codes.append(code)
            seen.add(code)
    return ",".join(codes) if codes else "K"


# ---------------------------------------------------------------------------
# CatalogItem
# ---------------------------------------------------------------------------
class CatalogItem:
    __slots__ = (
        "entity_id", "name", "url", "description", "keys", "test_type",
        "job_levels", "languages", "duration", "adaptive", "remote",
        "_cached_repr",
    )

    def __init__(self, raw: dict):
        self.entity_id: str = raw.get("entity_id", "")
        self.name: str = (raw.get("name", "") or "").strip()
        self.url: str = (raw.get("link", "") or "").strip()
        self.description: str = (raw.get("description", "") or "").strip()
        self.keys: list[str] = raw.get("keys", [])
        self.test_type: str = keys_to_type_code(self.keys)
        self.job_levels: list[str] = raw.get("job_levels", [])
        self.languages: list[str] = raw.get("languages", [])
        self.duration: str = (raw.get("duration", "") or "").strip()
        self.adaptive: bool = (raw.get("adaptive", "no") == "yes")
        self.remote: bool = (raw.get("remote", "no") == "yes")
        self._cached_repr: str = self._build_repr()

    def _build_repr(self) -> str:
        """
        Richer 3-line format for LLM prompt injection.
        Used for the final top-10 only — richer per item, but fewer items,
        so net token count drops significantly vs old 28-item sparse format.
        """
        dur = self.duration or "—"
        jl = "/".join(_JL_ABBREV.get(j, j) for j in self.job_levels[:3]) or "—"
        adp = " [adaptive]" if self.adaptive else ""
        langs = ", ".join(self.languages[:3])
        if len(self.languages) > 3:
            langs += f" +{len(self.languages)-3}"
        desc_short = self.description[:180].rstrip()
        if len(self.description) > 180:
            desc_short += "…"
        return (
            f"{self.name} | {self.test_type} | {dur} | {jl}{adp}\n"
            f"  URL: {self.url}\n"
            f"  LANG: {langs or '—'} | {desc_short}\n"
        )

    def compact_repr(self) -> str:
        return self._cached_repr


# ---------------------------------------------------------------------------
# CatalogIndex
# ---------------------------------------------------------------------------
class CatalogIndex:
    """
    Loads catalog + BM25 at startup.
    Optionally loads FAISS index for hybrid retrieval (run build_index.py first).
    Falls back gracefully to BM25-only if FAISS is not present.
    """

    def __init__(self, catalog_path: str | Path, data_dir: str | Path | None = None):
        catalog_path = Path(catalog_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found at {catalog_path}")

        if data_dir is None:
            data_dir = catalog_path.parent
        data_dir = Path(data_dir)

        # Load raw JSON
        with catalog_path.open("rb") as fh:
            raw_bytes = fh.read()
        raw_list: list[dict] = json.loads(raw_bytes, strict=False)

        self.items: list[CatalogItem] = [CatalogItem(r) for r in raw_list]
        self._name_index: dict[str, CatalogItem] = {
            item.name.lower(): item for item in self.items
        }

        # ── BM25 ──────────────────────────────────────────────────────────
        corpus_tokens: list[list[str]] = []
        for item in self.items:
            doc = " ".join([
                item.name, item.name, item.name,   # triple-weight name
                item.description,
                " ".join(item.keys),
                " ".join(item.job_levels),
                " ".join(item.languages),
                item.duration,
                "adaptive" if item.adaptive else "",
            ])
            corpus_tokens.append(_tokenize(doc))
        self._bm25 = BM25Okapi(corpus_tokens)
        logger.info("BM25 index built (%d docs)", len(self.items))

        # ── FAISS (optional — loaded from pre-built index) ────────────────
        self._faiss_index = None
        self._embed_model = None

        embeddings_path = data_dir / "catalog_embeddings.npy"
        faiss_path = data_dir / "catalog_faiss.index"

        if faiss_path.exists() and embeddings_path.exists():
            try:
                import faiss as _faiss
                import numpy as np
                from sentence_transformers import SentenceTransformer
                self._faiss_index = _faiss.read_index(str(faiss_path))
                self._embed_model = SentenceTransformer(MODEL_NAME)
                self._np = np
                logger.info("FAISS + embedding model loaded ✅  (hybrid mode active)")
            except Exception as e:
                logger.warning("FAISS load failed (%s) — BM25-only mode", e)
        else:
            logger.warning(
                "FAISS index not found. Run:  python scripts/build_index.py  "
                "for semantic retrieval. Running in BM25-only mode."
            )

    @property
    def has_semantic(self) -> bool:
        return self._faiss_index is not None and self._embed_model is not None

    # ── BM25 search ───────────────────────────────────────────────────────
    def bm25_search(self, query: str, top_k: int = 25) -> list[tuple[CatalogItem, float]]:
        if not query.strip():
            return [(item, 0.0) for item in self.items[:top_k]]
        tokens = _tokenize(query)
        if not tokens:
            return [(item, 0.0) for item in self.items[:top_k]]
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.items[i], float(scores[i])) for i in ranked[:top_k]]

    # ── FAISS semantic search ─────────────────────────────────────────────
    def faiss_search(self, query: str, top_k: int = 25) -> list[tuple[CatalogItem, float]]:
        if not self.has_semantic:
            return []
        import numpy as np
        q_vec = self._embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        scores, indices = self._faiss_index.search(q_vec, top_k)
        return [
            (self.items[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if 0 <= int(idx) < len(self.items)
        ]

    # ── Backward-compat (used by tests) ───────────────────────────────────
    def search(self, query: str, top_k: int = 25) -> list[CatalogItem]:
        return [item for item, _ in self.bm25_search(query, top_k)]

    # ── Name lookups ──────────────────────────────────────────────────────
    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._name_index.get(name.lower())

    def find_named(self, names: list[str]) -> list[CatalogItem]:
        found, seen = [], set()
        for name in names:
            nl = name.lower().strip()
            item = self._name_index.get(nl)
            if item and item.name not in seen:
                found.append(item); seen.add(item.name); continue
            for cname, citem in self._name_index.items():
                if nl in cname or cname in nl:
                    if citem.name not in seen:
                        found.append(citem); seen.add(citem.name)
                    break
        return found

    def all_names(self) -> list[str]:
        return [item.name for item in self.items]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_catalog_instance: Optional[CatalogIndex] = None


def get_catalog(catalog_path: str | Path | None = None) -> CatalogIndex:
    global _catalog_instance
    if _catalog_instance is None:
        if catalog_path is None:
            here = Path(__file__).parent.parent
            catalog_path = here / "data" / "shl_product_catalog.json"
        _catalog_instance = CatalogIndex(catalog_path)
    return _catalog_instance