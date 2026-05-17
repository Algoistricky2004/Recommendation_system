"""
scripts/build_index.py
======================
Run ONCE before starting the server:

    python scripts/build_index.py

What it does:
  1. Loads the SHL catalog JSON (377 items)
  2. Builds a rich document string per item (name x3 + keys + job_levels + description)
  3. Encodes all documents with all-MiniLM-L6-v2 (22 MB, CPU-safe)
  4. Builds a FAISS FlatIP index (cosine similarity via normalized vectors)
  5. Saves:
       data/catalog_embeddings.npy   — float32 matrix (377 x 384)
       data/catalog_faiss.index      — FAISS binary index

At server startup, these are loaded in ~50ms instead of re-encoding every time.

Model choice: all-MiniLM-L6-v2
  - 22 MB download (cached after first run)
  - 384-dim embeddings
  - ~2–5s to encode 377 items on CPU (one-time)
  - ~50ms per query encode at runtime
  - Best CPU/accuracy tradeoff in the sentence-transformers lineup
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
CATALOG_PATH = ROOT / "data" / "shl_product_catalog.json"
EMBEDDINGS_PATH = ROOT / "data" / "catalog_embeddings.npy"
FAISS_INDEX_PATH = ROOT / "data" / "catalog_faiss.index"
MODEL_NAME = "all-MiniLM-L6-v2"
# ---------------------------------------------------------------------------


def build_document(item: dict) -> str:
    """
    Build the text that gets embedded for each catalog item.

    Design choices:
    - Name × 3: boosts exact name recall in semantic space
    - keys: test type context ("Personality & Behavior", "Knowledge & Skills")
    - job_levels: critical for seniority filtering
    - description[:400]: semantic content about what the test measures
    
    Excluded: languages, duration, URL — not semantically meaningful for similarity
    """
    name = item.get("name", "")
    keys = " ".join(item.get("keys", []))
    job_levels = " ".join(item.get("job_levels", []))
    description = (item.get("description", "") or "")[:400]
    return f"{name} {name} {name} {keys} {job_levels} {description}"


def main():
    print(f"Loading catalog from {CATALOG_PATH}...")
    if not CATALOG_PATH.exists():
        print(f"ERROR: Catalog not found at {CATALOG_PATH}")
        sys.exit(1)

    with CATALOG_PATH.open("rb") as f:
        raw = f.read()
    items: list[dict] = json.loads(raw, strict=False)
    print(f"  {len(items)} products loaded.")

    # Build documents
    documents = [build_document(item) for item in items]

    # Load model
    print(f"\nLoading embedding model '{MODEL_NAME}'...")
    print("  (Downloads ~22 MB on first run, then cached locally)")
    t0 = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

    # Encode catalog
    print(f"\nEncoding {len(documents)} catalog items (CPU)...")
    t0 = time.perf_counter()
    embeddings: np.ndarray = model.encode(
        documents,
        batch_size=64,
        normalize_embeddings=True,   # L2-normalize → dot product = cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    encode_time = time.perf_counter() - t0
    print(f"  Encoded in {encode_time:.1f}s | shape: {embeddings.shape} | dtype: {embeddings.dtype}")

    # Cast to float32 (FAISS requirement)
    embeddings = embeddings.astype(np.float32)

    # Save raw embeddings (for inspection / reuse without FAISS)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"\nEmbeddings saved → {EMBEDDINGS_PATH}  ({EMBEDDINGS_PATH.stat().st_size // 1024} KB)")

    # Build FAISS index
    # FlatIP = exact inner product search (= cosine similarity on L2-normalized vectors)
    # For 377 items, Flat is faster than IVF (no quantization overhead)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index saved → {FAISS_INDEX_PATH}  ({FAISS_INDEX_PATH.stat().st_size // 1024} KB)")

    # Quick sanity check
    print("\nSanity check — top-5 for 'senior leadership executive selection benchmark':")
    q_vec = model.encode(
        ["senior leadership executive selection benchmark"],
        normalize_embeddings=True,
    ).astype(np.float32)
    scores, indices = index.search(q_vec, 5)
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {rank+1}. [{score:.3f}] {items[idx]['name']}")

    print("\n✅ Index build complete. Start the server with:")
    print("   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()