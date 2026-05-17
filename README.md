# SHL Assessment Recommender

A conversational AI agent that recommends SHL Individual Test Solutions through multi-turn dialogue.  
Built for the **SHL Labs Assignment**.

**Live API:** https://shl-recommendation-system-catalog.onrender.com  
**Interactive Docs:** https://shl-recommendation-system-catalog.onrender.com/docs

---

## What it does

- Asks one clarifying question when the query is too vague
- Recommends 1–10 SHL assessments once enough context is known
- Refines the shortlist when the user changes constraints mid-conversation
- Compares assessments using catalog facts only — never invents specs
- Refuses off-topic requests (legal advice, HR strategy, prompt injection)
- Never recommends outside the SHL catalog — hallucinated URLs are structurally impossible

---

## API

### `GET /health`
Readiness probe.
```json
{"status": "ok"}
```

### `POST /chat`
Stateless — send the full conversation history on every call.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I am hiring a mid-level Java developer"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Selection purpose, 4 years experience"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    },
    {
      "name": "Occupational Personality Questionnaire OPQ32r",
      "url": "https://www.shl.com/...",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

**Rules:**
- `recommendations` is `[]` when clarifying or refusing
- `recommendations` has 1–10 items when a shortlist is committed
- `end_of_conversation` is `true` only when user explicitly confirms
- Max 8 turns per conversation (enforced at API level)

**test_type codes:**

| Code | Meaning |
|------|---------|
| A | Ability & Aptitude |
| P | Personality & Behavior |
| K | Knowledge & Skills |
| S | Simulations |
| B | Biodata & Situational Judgment |
| C | Competencies |
| D | Development & 360 |
| E | Assessment Exercises |

---

## Architecture

```
POST /chat (stateless — full history every call)
        │
        ▼
BM25 Search (rank_bm25)          top-25 candidates
FAISS Semantic Search             top-25 candidates
(all-MiniLM-L6-v2, 384-dim)
        │
Reciprocal Rank Fusion (k=60)  → top-10 final candidates
        │
System Prompt Builder
(7 behavioral rules + catalog block injected)
        │
Groq API — llama-3.1-8b-instant
(temperature=0.15, max_tokens=1024)
        │  JSON response
Catalog Validator
(hallucinated items dropped; URL always replaced with catalog URL)
        │
ChatResponse { reply, recommendations[], end_of_conversation }
```

---

## Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| LLM | llama-3.1-8b-instant via Groq | Free tier, ~2s latency, no GPU |
| BM25 | rank-bm25 | Pure Python, CPU-only, exact token matching |
| Semantic | FAISS-CPU + all-MiniLM-L6-v2 | CPU-safe, 384-dim, conceptual matching |
| Fusion | Reciprocal Rank Fusion (k=60) | No score calibration needed |
| Framework | FastAPI + Pydantic v2 | Schema-enforced API contract |
| Deployment | Render free tier | Stateless, single process |

---

## Project Structure

```
Recommendation_system/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app — /health and /chat endpoints
│   ├── models.py      # Pydantic schemas (non-negotiable API contract)
│   ├── catalog.py     # Catalog loader + BM25 + FAISS index
│   ├── retrieval.py   # Hybrid retrieval + RRF fusion
│   ├── agent.py       # LLM orchestration + JSON parser + validator
│   └── prompts.py     # System prompt with 7 behavioral rules
├── data/
│   ├── shl_product_catalog.json    # 377 SHL Individual Test Solutions
│   ├── catalog_embeddings.npy      # Pre-built embeddings (384-dim)
│   └── catalog_faiss.index         # Pre-built FAISS FlatIP index
├── scripts/
│   └── build_index.py   # Run once to build FAISS index
├── tests/
│   └── test_agent.py    # Unit tests + behaviour probes + Recall@10
├── docs/
│   └── approach.md      # Submission approach document
├── render.yaml          # Render deployment config
├── requirements.txt
├── .env.example
└── README.md
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/Algoistricky2004/Recommendation_system
cd Recommendation_system
```

### 2. Create and activate virtual environment
```bash
python -m venv SHL-Recommender
SHL-Recommender\Scripts\activate        # Windows
# source SHL-Recommender/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install httpx==0.27.2
pip install groq==0.11.0
pip install fastapi==0.115.5 uvicorn[standard]==0.32.1
pip install rank-bm25==0.2.2
pip install pydantic==2.10.3
pip install python-dotenv==1.0.1
pip install sentence-transformers==3.3.1
pip install faiss-cpu==1.9.0
pip install pytest==8.3.4 requests==2.32.3
```

### 4. Set environment variable
```bash
echo GROQ_API_KEY=your_key_here > .env
```
Get a free Groq key at https://console.groq.com

### 5. Build FAISS index (run once)
```bash
python scripts/build_index.py
```

### 6. Start the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit http://127.0.0.1:8000/docs for the interactive UI.

---

## Running Tests

```bash
# Unit tests only (no API key needed)
pytest tests/ -v -k "not requires_groq"

# Full suite including behaviour probes (requires GROQ_API_KEY)
pytest tests/ -v
```

**Test coverage:**
- 12 unit tests — catalog loading, BM25 search, JSON parsing, validation
- 8 behaviour probes — vague query, off-topic refusal, legal refusal, prompt injection, specific query, refinement, end-of-conversation, catalog-only URLs
- Recall@10 computed against labeled expected shortlists (threshold ≥ 0.30)

---

## Deployment (Render)

The `render.yaml` in the repo root configures Render automatically.

1. Connect this GitHub repo to Render (must be public)
2. Add `GROQ_API_KEY` as an environment variable in the Render dashboard
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

**Cold start note:** Render free tier spins down after inactivity.  
First request after sleep takes ~30–60 seconds — within the 2-minute allowance.  
If FAISS is unavailable on the deployment platform, the server falls back to BM25-only mode gracefully.

---

## Evaluation

| Metric | Method |
|--------|--------|
| Schema compliance | Every response validated against Pydantic model |
| Catalog-only URLs | Validator overwrites URL from catalog record — structurally hallucination-proof |
| Turn cap | Enforced at API level before LLM call |
| Recall@10 | Measured against labeled shortlists from 10 sample traces |
| Behaviour probes | 8 pytest probes covering all assignment-specified behaviours |

---

## Author

**Chirag Chawla**  
https://algoistricky2004.github.io/
