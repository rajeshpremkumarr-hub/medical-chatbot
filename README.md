<img width="1677" height="921" alt="image" src="https://github.com/user-attachments/assets/278524d5-d0a6-4984-b1dd-88ecfbc4290c" />
# Medical AI Hub

A Streamlit-based “hub” app that combines:

- **Medical Q&A + RAG** over a MongoDB Atlas Vector Search knowledge base
- **Doctor search** (optional second vector store) with simple query constraints (rating/fee/nearby)
- **Prescription image analysis** (vision) that extracts *only medical info* and avoids personal data
- **Knowledge ingestion** (PDF/CSV/DOCX) into MongoDB Atlas via LangChain

The project is intentionally small and script-driven: one Streamlit app plus a couple of helper scripts to embed/verify data.

## Tech stack

- **UI**: Streamlit (`streamlit`)
- **Vector DB**: MongoDB Atlas + Atlas Vector Search (`pymongo`, `langchain-mongodb`)
- **Embeddings**: VoyageAI (`voyageai`) via a small LangChain `Embeddings` wrapper in the app
- **LLM providers**:
  - OpenAI Chat Completions (`https://api.openai.com/v1/chat/completions`)
  - OpenRouter Chat Completions (`https://openrouter.ai/api/v1/chat/completions`)
- **Document loaders (ingestion)**: LangChain community loaders (`PyPDFLoader`, `CSVLoader`, `UnstructuredWordDocumentLoader`)
- **Optional web snippets**: Google search results (`googlesearch-python`) + HTML meta image extraction

## How it’s implemented (high level)

### 1) Two vector stores (Medical + Doctors)
The Streamlit app creates **two** `MongoDBAtlasVectorSearch` instances:

- **Medical**: `DB_NAME` / `COLLECTION_NAME` / `INDEX_NAME`
- **Doctors (optional)**: `DOCTORS_DB_NAME` / `DOCTORS_COLLECTION_NAME` / `DOCTORS_INDEX_NAME`

Both use the same embedding provider (VoyageAI) and the same embedding field name by default (`embedding`).

### 2) Retrieval + chat
In “💬 Medical Assistant” mode, a user prompt triggers:

1. Vector search in **Medical**, **Doctors**, or **Both** (selectable in the sidebar)
2. Context formatting (different fields for medical docs vs doctor docs)
3. A Chat Completions call to the selected provider (OpenAI/OpenRouter)
4. The response is shown along with an expandable “Search from DB” log

For **Doctors**, the app additionally:

- Parses simple constraints from the query (examples: `rating >= 4`, `fee under 800`, `near 12.9,77.6`, `within 5 km`)
- Runs vector search with a higher candidate `k`, then filters down to the final `k`

### 3) Prescription image analysis (vision)
If you upload an image, the app sends a vision-capable chat request with instructions to:

- Extract **only medical information**
- **Avoid personal identifiers** (names, phone numbers, addresses, IDs) and replace with `[REDACTED]`
- Return **JSON only** in a strict schema

It then:

- Parses the JSON
- Optionally runs a small “second opinion” check against the Medical vector DB
- Optionally does a lightweight web price search (best-effort)

### 4) Knowledge ingestion
In “📂 Knowledge Ingestion” mode, uploaded PDF/CSV/DOCX files are loaded into LangChain `Document`s and added to the Medical vector store (`vector_store.add_documents`).

## Project layout

- [app.py](app.py) — Streamlit app (UI + RAG + vision + ingestion)
- [embed_existing.py](embed_existing.py) — one-time embedding backfill for existing MongoDB documents
- [verify_vector_search.py](verify_vector_search.py) — sanity check that vector search works end-to-end
- [test_openrouter.py](test_openrouter.py) — simple OpenRouter chat-completions request test
- [requirements.txt](requirements.txt) — Python dependencies
- [data/uploaded_dataset.csv](data/uploaded_dataset.csv) — sample dataset file
- `.env.template` — environment variable template

## Prerequisites

- Python 3.10+ (3.11 recommended)
- A **MongoDB Atlas** cluster with:
  - The target database/collection(s)
  - An **Atlas Vector Search** index created for each collection you query
- API keys:
  - `VOYAGE_API_KEY` (required for embeddings)
  - `OPENAI_API_KEY` and/or `OPENROUTER_API_KEY` (for chat/vision)

## Setup

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.template .env
```

Now edit `.env` and fill in the required values.

### Environment variables

The app reads these at startup:

- **LLM**
  - `OPENAI_API_KEY` — used when “OpenAI” is selected in the sidebar
  - `OPENROUTER_API_KEY` — used when “OpenRouter” is selected in the sidebar

- **Embeddings**
  - `VOYAGE_API_KEY` — required
  - `VOYAGE_EMBED_MODEL` — optional (default `voyage-4-large`)

- **MongoDB (Medical KB)**
  - `MONGODB_URI` — required
  - `DB_NAME` — default varies by script; set explicitly
  - `COLLECTION_NAME`
  - `INDEX_NAME` — Atlas Vector Search index name (default `vector_index`)
  - `MONGODB_TEXT_KEY` — the field used as “text” for retrieval display (default `Medicine Name`)
  - `MONGODB_EMBEDDING_KEY` — vector field name (default `embedding`)

- **MongoDB (Doctors KB, optional)**
  - `DOCTORS_DB_NAME`
  - `DOCTORS_COLLECTION_NAME`
  - `DOCTORS_INDEX_NAME`
  - `DOCTORS_TEXT_KEY` — default `name`
  - `DOCTORS_EMBEDDING_KEY` — default `embedding`

## MongoDB Atlas Vector Search index notes

This repo assumes:

- Each document has an embedding vector field (by default `embedding`)
- Your Atlas Vector Search index is configured to index that vector field

Because **embedding dimensionality depends on your Voyage model**, the safest way to confirm the correct dimensions is to run:

```bash
python verify_vector_search.py
```

If Atlas rejects queries due to a dimension mismatch, update the index dimensions to match the vectors stored in your documents.

## Running the app

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

In the sidebar:

- Pick **OpenAI** or **OpenRouter**
- Paste a key (or leave empty to use `OPENAI_API_KEY` / `OPENROUTER_API_KEY` from `.env`)
- Choose database scope: **Medical**, **Doctors**, or **Both**

## Populating embeddings (existing data)

If your MongoDB collection already has documents and you want to backfill embeddings into an `embedding` field:

```bash
python embed_existing.py
```

Useful env vars for [embed_existing.py](embed_existing.py):

- `TARGET_COLLECTION`: `medical` or `doctors` (default `medical`)
- `EMBEDDING_FIELD`: field to store vectors (default `embedding`)
- `FORCE_REEMBED`: `true` to overwrite existing vectors
- `BATCH_SIZE`, `MAX_BATCH_CHARS`: controls batch sizing
- `EMBED_TEXT_FIELD`: if set, embeds only that field; otherwise builds text from multiple fields

## Troubleshooting

- **“MongoDB Connection Error” on startup**
  - Ensure `MONGODB_URI` is set and reachable from your machine
  - If your Atlas network access is restricted, add your IP to Atlas Network Access

- **Vector search returns nothing / errors**
  - Confirm the Atlas Vector Search index exists and the index name matches `INDEX_NAME` / `DOCTORS_INDEX_NAME`
  - Run `python verify_vector_search.py` to validate the end-to-end setup

- **401 Invalid API key**
  - The app gives hints if you accidentally paste an OpenAI key into OpenRouter (or vice versa)
  - Prefer putting keys into `.env` and leaving the sidebar key empty

- **Vision parsing errors**
  - If the image contains names/phone numbers/addresses, crop or blur them and re-upload
  - The app is strict about JSON parsing; unclear images can produce invalid JSON

## Quick script checks

- Test OpenRouter connectivity:

```bash
python test_openrouter.py
```

- Verify vector search configuration:

```bash
python verify_vector_search.py
```
