"""embed_existing.py

One-time embedding migration script.

Embeds MongoDB documents using VoyageAI embeddings.
Default model: voyage-4-large.

Environment variables:
- MONGODB_URI (required)
- VOYAGE_API_KEY (required)
- VOYAGE_EMBED_MODEL (optional, default: voyage-4-large)
- EMBEDDING_FIELD (optional, default: embedding)
- FORCE_REEMBED (optional, true/false; default: false)
"""

import os
import time
import certifi
from pymongo import MongoClient
from pymongo.operations import UpdateOne
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
MONGODB_URI   = os.getenv("MONGODB_URI")
VOYAGE_KEY    = os.getenv("VOYAGE_API_KEY")
DB_NAME       = os.getenv("DB_NAME", "rajesh")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical")

# Optional: run against doctors collection using existing .env keys
TARGET_COLLECTION = (os.getenv("TARGET_COLLECTION") or "medical").strip().lower()
DOCTORS_DB_NAME = os.getenv("DOCTORS_DB_NAME", DB_NAME)
DOCTORS_COLLECTION_NAME = os.getenv("DOCTORS_COLLECTION_NAME", "doctors")
DOCTORS_GEOJSON_FIELD = os.getenv("DOCTORS_GEOJSON_FIELD", "location")

# Optional: embed from a single field if you maintain one (e.g. search_text)
EMBED_TEXT_FIELD = (os.getenv("EMBED_TEXT_FIELD") or "").strip()
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "256"))
MAX_BATCH_CHARS = int(os.getenv("MAX_BATCH_CHARS", "350000"))

EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large")
EMBEDDING_FIELD = os.getenv("EMBEDDING_FIELD", "embedding")
FORCE_REEMBED = os.getenv("FORCE_REEMBED", "false").strip().lower() in {"1", "true", "yes", "y"}


def _clean_embed_text(value: object, *, max_chars: int = 5000) -> str:
    s = str(value).replace("\n", " ").replace("\r", " ").strip()
    return s[:max_chars]


def _extract_embeddings(result: object) -> list:
    if hasattr(result, "embeddings"):
        return list(getattr(result, "embeddings"))
    if isinstance(result, dict):
        if "embeddings" in result:
            return list(result["embeddings"])
        if "data" in result:
            return [row["embedding"] for row in result["data"]]
    raise TypeError(f"Unexpected VoyageAI embed response type: {type(result)}")

def get_embeddings(texts: list) -> list:
    """Embed a list of strings with VoyageAI."""
    if not VOYAGE_KEY or not VOYAGE_KEY.strip():
        raise ValueError("VOYAGE_API_KEY is missing. Set it in your environment/.env file.")
    try:
        import voyageai  # type: ignore
    except Exception as e:
        raise ImportError("Missing dependency 'voyageai'. Install it first.") from e

    clean_texts = [_clean_embed_text(t) for t in texts]
    client = voyageai.Client(api_key=VOYAGE_KEY.strip())
    result = client.embed(clean_texts, model=EMBED_MODEL, input_type="document")
    return _extract_embeddings(result)


def _is_token_limit_error(err: Exception) -> bool:
    msg = str(err)
    return "max allowed tokens" in msg.lower() or "lower the number of tokens" in msg.lower()


def _process_batch(collection, batch_ids: list, batch_docs: list) -> int:
    vectors = get_embeddings(batch_docs)
    ops = [
        UpdateOne({"_id": _id}, {"$set": {EMBEDDING_FIELD: vec}})
        for _id, vec in zip(batch_ids, vectors)
    ]
    if ops:
        collection.bulk_write(ops, ordered=False)
    return len(batch_docs)


def _process_batch_with_split(collection, batch_ids: list, batch_docs: list) -> int:
    """Process a batch, splitting it if the embedding API complains about token limits."""
    if not batch_docs:
        return 0
    try:
        return _process_batch(collection, batch_ids, batch_docs)
    except Exception as e:
        if _is_token_limit_error(e) and len(batch_docs) > 1:
            mid = len(batch_docs) // 2
            left = _process_batch_with_split(collection, batch_ids[:mid], batch_docs[:mid])
            right = _process_batch_with_split(collection, batch_ids[mid:], batch_docs[mid:])
            return left + right

        # Avoid dumping huge payloads to stdout (especially vectors)
        msg = str(e)
        print(f"  ✗ Error: {type(e).__name__}: {msg[:500]}")
        time.sleep(2)
        return 0

def get_document_text(doc: dict) -> str:
    """Build embeddable text from *all* useful fields.

    Vector search only matches what you embedded. If you want queries to match
    on degree/experience/fee/location/etc, those must be part of the embedding input.
    """
    if EMBED_TEXT_FIELD:
        val = doc.get(EMBED_TEXT_FIELD)
        return _clean_embed_text(val) if val is not None else ""

    parts: list[str] = []
    skip_keys = {"_id", EMBEDDING_FIELD}

    for key, value in doc.items():
        if key in skip_keys or value is None:
            continue

        # Scalars
        if isinstance(value, (str, int, float, bool)):
            s = str(value).strip()
            if s:
                parts.append(f"{key}: {s[:1000]}")
            continue

        # Lists of small primitives (e.g., languages)
        if isinstance(value, list):
            if key == EMBEDDING_FIELD:
                continue
            if len(value) == 0:
                continue
            if len(value) <= 30 and all(isinstance(v, (str, int, float, bool)) for v in value):
                joined = ", ".join(str(v).strip() for v in value if str(v).strip())
                if joined:
                    parts.append(f"{key}: {joined[:1000]}")
            continue

        # GeoJSON-like dicts (especially for doctors location)
        if isinstance(value, dict):
            if key == DOCTORS_GEOJSON_FIELD:
                coords = value.get("coordinates")
                if isinstance(coords, list) and len(coords) == 2:
                    parts.append(f"{key}: {coords[1]}, {coords[0]}")
                continue

            # For other dicts, include a shallow summary if small
            if len(value) <= 8:
                flat_items = []
                for k2, v2 in value.items():
                    if isinstance(v2, (str, int, float, bool)):
                        s2 = str(v2).strip()
                        if s2:
                            flat_items.append(f"{k2}={s2}")
                if flat_items:
                    parts.append(f"{key}: " + ", ".join(flat_items)[:1000])

    return " | ".join(parts)

def run_migration():
    print("🔌 Connecting to MongoDB...")
    if not MONGODB_URI or not MONGODB_URI.strip():
        raise ValueError("MONGODB_URI is missing.")
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())

    db_name = DB_NAME
    coll_name = COLLECTION_NAME
    if TARGET_COLLECTION in {"doctors", "doctor", "doc"}:
        db_name = DOCTORS_DB_NAME
        coll_name = DOCTORS_COLLECTION_NAME
    elif TARGET_COLLECTION in {"medical", "med"}:
        db_name = DB_NAME
        coll_name = COLLECTION_NAME
    else:
        raise ValueError("TARGET_COLLECTION must be one of: medical | doctors")

    collection = client[db_name][coll_name]

    query = {} if FORCE_REEMBED else {EMBEDDING_FIELD: {"$exists": False}}
    total = collection.count_documents(query)
    target = "all documents" if FORCE_REEMBED else f"documents without '{EMBEDDING_FIELD}'"
    print(f"📦 Found {total} {target}.")
    print(f"🧭 Target: {TARGET_COLLECTION} | DB: {db_name} | Collection: {coll_name}")
    print(f"📐 Model: {EMBED_MODEL} | Field: {EMBEDDING_FIELD} | Batch size: {BATCH_SIZE}")
    if EMBED_TEXT_FIELD:
        print(f"🧾 EMBED_TEXT_FIELD: {EMBED_TEXT_FIELD}")
    print("🚀 Starting migration...\n")

    if total == 0:
        print("✅ All done!")
        client.close()
        return

    processed = 0
    batch_docs, batch_ids = [], []
    batch_chars = 0

    # Iterate over documents needing embeddings
    for doc in collection.find(query):
        text = get_document_text(doc)
        if text.strip():
            batch_docs.append(text)
            batch_ids.append(doc["_id"])
            batch_chars += len(text)

        if len(batch_docs) >= BATCH_SIZE or batch_chars >= MAX_BATCH_CHARS:
            wrote = _process_batch_with_split(collection, batch_ids, batch_docs)
            if wrote:
                processed += wrote
                pct = round(processed / total * 100, 1) if total else 100
                print(f"  ✓ {processed}/{total} ({pct}%)")

            batch_docs, batch_ids, batch_chars = [], [], 0
            time.sleep(0.1)

    # Final batch
    if batch_docs:
        wrote = _process_batch_with_split(collection, batch_ids, batch_docs)
        if wrote:
            processed += wrote
            print("  ✓ Finished last batch.")

    print("\n✅ Migration complete.")
    client.close()

if __name__ == "__main__":
    run_migration()
