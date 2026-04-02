import os
import io
import base64
import json
import re
import requests
import certifi
import streamlit as st
from PIL import Image
from pydantic import BaseModel, Field
from typing import Union, List, Any
from urllib.parse import urljoin

try:
    from googlesearch import search  # type: ignore
except Exception:
    search = None

# LangChain & MongoDB
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()


def _looks_like_pasted_error(value: str) -> bool:
    v = (value or "").strip().lower()
    return (
        v.startswith("vision error")
        or v.startswith("api error")
        or "incorrect api key" in v
        or "invalid_api_key" in v
    )


def _get_effective_llm_api_key(provider: str, typed_key: str) -> str:
    k = (typed_key or "").strip()
    if k and not _looks_like_pasted_error(k):
        return k
    if provider == "OpenAI":
        return (os.getenv("OPENAI_API_KEY") or "").strip()
    return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def _is_refusal_text(text: str) -> bool:
    t = (text or "").strip().lower()
    return (
        "i can't help" in t
        or "i cannot help" in t
        or "can't assist" in t
        or "cannot assist" in t
        or "sensitive information" in t
        or "personal details" in t
    )


def _extract_json_block(text: str) -> str | None:
    if not text:
        return None
    content = text.strip()
    if "```" in content:
        parts = content.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{") and block.endswith("}"):
                return block

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return None

def _clean_embed_text(value: Any, *, max_chars: int = 5000) -> str:
    s = str(value).replace("\n", " ").replace("\r", " ").strip()
    return s[:max_chars]


# --- VoyageAI Embeddings ---
class VoyageEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "voyage-4-large"):
        if not api_key or not api_key.strip():
            raise ValueError("VOYAGE_API_KEY is missing. Set it in your environment/.env file.")
        try:
            import voyageai  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing dependency 'voyageai'. Add it to requirements.txt and reinstall."
            ) from e

        self.model = model
        self._client = voyageai.Client(api_key=api_key.strip())

    def _extract_embeddings(self, result: Any) -> list:
        # Supports both SDK object response and dict response.
        if hasattr(result, "embeddings"):
            return list(result.embeddings)
        if isinstance(result, dict):
            if "embeddings" in result:
                return list(result["embeddings"])
            if "data" in result:
                return [row["embedding"] for row in result["data"]]
        raise TypeError(f"Unexpected VoyageAI embed response type: {type(result)}")

    def embed_documents(self, texts: list) -> list:
        clean_texts = [_clean_embed_text(t) for t in texts]
        result = self._client.embed(clean_texts, model=self.model, input_type="document")
        return self._extract_embeddings(result)

    def embed_query(self, text: str) -> list:
        clean_text = _clean_embed_text(text)
        result = self._client.embed([clean_text], model=self.model, input_type="query")
        return self._extract_embeddings(result)[0]

# --- Page Configuration ---
st.set_page_config(page_title="Medical AI Hub", page_icon="🏥", layout="wide")

# --- Constants & Environment ---
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "medical_assistant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_vectors")
INDEX_NAME = os.getenv("INDEX_NAME", "vector_index")

# Doctors collection (optional second vector store)
DOCTORS_DB_NAME = os.getenv("DOCTORS_DB_NAME", DB_NAME)
DOCTORS_COLLECTION_NAME = os.getenv("DOCTORS_COLLECTION_NAME", "doctors")
DOCTORS_INDEX_NAME = os.getenv("DOCTORS_INDEX_NAME", "vector_index_doctors")
DOCTORS_TEXT_KEY = os.getenv("DOCTORS_TEXT_KEY", "name")
DOCTORS_EMBEDDING_KEY = os.getenv(
    "DOCTORS_EMBEDDING_KEY", os.getenv("MONGODB_EMBEDDING_KEY", "embedding")
)

# --- Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_medicine_name" not in st.session_state:
    st.session_state.last_medicine_name = None
if "llm_history" not in st.session_state:
    # Stores plain user/assistant turns for conversational continuity.
    st.session_state.llm_history = []

# --- Sidebar: Settings ---
with st.sidebar:
    st.title("🏥 Settings")
    api_provider = st.radio("Select API Provider", ["OpenAI", "OpenRouter"], key="hub_provider")
    api_key = st.text_input("Enter API Key", type="password", 
                           value=st.session_state.get("saved_api_key", ""), 
                           key="hub_api_key")
    
    if api_provider == "OpenAI":
        model_name = "gpt-4o"
        base_url = "https://api.openai.com/v1/chat/completions"
    else:
        model_name = st.text_input("OpenRouter Model", value="openai/gpt-4o-mini", key="hub_model")
        base_url = "https://openrouter.ai/api/v1/chat/completions"

    kb_scope = st.selectbox("Search database", ["Medical", "Doctors", "Both"], index=0, key="hub_kb_scope")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.llm_history = []
        st.session_state.last_medicine_name = None
        st.rerun()

# --- MongoDB & Vector Store ---
@st.cache_resource(show_spinner=False)
def get_vector_stores():
    if not MONGODB_URI or not MONGODB_URI.strip():
        raise ValueError("MONGODB_URI is missing. Set it in your environment/.env file.")
    client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    embeddings = VoyageEmbeddings(
        api_key=os.getenv("VOYAGE_API_KEY"),
        model=os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large"),
    )

    medical_collection = client[DB_NAME][COLLECTION_NAME]
    medical_vs = MongoDBAtlasVectorSearch(
        collection=medical_collection,
        embedding=embeddings,
        index_name=INDEX_NAME,
        text_key=os.getenv("MONGODB_TEXT_KEY", "Medicine Name"),
        embedding_key=os.getenv("MONGODB_EMBEDDING_KEY", "embedding"),
        relevance_score_fn="cosine",
    )

    doctors_collection = client[DOCTORS_DB_NAME][DOCTORS_COLLECTION_NAME]
    doctors_vs = MongoDBAtlasVectorSearch(
        collection=doctors_collection,
        embedding=embeddings,
        index_name=DOCTORS_INDEX_NAME,
        text_key=DOCTORS_TEXT_KEY,
        embedding_key=DOCTORS_EMBEDDING_KEY,
        relevance_score_fn="cosine",
    )

    return {"medical": medical_vs, "doctors": doctors_vs}, client

try:
    vector_stores, mongo_client = get_vector_stores()
    vector_store = vector_stores["medical"]
except Exception as e:
    st.error(f"MongoDB Connection Error: {e}")
    st.stop()

# --- Helper Functions ---
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode('utf-8')

def call_ai_api(url, api_key, payload):
    k = api_key.strip()
    if not k:
        return {"error": "API Key is missing. Please enter it in the sidebar."}

    # Fast provider/key mismatch hints (common cases)
    if "openai.com" in (url or "") and k.startswith("sk-or-"):
        return {"error": "You selected OpenAI but the key looks like an OpenRouter key (sk-or-…). Switch provider to OpenRouter or use an OpenAI key."}
    if "openrouter.ai" in (url or "") and (k.startswith("sk-proj-") or k.startswith("sk-")):
        # OpenAI keys start with sk- / sk-proj-; OpenRouter keys typically start with sk-or-
        return {"error": "You selected OpenRouter but the key looks like an OpenAI key (sk-… / sk-proj-…). Switch provider to OpenAI or use an OpenRouter key."}
        
    headers = {
        "Authorization": f"Bearer {k}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", 
        "X-Title": "Medical AI Hub"
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 401:
            # Provide actionable guidance; do not echo keys.
            return {
                "error": (
                    "API Error 401: Invalid API key for the selected provider. "
                    "If you selected OpenAI, use an OpenAI key (env: OPENAI_API_KEY). "
                    "If you selected OpenRouter, use an OpenRouter key (env: OPENROUTER_API_KEY)."
                )
            }
        if response.status_code != 200:
            return {"error": f"API Error {response.status_code}: {response.text}"}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def search_prices(medicine_name, dosage):
    if search is None:
        return "Price data unavailable."
    query = f"buy {medicine_name} {dosage} online pharmacy price"
    try:
        results = ""
        for j in search(query, num_results=2, advanced=True):
             results += f"Source: {j.title}\nInfo: {j.description}\n\n"
        return results
    except:
        return "Price data unavailable."


def _extract_meta_image(html: str, base_url: str) -> str | None:
    if not html:
        return None
    head = html[:200_000]
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    ]
    for pat in patterns:
        m = re.search(pat, head, flags=re.IGNORECASE)
        if m:
            raw = (m.group(1) or "").strip()
            if not raw:
                continue
            absolute = urljoin(base_url, raw)
            if absolute.startswith("http://") or absolute.startswith("https://"):
                return absolute
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def internet_results(query: str, *, max_results: int = 3) -> list[dict]:
    """Fetch a few web snippets and try to extract a representative image per result.

    Uses googlesearch-python for results (title/url/description) and reads the
    target page HTML to find an OpenGraph/Twitter preview image.
    """
    if search is None:
        return []
    q = (query or "").strip()
    if not q:
        return []

    out: list[dict] = []
    # Keep this small to avoid slow UI.
    for r in search(q, num_results=max_results, advanced=True):
        title = getattr(r, "title", "") or ""
        description = getattr(r, "description", "") or ""
        url = getattr(r, "url", "") or ""
        if not url:
            continue

        image_url: str | None = None
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari",
                },
            )
            if resp.ok:
                image_url = _extract_meta_image(resp.text or "", url)
        except Exception:
            image_url = None

        out.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "description": description.strip(),
                "image_url": image_url,
            }
        )
    return out


def _tag_docs(docs, source: str):
    for d in docs or []:
        meta = dict(d.metadata or {})
        meta["_source"] = source
        d.metadata = meta
    return docs


def _interleave(a: list, b: list) -> list:
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return out


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return r * c


def _doc_lat_lng(meta: dict) -> tuple[float, float] | None:
    loc = meta.get("location")
    if isinstance(loc, dict):
        coords = loc.get("coordinates")
        if isinstance(coords, list) and len(coords) == 2:
            try:
                lng = float(coords[0])
                lat = float(coords[1])
                return (lat, lng)
            except Exception:
                pass

    lat = meta.get("latitude")
    lng = meta.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
        return (float(lat), float(lng))

    coord_str = meta.get("coordinates")
    if isinstance(coord_str, str) and "," in coord_str:
        try:
            a, b = coord_str.split(",", 1)
            return (float(a.strip()), float(b.strip()))
        except Exception:
            pass
    return None


def _parse_doctor_constraints(query: str) -> dict:
    """Extract simple constraints from the user's query.

    Supports patterns like:
    - "rating 4" / "rating >= 4"
    - "fee under 800" / "fee 800"
    - "near 12.9,77.6" / "12.9 77.6"
    """
    q = (query or "").lower()
    out: dict = {}

    m = re.search(r"rating\s*(?:>=|>|at\s*least)?\s*([0-5](?:\.\d)?)", q)
    if m:
        try:
            out["min_rating"] = float(m.group(1))
        except Exception:
            pass

    m = re.search(r"(?:fee|fees|consultation)\s*(?:<=|<|under|below)?\s*₹?\s*(\d{2,6})", q)
    if m:
        try:
            out["max_fee"] = int(m.group(1))
        except Exception:
            pass

    m = re.search(r"near\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", q)
    if not m:
        m = re.search(r"\b(-?\d{1,2}\.\d+)\s+(-?\d{1,3}\.\d+)\b", q)
    if m:
        try:
            out["lat"] = float(m.group(1))
            out["lng"] = float(m.group(2))
        except Exception:
            pass

    # Location phrase like: "in koramangala", "near indiranagar"
    m = re.search(r"\b(?:in|at|near)\s+([a-z][a-z\s\-]{2,40})", q)
    if m:
        loc = (m.group(1) or "").strip()
        if loc:
            # stop at common constraint keywords
            loc = re.split(r"\b(?:fee|fees|rating|under|below|less|greater|with)\b", loc, maxsplit=1)[0].strip()
            if loc and len(loc) <= 40:
                out["location_text"] = loc

    m = re.search(r"\bwithin\s*(\d+(?:\.\d+)?)\s*(?:km|kilometers|kilometres)\b", q)
    if m:
        try:
            out["radius_km"] = float(m.group(1))
        except Exception:
            pass

    return out


def _filter_doctor_docs(docs: list, *, query: str) -> list:
    """Apply rating/fee/location/nearby filtering to doctor docs."""
    if not docs:
        return docs

    parsed = _parse_doctor_constraints(query)

    min_rating = float(parsed.get("min_rating") or 0.0)
    max_fee = int(parsed.get("max_fee") or 0)
    location_text = (parsed.get("location_text") or "").strip().lower()

    use_geo = ("lat" in parsed and "lng" in parsed)
    lat = parsed.get("lat")
    lng = parsed.get("lng")

    radius_km = float(parsed.get("radius_km") or 0.0)
    if use_geo and radius_km <= 0:
        radius_km = 10.0

    filtered = []
    for d in docs:
        meta = d.metadata or {}

        rating = meta.get("rating")
        if isinstance(rating, (int, float)) and rating < min_rating:
            continue

        fee = meta.get("consultation_fee")
        if max_fee > 0 and isinstance(fee, (int, float)) and fee > max_fee:
            continue

        if location_text:
            hay = " ".join(
                str(meta.get(k, "") or "")
                for k in ["bangalore_location", "city", "state", "specialty", "speciality", "degree"]
            ).lower()
            if location_text not in hay:
                continue

        if use_geo and isinstance(lat, (int, float)) and isinstance(lng, (int, float)) and radius_km > 0:
            ll = _doc_lat_lng(meta)
            if not ll:
                continue
            dist = _haversine_km(float(lat), float(lng), ll[0], ll[1])
            # store for display
            meta2 = dict(meta)
            meta2["_distance_km"] = round(dist, 2)
            d.metadata = meta2
            if dist > radius_km:
                continue

        filtered.append(d)

    return filtered


def _doctor_vector_query(query: str) -> str:
    """Remove structured constraint tokens so vector search stays semantically focused."""
    q = (query or "").strip()
    if not q:
        return q
    q2 = q
    q2 = re.sub(r"\brating\s*(?:>=|>|at\s*least)?\s*[0-5](?:\.\d)?\b", " ", q2, flags=re.IGNORECASE)
    q2 = re.sub(r"\b(?:fee|fees|consultation)\s*(?:<=|<|under|below)?\s*₹?\s*\d{2,6}\b", " ", q2, flags=re.IGNORECASE)
    q2 = re.sub(r"\bnear\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\b", " ", q2, flags=re.IGNORECASE)
    q2 = re.sub(r"\b-?\d{1,2}\.\d+\s+-?\d{1,3}\.\d+\b", " ", q2)
    q2 = re.sub(r"\s+", " ", q2).strip()
    return q2 or q


def search_vector_db(query: str, *, k: int, scope: str) -> tuple[list, list[str]]:
    """Search medical/doctors/both. Returns (docs, errors)."""
    errors: list[str] = []
    scope = (scope or "Medical").strip()

    if scope == "Doctors":
        try:
            candidate_k = max(50, k * 10)
            vq = _doctor_vector_query(query)
            docs = vector_stores["doctors"].similarity_search(vq, k=candidate_k)
            docs = _filter_doctor_docs(_tag_docs(docs, "doctors"), query=query)[:k]
            return docs, errors
        except Exception as e:
            errors.append(f"Doctors search error: {e}")
            return [], errors

    if scope == "Both":
        med_k = max(1, (k + 1) // 2)
        doc_k = max(1, k // 2)
        med_docs: list = []
        doc_docs: list = []
        try:
            med_docs = _tag_docs(vector_stores["medical"].similarity_search(query, k=med_k), "medical")
        except Exception as e:
            errors.append(f"Medical search error: {e}")
        try:
            candidate_k = max(50, doc_k * 10)
            vq = _doctor_vector_query(query)
            doc_docs = _tag_docs(vector_stores["doctors"].similarity_search(vq, k=candidate_k), "doctors")
            doc_docs = _filter_doctor_docs(doc_docs, query=query)[:doc_k]
        except Exception as e:
            errors.append(f"Doctors search error: {e}")
        return _interleave(med_docs, doc_docs), errors

    # Default: Medical
    try:
        docs = vector_stores["medical"].similarity_search(query, k=k)
        return _tag_docs(docs, "medical"), errors
    except Exception as e:
        errors.append(f"Medical search error: {e}")
        return [], errors


def _format_docs_for_context(docs):
    lines = []
    for doc in docs:
        name = (doc.page_content or "").strip()
        meta = doc.metadata or {}
        source = meta.get("_source", "medical")
        parts = []

        def _fmt_meta_value(v: Any) -> str | None:
            if v is None:
                return None
            if isinstance(v, str):
                s = v.strip()
                return s or None
            if isinstance(v, (int, float, bool)):
                return str(v)
            if isinstance(v, list) and len(v) <= 30 and all(
                isinstance(x, (str, int, float, bool)) for x in v
            ):
                joined = ", ".join(str(x).strip() for x in v if str(x).strip())
                return joined or None
            if isinstance(v, dict):
                coords = v.get("coordinates")
                if isinstance(coords, list) and len(coords) == 2:
                    # GeoJSON: [lng, lat]
                    return f"{coords[1]}, {coords[0]}"
            return None

        if source == "doctors":
            if name:
                parts.append(f"Doctor: {name}")
            for key in [
                "specialty",
                "speciality",
                "degree",
                "department",
                "hospital",
                "clinic",
                "city",
                "state",
                "experience",
                "consultation_fee",
                "rating",
                "bangalore_location",
                "google_maps_link",
                "location",
                "languages",
            ]:
                val = _fmt_meta_value(meta.get(key))
                if val:
                    parts.append(f"{key}: {val}")
        else:
            if name:
                parts.append(f"Medicine Name: {name}")
            for key in ["Composition", "Uses", "Side_effects", "Manufacturer"]:
                val = meta.get(key)
                if isinstance(val, str) and val.strip():
                    parts.append(f"{key}: {val.strip()}")

        if parts:
            lines.append("\n".join(parts))
    return "\n\n".join(lines)


def _format_docs_for_log(docs, *, max_items: int = 4, errors: list[str] | None = None) -> str:
    if not docs:
        if errors:
            return "\n".join([f"- ⚠️ {e}" for e in errors])
        return "- No matches found in DB."
    out = []
    if errors:
        out.extend([f"- ⚠️ {e}" for e in errors])
    for doc in docs[:max_items]:
        name = (doc.page_content or "").strip() or "(no name)"
        meta = doc.metadata or {}
        source = meta.get("_source")
        comp = meta.get("Composition")
        prefix = ""
        if source == "doctors":
            prefix = "[Doctors] "
        elif source == "medical":
            prefix = "[Medical] "

        if source == "doctors":
            extra_bits = []
            spec = meta.get("specialty") or meta.get("speciality")
            loc = meta.get("bangalore_location") or meta.get("city")
            rating = meta.get("rating")
            fee = meta.get("consultation_fee")
            dist = meta.get("_distance_km")
            if isinstance(spec, str) and spec.strip():
                extra_bits.append(spec.strip())
            if isinstance(loc, str) and loc.strip():
                extra_bits.append(loc.strip())
            if isinstance(rating, (int, float)):
                extra_bits.append(f"rating {rating}")
            if isinstance(fee, (int, float)):
                extra_bits.append(f"fee {fee}")
            if isinstance(dist, (int, float)):
                extra_bits.append(f"{dist} km")
            suffix = (" — " + ", ".join(extra_bits)) if extra_bits else ""
            out.append(f"- **{prefix}{name}**{suffix}")
        else:
            if isinstance(comp, str) and comp.strip():
                out.append(f"- **{prefix}{name}** — {comp.strip()}")
            else:
                out.append(f"- **{prefix}{name}**")
    return "\n".join(out)

# --- Data Ingestion ---
def ingest_file(uploaded_file):
    temp_path = f"data/{uploaded_file.name}"
    os.makedirs("data", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    if uploaded_file.name.endswith(".csv"):
        loader = CSVLoader(temp_path)
    elif uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith((".doc", ".docx")):
        loader = UnstructuredWordDocumentLoader(temp_path)
    else:
        st.error("Unsupported file format for indexing.")
        return

    documents = loader.load()
    vector_store.add_documents(documents)
    os.remove(temp_path)
    st.success(f"Successfully indexed '{uploaded_file.name}' into MongoDB!")

# --- Main Interface ---
st.title("🛡️ Unified Medical AI Hub")
st.write("A single interface for Prescription Analysis, Medical RAG, and AI Recommendations powered by MongoDB Atlas.")

mode = st.radio("Mode", ["💬 Medical Assistant", "📂 Knowledge Ingestion"], horizontal=True)

if mode == "📂 Knowledge Ingestion":
    st.subheader("Add to Knowledge Base")
    st.info("Upload medical documents (PDF/CSV/DOCX) to make them searchable in the hub.")
    ingest_files = st.file_uploader("Upload Medical Docs", type=["pdf", "csv", "docx"], accept_multiple_files=True)
    if st.button("🚀 Index Documents"):
        if ingest_files:
            with st.spinner("Indexing to MongoDB Atlas..."):
                for f in ingest_files:
                    ingest_file(f)
        else:
            st.warning("Please upload files first.")

else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message.get("content")
            if isinstance(content, dict) and message["role"] == "assistant":
                answer = content.get("answer", "")
                db_log = content.get("db_log", "")
                if answer:
                    st.markdown(answer)
                if db_log:
                    with st.expander("🔎 Search from DB", expanded=False):
                        st.markdown(db_log)
            else:
                st.markdown(content)
            if "image" in message:
                st.image(message["image"])

    image_upload = st.file_uploader("🖼️ Analyze Prescription", type=["jpg", "jpeg", "png"])
    prompt = st.chat_input("Ask a medical question, check symptoms, or upload a prescription below...")

    # Processing Logic
    if prompt or image_upload:
        effective_api_key = _get_effective_llm_api_key(api_provider, api_key)
        if not effective_api_key:
            st.error("👈 Please enter your API Key in the sidebar.")
        else:
            # Add user message
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            if image_upload:
                st.session_state.messages.append({"role": "user", "content": "Analyzing this prescription...", "image": image_upload})
                with st.chat_message("user"):
                    st.image(image_upload)

            # Response Logic
            with st.chat_message("assistant"):
                with st.spinner("AI is thinking..."):
                    response_content = ""
                    
                    if image_upload:
                        # 1. Vision OCR & Extraction
                        b64_img = encode_image(image_upload)
                        payload = {
                            "model": model_name,
                            "messages": [{"role": "user", "content": [
                                {"type": "text", "text": "Extract ONLY medical information from this prescription image. IMPORTANT: Do NOT transcribe or identify any personal information (patient/doctor names, phone numbers, addresses, IDs). Ignore or redact those as '[REDACTED]'.\n\nReturn JSON ONLY (no markdown, no extra text) in this exact schema:\n{\n  \"diagnosed_cause\": string,\n  \"precautions\": string,\n  \"medicines\": [\n    {\n      \"name\": string,\n      \"dosage\": string\n    }\n  ]\n}\n\nIf you cannot confidently read something, use an empty string for that field."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]}]
                        }
                        if api_provider == "OpenAI":
                            payload["response_format"] = {"type": "json_object"}
                        res = call_ai_api(base_url, effective_api_key, payload)
                        
                        if "error" in res:
                            response_content = f"Vision Error: {res['error']}"
                        else:
                            content = res["choices"][0]["message"]["content"]
                            # Clean and Parse
                            try:
                                if _is_refusal_text(content):
                                    raise ValueError(
                                        "The model refused to read the image because it may contain personal information. "
                                        "Try cropping/blurring names/phone/address and re-upload."
                                    )

                                json_text = _extract_json_block(content) or content
                                data = json.loads(json_text)
                                
                                response_content = f"### 💊 Prescription Results\n**Diagnosis:** {data.get('diagnosed_cause')}\n**Precautions:** {data.get('precautions')}\n\n"
                                
                                # Search MongoDB for Medical Second Opinion
                                st.write("🔍 Searching MongoDB for second opinion...")
                                try:
                                    docs, _errs = search_vector_db(
                                        f"Condition: {data.get('diagnosed_cause')}",
                                        k=2,
                                        scope="Medical",
                                    )
                                    context = _format_docs_for_context(docs) if docs else "No specific matches in DB."
                                    
                                    so_payload = {
                                        "model": model_name,
                                        "messages": [
                                            {"role": "system", "content": "Compare the diagnosis and medicines with the following medical record context. Highlight matches or areas of concern."},
                                            {"role": "user", "content": f"Context: {context}\n\nPrescription: {content}"}
                                        ]
                                    }
                                    so_res = call_ai_api(base_url, effective_api_key, so_payload)
                                    so_text = so_res["choices"][0]["message"]["content"] if "choices" in so_res else "Could not generate second opinion."
                                    response_content += f"**🛡️ Database Second Opinion:**\n{so_text}\n\n"
                                except:
                                    response_content += "**🛡️ Database Second Opinion:** (Index not yet created on Atlas)\n\n"

                                # Append Medicines & Prices
                                response_content += "#### 💰 Pricing Estimates\n"
                                for med in data.get("medicines", []):
                                    price_raw = search_prices(med['name'], med['dosage'])
                                    price_sum_payload = {
                                        "model": model_name,
                                        "messages": [{"role": "user", "content": f"Brief price range for {med['name']} from: {price_raw}"}]
                                    }
                                    p_res = call_ai_api(base_url, effective_api_key, price_sum_payload)
                                    p_text = p_res["choices"][0]["message"]["content"] if "choices" in p_res else "Price search failed."
                                    response_content += f"- **{med['name']}**: {p_text}\n"

                            except Exception as e:
                                response_content = (
                                    f"Parsing Error: {e}\n\n"
                                    "Tip: If the image contains names/phone/address, crop or blur them and try again.\n\n"
                                    f"Raw Response: {content}"
                                )

                    elif prompt:
                        # 2. General RAG Search against MongoDB
                        try:
                            last_name = st.session_state.get("last_medicine_name")
                            lowered = (prompt or "").strip().lower()
                            pronoun_followup = any(
                                p in lowered.split()
                                for p in ["it", "this", "that", "its", "them", "these", "those"]
                            )

                            effective_query = prompt
                            if last_name and pronoun_followup and last_name.lower() not in lowered:
                                effective_query = f"{last_name}: {prompt}"

                            docs, errs = search_vector_db(effective_query, k=4, scope=kb_scope)
                            for d in docs or []:
                                meta = d.metadata or {}
                                if meta.get("_source") == "medical" and (d.page_content or "").strip():
                                    st.session_state.last_medicine_name = (d.page_content or "").strip()
                                    break

                            context = _format_docs_for_context(docs) if docs else ""
                            db_log = _format_docs_for_log(docs, errors=errs)

                            # Build a ChatGPT-like conversational payload: include prior turns.
                            history = st.session_state.llm_history[-12:]
                            
                            rag_payload = {
                                "model": model_name,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a helpful medical assistant. "
                                            "First, use the provided DATABASE CONTEXT if it contains the answer. "
                                            "If the database context is empty or does not contain enough information, "
                                            "answer using general medical knowledge and clearly label it as: 'General (not from DB): ...'. "
                                            "Do not repeat the database context verbatim or mention 'DATABASE CONTEXT' in your response. "
                                            "Keep answers concise and avoid giving definitive diagnosis; suggest consulting a clinician when appropriate."
                                        ),
                                    },
                                    *history,
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Context (use silently):\n{context if context.strip() else '(none)'}\n\n"
                                            f"User question: {prompt}"
                                        ),
                                    },
                                ]
                            }
                            rag_res = call_ai_api(base_url, effective_api_key, rag_payload)
                            llm_answer = rag_res["choices"][0]["message"]["content"] if "choices" in rag_res else rag_res.get("error", "Failed to get response.")

                            # Internet snippets + images (best-effort)
                            web_query = st.session_state.get("last_medicine_name") or effective_query
                            web_results = internet_results(f"{web_query} medicine")

                            # Update conversation memory (store only the plain turn text).
                            st.session_state.llm_history.append({"role": "user", "content": prompt})
                            st.session_state.llm_history.append({"role": "assistant", "content": llm_answer})
                            if len(st.session_state.llm_history) > 40:
                                st.session_state.llm_history = st.session_state.llm_history[-40:]

                            response_content = {
                                "answer": llm_answer,
                                "db_log": db_log,
                                "web_results": web_results,
                            }
                        except Exception as e:
                            response_content = f"Search Error (Check if your MongoDB Atlas Vector Index is created): {e}"

                    if isinstance(response_content, dict):
                        st.markdown(response_content.get("answer", ""))
                        with st.expander("🔎 Search from DB", expanded=False):
                            st.markdown(response_content.get("db_log", ""))

                        web_results = response_content.get("web_results") or []
                        if web_results:
                            with st.expander("🌐 Internet results", expanded=False):
                                for item in web_results:
                                    title = item.get("title") or item.get("url") or "Result"
                                    url = item.get("url") or ""
                                    desc = item.get("description") or ""
                                    img = item.get("image_url")

                                    if url:
                                        st.markdown(f"[{title}]({url})")
                                    else:
                                        st.markdown(f"{title}")
                                    if desc:
                                        st.caption(desc)
                                    if img:
                                        st.image(img, caption=title)
                    else:
                        st.markdown(response_content)

                    st.session_state.messages.append({"role": "assistant", "content": response_content})
