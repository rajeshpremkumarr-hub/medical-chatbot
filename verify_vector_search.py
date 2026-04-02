import os

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

import voyageai
from langchain_mongodb import MongoDBAtlasVectorSearch


def _mask(s: str | None) -> str:
    if not s:
        return "<missing>"
    if len(s) <= 8:
        return "<set>"
    return s[:4] + "…" + s[-4:]


def main() -> None:
    load_dotenv(dotenv_path=".env")

    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME")
    coll_name = os.getenv("COLLECTION_NAME")

    index_name = os.getenv("INDEX_NAME", "vector_index")
    text_key = os.getenv("MONGODB_TEXT_KEY", "Medicine Name")
    embed_key = os.getenv("MONGODB_EMBEDDING_KEY", "embedding")

    doctors_db_name = os.getenv("DOCTORS_DB_NAME", db_name or "")
    doctors_coll_name = os.getenv("DOCTORS_COLLECTION_NAME", "doctors")
    doctors_index_name = os.getenv("DOCTORS_INDEX_NAME", "vector_index_doctors")
    doctors_text_key = os.getenv("DOCTORS_TEXT_KEY", "name")
    doctors_embed_key = os.getenv("DOCTORS_EMBEDDING_KEY", embed_key)

    target = (os.getenv("TARGET_COLLECTION") or "medical").strip().lower()

    model = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large")
    api_key = os.getenv("VOYAGE_API_KEY")

    if not uri or not db_name or not coll_name:
        raise SystemExit(
            "Missing MONGODB_URI / DB_NAME / COLLECTION_NAME in .env"
        )
    if not api_key:
        raise SystemExit("Missing VOYAGE_API_KEY in .env")

    print("Config")
    print("- MONGODB_URI:", _mask(uri))
    print("- Voyage model:", model)
    print("- TARGET_COLLECTION:", target)

    client = MongoClient(uri, tlsCAFile=certifi.where())

    def run_one(
        *,
        label: str,
        db: str,
        coll: str,
        index: str,
        tkey: str,
        ekey: str,
        queries: list[str],
    ) -> None:
        print(f"\n=== {label} ===")
        print("- DB:", db)
        print("- Collection:", coll)
        print("- Index:", index)
        print("- text_key:", tkey)
        print("- embedding_key:", ekey)

        collection = client[db][coll]
        total = collection.estimated_document_count()
        print("- estimated_document_count:", total)

        sample = collection.find_one({tkey: {"$exists": True}})
        if not sample:
            print(f"No documents contain text_key={tkey!r}.")
            return

        emb = sample.get(ekey)
        if isinstance(emb, list):
            print("- embedding dim:", len(emb))
        else:
            print("- embedding type:", type(emb).__name__)

        vc = voyageai.Client(api_key=api_key)

        class VoyageEmbeddings:
            def embed_documents(self, texts):
                r = vc.embed(texts, model=model, input_type="document")
                return list(r.embeddings)

            def embed_query(self, text):
                r = vc.embed([text], model=model, input_type="query")
                return list(r.embeddings)[0]

        vs = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=VoyageEmbeddings(),
            index_name=index,
            text_key=tkey,
            embedding_key=ekey,
            relevance_score_fn="cosine",
        )

        print("\nVector search")
        for q in queries:
            docs = vs.similarity_search(q, k=3)
            print("\nQuery:", q)
            print("- hits:", len(docs))
            for i, d in enumerate(docs, 1):
                print(f"  {i}. {str(d.page_content)[:140]!r}")

    medical_queries = [
        "composition of Augmentin 625 Duo Tablet",
        "Augmentin 625 Duo Tablet",
        "amoxicillin clavulanic acid composition",
    ]
    doctors_queries = [
        "cardiologist in hyderabad",
        "pediatrician near me",
        "dermatologist clinic",
    ]

    if target in {"medical", "med"}:
        run_one(
            label="MEDICAL",
            db=db_name,
            coll=coll_name,
            index=index_name,
            tkey=text_key,
            ekey=embed_key,
            queries=medical_queries,
        )
    elif target in {"doctors", "doctor", "doc"}:
        if not doctors_db_name:
            raise SystemExit("Missing DOCTORS_DB_NAME (or DB_NAME) in .env")
        run_one(
            label="DOCTORS",
            db=doctors_db_name,
            coll=doctors_coll_name,
            index=doctors_index_name,
            tkey=doctors_text_key,
            ekey=doctors_embed_key,
            queries=doctors_queries,
        )
    elif target in {"both", "all"}:
        run_one(
            label="MEDICAL",
            db=db_name,
            coll=coll_name,
            index=index_name,
            tkey=text_key,
            ekey=embed_key,
            queries=medical_queries,
        )
        if doctors_db_name:
            run_one(
                label="DOCTORS",
                db=doctors_db_name,
                coll=doctors_coll_name,
                index=doctors_index_name,
                tkey=doctors_text_key,
                ekey=doctors_embed_key,
                queries=doctors_queries,
            )
        else:
            print("\nSkipping DOCTORS (missing DOCTORS_DB_NAME).")
    else:
        raise SystemExit(
            "TARGET_COLLECTION must be one of: medical | doctors | both"
        )


if __name__ == "__main__":
    main()
