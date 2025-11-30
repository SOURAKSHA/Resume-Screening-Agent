import os
import numpy as np

USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_full_key():
    p1 = os.getenv("OPENAI_API_KEY_PART1", "")
    p2 = os.getenv("OPENAI_API_KEY_PART2", "")
    return p1 + p2


def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=get_full_key())


_LOCAL_MODEL = None


def embed_text(text: str):
    text = (text or "").strip()
    if text == "":
        return []

    client = get_openai_client()

    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text   # FIXED: MUST NOT BE A LIST
        )
        return resp.data[0].embedding

    except Exception as e:
        print("Embedding Error:", e)
        return []


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        return 0.0

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


RESUME_STORE = []


def index_resume(text, filename, metadata=None):
    embedding = embed_text(text)
    RESUME_STORE.append({
        "filename": filename,
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    })


def rank_resumes(job_description):
    if not RESUME_STORE:
        return []

    jd_embedding = embed_text(job_description)

    results = []
    for r in RESUME_STORE:
        score = cosine_similarity(jd_embedding, r["embedding"])
        results.append({
            "filename": r["filename"],
            "text": r["text"],
            "metadata": r["metadata"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
