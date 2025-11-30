import os
import numpy as np


# Read models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# -------------------------
# OpenAI Client Builder
# -------------------------
def get_openai_client():
    """
    Creates OpenAI client using the API key from Streamlit secrets.
    """
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")

    return OpenAI(api_key=key)


# -------------------------
# Embedding Function
# -------------------------
def embed_text(text: str):
    """
    Returns embedding for text using OpenAI.
    """
    text = (text or "").strip()
    if text == "":
        return []

    client = get_openai_client()

    try:
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]   # MUST be a list
        )
        return resp.data[0].embedding

    except Exception as e:
        print("Embedding Error:", e)
        return []


# -------------------------
# Cosine Similarity
# -------------------------
def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        return 0.0

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# -------------------------
# Resume Storage
# -------------------------
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
