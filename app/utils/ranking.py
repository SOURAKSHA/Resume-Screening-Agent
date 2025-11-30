import os
import math
import numpy as np

# Read flags from env
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# OpenAI client: create lazily when needed so imports don't fail on startup.
# We import inside functions to avoid SDK init problems during cold start.
def get_openai_client():
    from openai import OpenAI
    # The SDK will read OPENAI_API_KEY from environment / Streamlit secrets
    return OpenAI()

# Local sentence-transformers model handle (set to None unless loaded)
_LOCAL_MODEL = None

def _ensure_local_model():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return

    try:
        # Lazy import so the package is only required if USE_LOCAL_EMBEDDINGS is True
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Local embeddings requested but `sentence_transformers` isn't installed. "
            "Either set USE_LOCAL_EMBEDDINGS = \"0\" in your Streamlit Secrets to use OpenAI embeddings "
            "or add sentence-transformers (and torch) to requirements.txt."
        ) from e

    # Load the model (this will download weights on first run)
    _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Return embedding for `text`. Uses local model if USE_LOCAL_EMBEDDINGS is True,
    otherwise uses OpenAI embeddings API.
    """
    text = text or ""
    text = text.strip()
    if text == "":
        return []

    if USE_LOCAL_EMBEDDINGS:
        # ensure local model is available (or raise a clear error)
        _ensure_local_model()
        return _LOCAL_MODEL.encode(text)

    # Use OpenAI embeddings (requires OPENAI_API_KEY in secrets)
    client = get_openai_client()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def cosine_similarity(a, b):
    # safe numeric cosine similarity
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
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
