import os
import numpy as np

# Read flags from env
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# Lazily create OpenAI client when needed
def get_openai_client():
    from openai import OpenAI
    return OpenAI()


# Local model (loaded only when needed)
_LOCAL_MODEL = None


def _ensure_local_model():
    """Load sentence-transformers ONLY if local embeddings enabled."""
    global _LOCAL_MODEL

    if _LOCAL_MODEL is not None:
        return

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Local embeddings requested but `sentence_transformers` is not installed. "
            "Set USE_LOCAL_EMBEDDINGS='0' in secrets, or install sentence-transformers."
        ) from e

    _LOCAL_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    """
    Generate embedding for given text.
    Uses local model if enabled, else OpenAI embeddings.
    """
    text = (text or "").strip()
    if text == "":
        return []

    # Local embeddings
    if USE_LOCAL_EMBEDDINGS:
        _ensure_local_model()
        return _LOCAL_MODEL.encode(text)

    # OpenAI embeddings
    client = get_openai_client()
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]      # MUST BE LIST
    )
    return resp.data[0].embedding


def cosine_similarity(a, b):
    """Compute cosine similarity safely."""
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
    """Store resume text + embedding."""
    embedding = embed_text(text)
    RESUME_STORE.append({
        "filename": filename,
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    })


def rank_resumes(job_description):
    """Rank resumes using cosine similarity."""
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
