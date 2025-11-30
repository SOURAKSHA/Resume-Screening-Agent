

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

RESUME_STORE = []

def index_resume(text, filename, metadata=None):
    """
    Store full resume text + embedding + metadata.
    """
    embedding = MODEL.encode(text)

    RESUME_STORE.append({
        "filename": filename,
        "text": text,          
        "embedding": embedding,
        "metadata": metadata
    })


def rank_resumes(job_description):
    """
    Rank resumes based on cosine similarity.
    Returns: list sorted by score descending.
    """
    if not RESUME_STORE:
        return []

    jd_embedding = MODEL.encode(job_description)

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


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
