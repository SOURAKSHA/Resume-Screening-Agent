import math
import os
from openai import OpenAI

RESUME_STORE = []


def get_client():
    """
    Create OpenAI client only when needed,
    after environment variables are loaded.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in environment!")
    return OpenAI(api_key=api_key)


def embed_text(text: str):
    client = get_client()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(v1, v2):
    dot = sum(a * b for a in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


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

    ranked = []
    for r in RESUME_STORE:
        score = cosine_similarity(jd_embedding, r["embedding"])
        ranked.append({
            "filename": r["filename"],
            "text": r["text"],
            "metadata": r["metadata"],
            "score": score
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
