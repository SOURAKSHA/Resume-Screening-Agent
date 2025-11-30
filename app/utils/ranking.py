from openai import OpenAI
import math

client = OpenAI()

RESUME_STORE = []


def embed_text(text: str):
    """
    Generate OpenAI embedding.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(v1, v2):
    """
    Pure Python cosine similarity (no numpy needed).
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def index_resume(text, filename, metadata=None):
    """
    Save resume text + embedding.
    """
    embedding = embed_text(text)

    RESUME_STORE.append({
        "filename": filename,
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    })


def rank_resumes(job_description):
    """
    Rank all indexed resumes by similarity to JD.
    """
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

    # highest score first
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
