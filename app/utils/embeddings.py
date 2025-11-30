import os
from dotenv import load_dotenv

load_dotenv()

USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Load local model only if enabled
if USE_LOCAL:
    from sentence_transformers import SentenceTransformer
    _local_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    """
    If USE_LOCAL_EMBEDDINGS=1 -> use sentence-transformers.
    Otherwise -> use OpenAI embeddings.
    """
    text = (text or "").strip()
    if text == "":
        return []

    # Local embeddings (sentence-transformers)
    if USE_LOCAL:
        vec = _local_model.encode(text)
        return vec.tolist()

    # OpenAI embeddings
    from openai import OpenAI
    client = OpenAI()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text]   # must be list for embeddings
    )

    return response.data[0].embedding
