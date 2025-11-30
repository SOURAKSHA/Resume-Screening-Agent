

import os
from dotenv import load_dotenv
load_dotenv()

USE_LOCAL = os.getenv("USE_LOCAL_EMBEDDINGS", "0") == "1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

if USE_LOCAL:
   
    from sentence_transformers import SentenceTransformer
    
    _local_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    """
    If USE_LOCAL_EMBEDDINGS=1 in .env -> use sentence-transformers locally.
    Otherwise use OpenAI embeddings.
    """
    if USE_LOCAL:
        vec = _local_model.encode(text)
       
        return vec.tolist()

   
    from openai import OpenAI
    client = OpenAI() 
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing in .env")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding
