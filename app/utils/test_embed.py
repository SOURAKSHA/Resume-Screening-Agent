# test_embed.py
import os
from openai import OpenAI

key = os.environ.get("OPENAI_API_KEY")
print("OPENAI_API_KEY present:", bool(key))

client = OpenAI(api_key=key)

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Hello world"]
)

print("Embedding length:", len(resp.data[0].embedding))
print("First 5 dims:", resp.data[0].embedding[:5])
