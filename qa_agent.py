# === qa_agent.py ===
# Script to load the index and answer questions from the user

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
index_path = "pdf_index.faiss"
metadata_path = "pdf_metadata.pkl"
model_name = "all-MiniLM-L6-v2"

# --- Load Model and Data ---
print("Loading model and index...")
model = SentenceTransformer(model_name)
index = faiss.read_index(index_path)

with open(metadata_path, "rb") as f:
    data = pickle.load(f)
    chunks = data["chunks"]
    sources = data["sources"]

print("Ready to answer questions. Type 'exit' to quit.\n")

while True:
    query = input("Q: ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k=10)

    context_parts = []
    cited_sources = set()
    for i in indices[0]:
        context_parts.append(f"[{sources[i]}]\n{chunks[i]}")
        cited_sources.add(sources[i])

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant. Answer the user's question completely and thoroughly using the context below.
Even if the information is split across multiple chunks, do your best to reconstruct the full answer.

Question: {query}

Context:
{context}

Answer:
"""

    print("\nAnswering with Mistral...\n")
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    print(result.stdout.decode())
    print("\nSources used:")
    for src in cited_sources:
        print(f"- {src}")
    print("-" * 60)