# === qa_agent.py ===
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
index_path = "pdf_index.faiss"
metadata_path = "pdf_metadata.pkl"
model_name = "all-MiniLM-L6-v2"
image_folder = "extracted_images"

# --- Use MPS GPU if available ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔧 Using device: {device}")

# --- Load Model and Data ---
print("Loading model and index...")
model = SentenceTransformer(model_name, device=device)
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

    query_embedding = model.encode([query], convert_to_numpy=True, device=device)
    if query_embedding.shape[1] < 512:
        pad_width = 512 - query_embedding.shape[1]
        query_embedding = np.hstack([query_embedding, np.zeros((query_embedding.shape[0], pad_width))])

    _, indices = index.search(query_embedding, k=10)

    context_parts = []
    cited_sources = set()

    for i in indices[0]:
        if chunks[i].startswith("Image:"):
            img_file = chunks[i].replace("Image: ", "")
            img_path = os.path.join(image_folder, img_file)
            if os.path.exists(img_path):
                view = input(f"[{sources[i]}] - (Image: {img_file})\nWould you like to view this image? (y/n): ").strip().lower()
                if view == "y":
                    subprocess.run(["open", img_path])
            else:
                print(f"⚠️ Image file not found: {img_path}")
            context_parts.append(f"[{sources[i]}] - Image: {img_file}")
        else:
            context_parts.append(f"[{sources[i]}]\n{chunks[i]}")
        cited_sources.add(sources[i])

    # === Fallback: Show top 3 images if query likely wants images ===
    if any(word in query.lower() for word in ["image", "figure", "diagram"]):
        print("\nYour question mentions images. Here are a few images you might want to preview:")
        img_files = [f for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
        for img_file in img_files[:3]:
            img_path = os.path.join(image_folder, img_file)
            view = input(f"Would you like to view {img_file}? (y/n): ").strip().lower()
            if view == "y":
                subprocess.run(["open", img_path])

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