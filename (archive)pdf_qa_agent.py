import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

# Load and extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into chunks
def chunk_text(text, max_chars=500):
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind(".", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at+1].strip())
        text = text[split_at+1:]
    if text:
        chunks.append(text.strip())
    return chunks

# === Load and index the PDF ===
pdf_path = ".pdf"  # <-- Replace with your actual pdf folder
print(f"Loading PDF: {pdf_path}")
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === Question loop ===
print("\nPDF loaded and indexed. Ask your questions (type 'exit' to quit).\n")
while True:
    query = input("Q: ")
    if query.strip().lower() in ["exit", "quit"]:
        break

    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k=3)
    context = "\n\n".join([chunks[i] for i in indices[0]])

    prompt = f"""You are a helpful assistant. Answer the question below using only the given context.

Question: {query}

Context:
{context}

Answer:"""

    print("\nAnswering with Mistral...\n")
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    print(result.stdout.decode())
    print("-" * 60)