import os
import sys
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess
import torch
from PyQt5 import QtWidgets, QtCore

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

print("✅ Ready to answer questions.\n")

class QAWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ODIN QA Agent (Local Mistral version)")
        self.resize(800, 700)

        layout = QtWidgets.QVBoxLayout()

        # Smaller question field
        layout.addWidget(QtWidgets.QLabel("Ask your question:"))
        self.question_edit = QtWidgets.QTextEdit()
        self.question_edit.setPlaceholderText("Enter your question here...")
        self.question_edit.setFixedHeight(60)  # Smaller height
        layout.addWidget(self.question_edit)

        ask_button = QtWidgets.QPushButton("Ask")
        ask_button.clicked.connect(self.ask_question)
        layout.addWidget(ask_button)

        # Bigger answer field
        layout.addWidget(QtWidgets.QLabel("Answer:"))
        self.answer_text = QtWidgets.QTextEdit()
        self.answer_text.setReadOnly(True)
        self.answer_text.setFixedHeight(400)  # Bigger height
        layout.addWidget(self.answer_text)

        # Sources remain the same
        layout.addWidget(QtWidgets.QLabel("Sources:"))
        self.sources_text = QtWidgets.QTextEdit()
        self.sources_text.setReadOnly(True)
        layout.addWidget(self.sources_text)

        self.setLayout(layout)

    def ask_question(self):
        query = self.question_edit.toPlainText().strip()
        if not query:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please enter a question.")
            return

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
                    reply = QtWidgets.QMessageBox.question(
                        self, "Image Found",
                        f"Would you like to view {img_file}?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                    )
                    if reply == QtWidgets.QMessageBox.Yes:
                        subprocess.run(["open", img_path])
                context_parts.append(f"[{sources[i]}] - Image: {img_file}")
            else:
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

        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt.encode(),
            stdout=subprocess.PIPE
        )
        answer = result.stdout.decode().strip()

        self.answer_text.setPlainText(answer)
        self.sources_text.setPlainText("\n".join(cited_sources))

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = QAWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()