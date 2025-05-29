# === index_documents.py ===
# One-time script to process and index all PDFs in a folder (with images!)

import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be .* leaked semaphore objects")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Config ---
pdf_folder = "data"
image_folder = "extracted_images"
os.makedirs(image_folder, exist_ok=True)

output_index = "pdf_index.faiss"
output_metadata = "pdf_metadata.pkl"
model_name = "all-MiniLM-L6-v2"

# --- Use MPS GPU if available ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🔧 Using device: {device}")

# --- Load CLIP Model for images ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Map filenames to (doc title, source URL)
pdf_sources = {
    "00.pdf": ("D7.6 KPI Evolution Report (I to IX) v6", "https://google.com"),
    "01.pdf": ("D3.11 ODIN Platform v2", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5fc22c187&appId=PPGMS"),
    "02.pdf": ("D6.4 Models of emergency prediction and handling v1", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5f86eecc0&appId=PPGMS"),
    "03.pdf": ("D4.2 Implementation of Local CPS-IoT RSM", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ec767b39&appId=PPGMS"),
    "04.pdf": ("D7.4 KPI Evolution Report (I to IX) v3", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5fbfe1e12&appId=PPGMS"),
    "05.pdf": ("D8.2 ODIN Policy, Legal and Ethics Framework", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5e5642173&appId=PPGMS"),
    "06.pdf": ("D10.1 Trust building ecosystem", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5f0a1ea27&appId=PPGMS"),
    "07.pdf": ("D2.1 ODIN co-creation workshop and end-user requirements", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ea5bcecb&appId=PPGMS"),
    "08.pdf": ("D3.3 Hospital Knowledge Base and ODIN semantic ontology v2", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5fc217803&appId=PPGMS"),
    "09.pdf": ("D3.1 Operational framework", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5df29d067&appId=PPGMS"),
    "10.pdf": ("D3.2 Hospital Knowledge Base and ODIN semantic ontology v1", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ebb31806&appId=PPGMS"),
    "11.pdf": ("D7.3- KPIs Evolution Report (I to IX) v2", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5f0e09176&appId=PPGMS"),
    "12.pdf": ("D3.10 ODIN Platform v1", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5eba13582&appId=PPGMS"),
    "13.pdf": ("D7.2- KPIs Evolution Report M12", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ebcba76c&appId=PPGMS"),
    "14.pdf": ("D4.1 CPS-IoT Resource Management System Specification", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ebcb89c0&appId=PPGMS"),
    "15.pdf": ("D2.5 Innovative Procurement delivery", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ea1b6afd&appId=PPGMS"),
    "16.pdf": ("D6.1 The data model ecosystem for AI operations and modules implementation", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5f86cc446&appId=PPGMS"),
    "17.pdf": ("D4.6 Implementation of Advanced CPS-IoT RSM Features v2", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5fb88b8da&appId=PPGMS"),
    "18.pdf": ("D4.3 Implementation of Local CPS-IoT RSM Features v2", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5f9ce6fc5&appId=PPGMS"),
    "19.pdf": ("D4.5 Implementation of Advanced CPS-IoT RSM Features v1", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5e9d5d6ff&appId=PPGMS"),
    "20.pdf": ("D8.1 ODIN Webinar Series on “Data protection and health”", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5e5cdd605&appId=PPGMS"),
    "21.pdf": ("D1.3 Data Management Plan v2", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5fc21c9db&appId=PPGMS"),
    "22.pdf": ("D1.2 Data Management Plan", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5ea3beabe&appId=PPGMS"),
    "23.pdf": ("D9.1Project Website", "https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5e06685ec&appId=PPGMS"),
    "24.pdf": ("The WGD—A Dataset of Assembly Line Working Gestures for Ergonomic Analysis and Work-Related Injuries Prevention", "https://doi.org/10.3390/s21227600"),
    "25.pdf": ("The Effect of Artificial Intelligence on Patient-Physician Trust: Cross-Sectional Vignette Study", "https://doi.org/10.2196/50853"),
    "26.pdf": ("Optimizing cardiovascular risk assessment and registration in a developing cardiovascular learning health care system: Women benefit most", "https://doi.org/10.1371/journal.pdig.0000190"),
    "27.pdf": ("Semantic Ontologies for Complex Healthcare Structures: A Scoping Review", "https://doi.org/10.1109/access.2023.3248969"),
    "28.pdf": ("Comparison of the Response to an Electronic Versus a Traditional Informed Consent Procedure in Terms of Clinical Patient Characteristics: Observational Study", "https://doi.org/10.2196/54867"),
    "29.pdf": ("Integrating Physical and Cognitive Interaction Capabilities in a Robot-Aided Rehabilitation Platform", "https://doi.org/10.1109/jsyst.2023.3317504"),
    "30.pdf": ("Asking informed consent may lead to significant participation bias and suboptimal cardiovascular risk management in learning healthcare systems", "https://doi.org/10.1186/s12874-023-01924-6"),
}

# --- Helpers ---
def chunk_text(text, max_chars=1000):
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind(".", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at + 1].strip())
        text = text[split_at + 1:]
    if text:
        chunks.append(text.strip())
    return chunks

def extract_images_from_pdf(pdf_path, pdf_file):
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        images = doc.get_page_images(page_num)
        for img_index, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_filename = f"{os.path.splitext(pdf_file)[0]}_page{page_num+1}_img{img_index+1}.png"
            img_path = os.path.join(image_folder, img_filename)
            try:
                if pix.colorspace and pix.colorspace.n < 5:
                    pix.save(img_path)
                    image_paths.append((img_path, page_num+1))
                else:
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.save(img_path)
                    pix1 = None
                    image_paths.append((img_path, page_num+1))
            except Exception as e:
                print(f"⚠️ Error saving {img_filename}: {e}")
            finally:
                pix = None
    return image_paths

def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    outputs = clip_model.get_image_features(**inputs)
    return outputs.detach().cpu().numpy()[0]  # 512-dim embedding, back to CPU for FAISS

# --- Load Sentence-Transformer Model ---
model = SentenceTransformer(model_name, device=device)

# --- Process PDFs ---
all_chunks = []
sources = []
all_embeddings = []

for file in os.listdir(pdf_folder):
    if not file.endswith(".pdf"):
        continue
    pdf_path = os.path.join(pdf_folder, file)
    doc = fitz.open(pdf_path)
    title, url = pdf_sources.get(file, (os.path.splitext(file)[0], "Unknown Source"))

    # Text chunks
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            sources.append(f"{title} (Page {page_num}) - {url}")

    # Image chunks
    image_paths = extract_images_from_pdf(pdf_path, file)
    for img_path, page_num in image_paths:
        emb = embed_image(img_path)
        all_chunks.append(f"Image: {os.path.basename(img_path)}")
        sources.append(f"{title} (Page {page_num}, Image) - {url}")
        all_embeddings.append(emb)

# --- Embed Text Chunks ---
print(f"Embedding {len(all_chunks) - len(all_embeddings)} text chunks...")
text_embeddings = model.encode(
    all_chunks[:-len(all_embeddings)] if all_embeddings else all_chunks,
    convert_to_numpy=True,
    device=device
)

# --- Combine Embeddings (pad text to 512 dims for alignment) ---
if text_embeddings.shape[1] < 512:
    pad_width = 512 - text_embeddings.shape[1]
    text_embeddings = np.hstack([text_embeddings, np.zeros((text_embeddings.shape[0], pad_width))])

combined_embeddings = np.vstack([text_embeddings, np.array(all_embeddings)])

# --- Save FAISS Index ---
index = faiss.IndexFlatL2(combined_embeddings.shape[1])
index.add(combined_embeddings)
faiss.write_index(index, output_index)

# --- Save Metadata ---
with open(output_metadata, "wb") as f:
    pickle.dump({"chunks": all_chunks, "sources": sources}, f)

print("✅ Index and metadata saved, including images!")