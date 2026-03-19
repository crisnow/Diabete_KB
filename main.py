import os
import re
import pdfplumber
import nltk

# ------------------------------
# 1️⃣ Setup
# ------------------------------
# Folder containing your PDFs
PDF_FOLDER = "/Users/kristigong/Documents/GitHub/Diabete_KB/info_diabetes"  # <-- CHANGE THIS
import os

if not os.path.exists(PDF_FOLDER):
    raise FileNotFoundError(f"Folder not found: {PDF_FOLDER}")
else:
    print(f"Folder found: {PDF_FOLDER}")


import nltk
import ssl

# Bypass SSL issues (if any)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download the correct tokenizer resource
nltk.download('punkt_tab', quiet=True)

# Chunk size in words
CHUNK_SIZE = 200

# Make sure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# ------------------------------
# 2️⃣ Extract text from PDFs
# ------------------------------
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
all_texts = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    all_texts.append({"filename": pdf_file, "text": text})

print(f"Extracted text from {len(all_texts)} PDFs.")

# ------------------------------
# 3️⃣ Preprocess text
# ------------------------------
def clean_text(text):
    text = text.replace("\n", " ")       # Join lines
    text = re.sub(r"\s+", " ", text)     # Collapse multiple spaces
    return text.strip()

for doc in all_texts:
    doc['text'] = clean_text(doc['text'])
    doc['sentences'] = sent_tokenize(doc['text'])

# ------------------------------
# 4️⃣ Chunk text for RAG
# ------------------------------
def chunk_text(sentences, chunk_size=200):
    chunks = []
    chunk = []
    word_count = 0
    for sent in sentences:
        words = sent.split()
        chunk.append(sent)
        word_count += len(words)
        if word_count >= chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []
            word_count = 0
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

for doc in all_texts:
    doc['chunks'] = chunk_text(doc['sentences'], chunk_size=CHUNK_SIZE)

# ------------------------------
# 5️⃣ Output summary
# ------------------------------
for doc in all_texts:
    print(f"\nFile: {doc['filename']}")
    print(f"Total sentences: {len(doc['sentences'])}")
    print(f"Total chunks: {len(doc['chunks'])}")
    print(f"Example chunk: {doc['chunks'][0][:200]}...")  # show first 200 chars

# -----------------------------------------------------
#  Step 2 | Generate local Embeddings | no chatGPT API
# -----------------------------------------------------


import os
from sentence_transformers import SentenceTransformer
import json

# Load your chunked data from Step 1
# all_texts = [{'filename':..., 'chunks':[...]}]

# Step 2a: Load a local embedding model
# 'all-MiniLM-L6-v2' is small, fast, and works well
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings_data = []

for doc in all_texts:
    for i, chunk in enumerate(doc['chunks']):
        vector = model.encode(chunk)  # vector is a list of floats
        embeddings_data.append({
            "filename": doc['filename'],
            "chunk_index": i,
            "text": chunk,
            "embedding": vector.tolist()  # convert numpy array to list for JSON
        })

print(f"Created embeddings for {len(embeddings_data)} chunks.")

# Step 2b: Save embeddings for later use
with open("diabetes_chunks_embeddings_local.json", "w") as f:
    json.dump(embeddings_data, f)

print("Embeddings saved to diabetes_chunks_embeddings_local.json")


# -------------------------------------------------
# Step 3 | Build Retriever + Q&A System (Offline)
# -------------------------------------------------

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------
# 1. Load embeddings
# ------------------------------
with open("diabetes_chunks_embeddings_local.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
embeddings = [item["embedding"] for item in data]

# Convert to numpy array
embeddings = np.array(embeddings).astype("float32")

print(f"Loaded {len(texts)} chunks.")

# ------------------------------
# 2. Create FAISS index
# ------------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index created.")

# ------------------------------
# 3. Load embedding model (same as before!)
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# 4. Ask function
# ------------------------------
def ask_question(query, top_k=3):
    # Convert query to embedding
    query_vector = model.encode([query]).astype("float32")

    # Search similar chunks
    distances, indices = index.search(query_vector, top_k)

    print("\n🔍 Top relevant chunks:\n")
    retrieved_texts = []

    for i, idx in enumerate(indices[0]):
        print(f"--- Result {i+1} ---")
        print(texts[idx][:300], "...\n")
        retrieved_texts.append(texts[idx])

    # Combine retrieved context
    context = "\n".join(retrieved_texts)

    # Simple answer (no LLM yet)
    print("🧠 Combined context (for LLM):\n")
    print(context[:500], "...")

# ------------------------------
# 5. Test it
# ------------------------------
ask_question("What are symptoms of diabetes?")