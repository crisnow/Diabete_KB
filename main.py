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

