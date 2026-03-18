# Install dependencies first if you don't have them
# pip install requests beautifulsoup4 nltk

import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')  # needed for sentence tokenizer

# -----------------------------
# 1️⃣ List of diabetes-related URLs
urls = [
    "https://www.diabetes.org/diabetes/overview",
    "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "https://www.cdc.gov/diabetes/library/index.html"
]

# -----------------------------
# 2️⃣ Function to extract text from webpage
def extract_text_from_url(url):
    try:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        # Extract paragraphs
        paragraphs = [p.text for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# -----------------------------
# 3️⃣ Chunking function
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of approx chunk_size words with overlap.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # overlap words
    return chunks

# -----------------------------
# 4️⃣ Main process
all_chunks = []

for url in urls:
    text = extract_text_from_url(url)
    if text:
        chunks = chunk_text(text, chunk_size=300, overlap=50)  # you can adjust sizes
        print(f"{len(chunks)} chunks created from {url}")
        all_chunks.extend(chunks)

# -----------------------------
# 5️⃣ Preview first 3 chunks
for i, chunk in enumerate(all_chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk[:500])  # show first 500 characters
    