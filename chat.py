import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------
# 1. Load embeddings
# ------------------------------
with open("diabetes_chunks_embeddings_local.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
embeddings = np.array([item["embedding"] for item in data]).astype("float32")

print(f"Loaded {len(texts)} chunks.")

# ------------------------------
# 2. Load embedding model
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# 3. Similarity search (no FAISS)
# ------------------------------
def retrieve(query, top_k=3):
    query_vec = model.encode([query]).astype("float32")
    
    # cosine similarity
    similarities = np.dot(embeddings, query_vec.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [texts[i] for i in top_indices]

# ------------------------------
# 4. Simple answer generator
# ------------------------------
def generate_answer(query, context_chunks):
    context = " ".join(context_chunks)

    # VERY simple rule-based "LLM-like" response
    answer = f"""
Question: {query}

Answer:
Based on the available medical information:

{context[:500]}...

This information is for educational purposes only and not medical advice.
If you have concerns, please consult a healthcare professional.
"""
    return answer

# ------------------------------
# 5. Chat loop
# ------------------------------
def chat():
    print("\n🩺 Diabetes Assistant (type 'exit' to quit)\n")
    
    while True:
        query = input("You: ")
        
        if query.lower() == "exit":
            break
        
        retrieved = retrieve(query)
        answer = generate_answer(query, retrieved)
        
        print("\nAssistant:", answer)
        print("\n" + "-"*50)

# ------------------------------
# Run chatbot
# ------------------------------
chat()