# ==========================================================
# PROJECT: Enterprise-Level RAG System (Document AI)
# ==========================================================
# FLOW:
# 1) Load .env (API key)
# 2) Read PDF (PyMuPDF)
# 3) Chunk text
# 4) Create embeddings (SentenceTransformers)
# 5) Build FAISS index
# 6) Retrieve top-k chunks for a query
# 7) Send context + query to OpenAI (Generation)
# ==========================================================


# ==========================================================
# TOPIC: Load Environment Variables (.env) using python-dotenv
# YouTube search: "python dotenv load env variables"
# ==========================================================
from dotenv import load_dotenv
import os

load_dotenv()  # This loads .env from project root (where you run python)

# DEBUG (optional): Check first few chars of key
# If this prints empty, your .env isn't loading correctly.
print("KEY CHECK:", (os.getenv("OPENAI_API_KEY") or "")[:10])


# ==========================================================
# TOPIC: PDF Text Extraction using PyMuPDF (fitz)
# YouTube search: "PyMuPDF fitz extract text python"
# ==========================================================
import fitz  # PyMuPDF

pdf_path = "data/entire-vw-ar24.pdf"  # Your PDF path in data/ folder

doc = fitz.open(pdf_path)
full_text = ""

for page_number in range(len(doc)):
    page = doc[page_number]
    text = page.get_text("text")
    full_text += text + "\n"

doc.close()

print("\nPDF Preview (first 500 chars):\n")
print(full_text[:500])


# ==========================================================
# TOPIC: Text Chunking for RAG (Character-Based)
# YouTube search: "text chunking for RAG beginner"
# ==========================================================
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Splits text into overlapping chunks.
    chunk_size: how many characters in one chunk
    chunk_overlap: how many characters repeat between chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap

        if start < 0:
            start = 0

    return chunks


chunks = chunk_text(full_text, chunk_size=1000, chunk_overlap=200)

print("\nTotal Chunks:", len(chunks))
print("\nFirst Chunk Preview (first 400 chars):\n")
print(chunks[0][:400])


# ==========================================================
# TOPIC: Sentence Embeddings (SentenceTransformers)
# YouTube search: "sentence transformers embeddings python"
# ==========================================================
from sentence_transformers import SentenceTransformer
import numpy as np

print("\nLoading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")

print("\nCreating embeddings for chunks...")
chunk_embeddings = embed_model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings)

print("Embedding shape:", chunk_embeddings.shape)  # expected (num_chunks, 384)


# ==========================================================
# TOPIC: FAISS Vector Index (Similarity Search)
# YouTube search: "FAISS similarity search python"
# ==========================================================
import faiss

dimension = chunk_embeddings.shape[1]  # should be 384
index = faiss.IndexFlatL2(dimension)   # L2 distance index
index.add(chunk_embeddings)

print("\nFAISS index created!")
print("Total vectors in index:", index.ntotal)


# ==========================================================
# TOPIC: Retrieval (Top-K similar chunks)
# ==========================================================
query = "Electric vehicle sales performance in 2024"

query_embedding = embed_model.encode([query])
query_embedding = np.array(query_embedding)

k = 5
distances, indices = index.search(query_embedding, k)

print("\nTop matching chunks (retrieval results):\n")
for i, idx in enumerate(indices[0]):
    print(f"\nResult {i+1} (chunk index: {idx}):")
    print(chunks[idx][:500])
    print("----------")


# ==========================================================
# TOPIC: RAG Prompting (Grounded Answer Generation)
# YouTube search: "RAG prompt template reduce hallucination"
# ==========================================================
context = "\n\n".join([chunks[idx] for idx in indices[0]])

prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the CONTEXT below.
If the answer is not in the context, say: "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()


# ==========================================================
# TOPIC: Local LLM Generation using Ollama (Mistral)
# YouTube search: "ollama api generate python"
# ==========================================================

import requests
import json

# Build context from retrieved chunks
context = "\n\n".join([chunks[idx] for idx in indices[0]])

# Strict prompt to reduce hallucination
prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the CONTEXT below.
If the answer is not in the context, say: "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()

# Ollama local API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

payload = {
    "model": "mistral",      # local model name
    "prompt": prompt,
    "stream": False          # full response ek saath
}

print("\n================ FINAL RAG ANSWER (Ollama) ================\n")

response = requests.post(OLLAMA_URL, json=payload)
response.raise_for_status()

data = response.json()
print(data["response"])