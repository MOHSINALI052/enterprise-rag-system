# ==========================================================
# PROJECT: Enterprise-Level RAG System (Document AI) - CHATBOT
# ==========================================================
# FLOW:
# 1) Read PDF (PyMuPDF)
# 2) Chunk text
# 3) Create embeddings (SentenceTransformers)
# 4) Build FAISS index
# 5) Interactive chatbot:
#       user question -> FAISS retrieval -> context -> Ollama Mistral -> answer
# ==========================================================


# ==========================================================
# TOPIC: (Optional) Load Environment Variables (.env)
# YouTube search: "python dotenv load env variables"
# NOTE: Ollama ke liye API key zaroori nahi.
# ==========================================================
from dotenv import load_dotenv
load_dotenv()


# ==========================================================
# TOPIC: PDF Text Extraction using PyMuPDF (fitz)
# YouTube search: "PyMuPDF fitz extract text python"
# ==========================================================
import fitz  # PyMuPDF

pdf_path = "data/entire-vw-ar24.pdf"  # PDF path in data/ folder

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
# TOPIC: Interactive RAG Chatbot (FAISS + Ollama Mistral)
# YouTube search: "RAG chatbot python faiss ollama"
# ==========================================================
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

print("\n================ RAG CHATBOT READY (Ollama) ================\n")
print("Type 'exit' to quit.\n")


while True:
    # ----------------------------------------------------------
    # TOPIC: User Input in Python
    # YouTube search: "python input function"
    # ----------------------------------------------------------
    query = input("Ask a question (or type 'exit'): ").strip()

    if query.lower() == "exit":
        print("\nExiting chatbot... Bye!")
        break

    if not query:
        print("Please type a question.\n")
        continue

    # ----------------------------------------------------------
    # TOPIC: Convert query into embedding
    # YouTube search: "sentence transformer encode query"
    # ----------------------------------------------------------
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding)

    # ----------------------------------------------------------
    # TOPIC: FAISS Similarity Search (Top-K)
    # YouTube search: "faiss index search top k"
    # ----------------------------------------------------------
    k = 5
    distances, indices = index.search(query_embedding, k)

    # ----------------------------------------------------------
    # TOPIC: Build context from retrieved chunks
    # YouTube search: "RAG context building"
    # ----------------------------------------------------------
    context = "\n\n".join([chunks[idx] for idx in indices[0]])

    # OPTIONAL: context limit (speed + better answers)
    context = context[:4000]

    # ----------------------------------------------------------
    # TOPIC: RAG Prompt Template
    # YouTube search: "RAG prompt template reduce hallucination"
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # TOPIC: Call Ollama Local API (Mistral)
    # YouTube search: "ollama api generate python"
    # ----------------------------------------------------------
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        print("\n---------------- ANSWER ----------------\n")
        print(data.get("response", "").strip())
        print("\n----------------------------------------\n")

    except requests.exceptions.RequestException as e:
        print("\n❌ Ollama request failed. Check if Ollama is running.")
        print("Try:  ollama run mistral")
        print("Error:", e, "\n")