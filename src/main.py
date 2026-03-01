# ==========================================================
# TOPIC: PDF Text Extraction using PyMuPDF (fitz)
# YouTube search: "PyMuPDF fitz extract text python"
# ==========================================================

# PyMuPDF ka import name "fitz" hota hai
import fitz

# ----------------------------------------------------------
# TOPIC: File Paths in Python
# YouTube search: "python relative path explained"
# ----------------------------------------------------------
# Tumhari PDF "data" folder mein hai:
# data/entire-vw-ar24.pdf
pdf_path = "data/entire-vw-ar24.pdf"

# ----------------------------------------------------------
# TOPIC: Opening a PDF file
# ----------------------------------------------------------
doc = fitz.open(pdf_path)  # PDF open

# ----------------------------------------------------------
# TOPIC: Looping in Python (for loop)
# YouTube search: "python for loop beginner"
# ----------------------------------------------------------
full_text = ""  # yahan hum saara text jama karenge

for page_number in range(len(doc)):        # total pages ka loop
    page = doc[page_number]               # specific page
    text = page.get_text("text")          # page ka text nikalo
    full_text += text + "\n"              # text add karo

doc.close()  # PDF close

# ----------------------------------------------------------
# TOPIC: String slicing in Python
# YouTube search: "python string slicing"
# ----------------------------------------------------------
print(full_text[:500])  # first 500 characters print

# ==========================================================
# TOPIC: Text Chunking for RAG (Character-Based)
# YouTube Search: "text chunking for RAG beginner"
# ==========================================================

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    text: full document text
    chunk_size: size of each chunk (characters)
    chunk_overlap: overlapping characters between chunks
    """

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - chunk_overlap

        if start < 0:
            start = 0

    return chunks


# ==========================================================
# CREATE CHUNKS
# ==========================================================

chunks = chunk_text(full_text, chunk_size=1000, chunk_overlap=200)

print("\nTotal Chunks:", len(chunks))
print("\nFirst Chunk Preview:\n", chunks[0][:400])
# ==========================================================
# TOPIC: Sentence Embeddings
# ==========================================================

from sentence_transformers import SentenceTransformer
import numpy as np

print("\nLoading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded!")

print("\nCreating embeddings...")
chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings)

print("Embedding shape:", chunk_embeddings.shape)


# ==========================================================
# TOPIC: FAISS Vector Index
# ==========================================================

import faiss

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("FAISS index created!")
print("Total vectors in index:", index.ntotal)


# ==========================================================
# TOPIC: Similarity Search Test
# ==========================================================

query = "Electric vehicle sales performance in 2024"

query_embedding = model.encode([query])
query_embedding = np.array(query_embedding)

k = 5
distances, indices = index.search(query_embedding, k)

print("\nTop matching chunks:\n")

for i, idx in enumerate(indices[0]):
    print(f"\nResult {i+1}:")
    print(chunks[idx][:500])
    print("----------")
# TOPIC: RAG Prompting (Grounded Answers)
# YouTube search: "RAG prompt template"
# ==========================================================

# Combine top chunks into one context string
context = "\n\n".join([chunks[idx] for idx in indices[0]])

# Make a strict prompt to reduce hallucination
prompt = f"""
You are a helpful assistant. Answer the question ONLY using the context below.
If the answer is not in the context, say: "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
""".strip()

# ==========================================================
# TOPIC: Calling OpenAI API
# YouTube search: "OpenAI chat completions python"
# ==========================================================

from openai import OpenAI
import os

# Load API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print("\n===== FINAL ANSWER (LLM) =====\n")
print(response.choices[0].message.content)