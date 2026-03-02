import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Enterprise RAG Chatbot (Ollama)", layout="wide")
OLLAMA_URL = "http://localhost:11434/api/generate"

# ----------------------------
# HELPERS
# ----------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF."""
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Character-based chunking with overlap."""
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
        if start < 0:
            start = 0

    return chunks


@st.cache_resource
def load_embedding_model():
    """Load embedding model once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index for L2 similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def ollama_generate(prompt: str, model_name: str = "mistral") -> str:
    """Call Ollama local API to generate response."""
    payload = {"model": model_name, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", "")


# ----------------------------
# UI
# ----------------------------
st.title("📄 Enterprise RAG Chatbot (FAISS + Ollama Mistral)")
st.caption("Upload a PDF → Ask questions → Get grounded answers (free, local).")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Ollama Model", ["mistral", "llama3", "llama2"], index=0)
    chunk_size = st.slider("Chunk size (chars)", 400, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 200, 50)
    top_k = st.slider("Top-K chunks", 1, 10, 5, 1)
    max_context_chars = st.slider("Max context chars", 1000, 12000, 4000, 500)

st.divider()

uploaded = st.file_uploader("Upload your PDF", type=["pdf"])

# Session state storage
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

embed_model = load_embedding_model()

if uploaded is not None:
    file_bytes = uploaded.read()

    # Rebuild only if new PDF or settings changed
    rebuild_needed = (
        st.session_state.pdf_name != uploaded.name
        or st.session_state.chunks is None
    )

    if rebuild_needed:
        with st.spinner("Extracting text from PDF..."):
            full_text = extract_text_from_pdf(file_bytes)

        if not full_text.strip():
            st.error("No text found in PDF (it may be scanned image). Try another PDF.")
            st.stop()

        with st.spinner("Chunking text..."):
            chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        with st.spinner("Creating embeddings (this may take a bit)..."):
            embeddings = embed_model.encode(chunks)
            embeddings = np.array(embeddings, dtype="float32")

        with st.spinner("Building FAISS index..."):
            index = build_faiss_index(embeddings)

        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings
        st.session_state.index = index
        st.session_state.pdf_name = uploaded.name

        st.success(f"✅ Ready! PDF: {uploaded.name} | Chunks: {len(chunks)}")

    # Ask question UI
    question = st.text_input("Ask a question about the PDF", placeholder="e.g., Electric vehicle sales performance in 2024")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("🔎 Ask", use_container_width=True)
    with col2:
        show_sources = st.toggle("Show sources", value=True)

    if ask and question.strip():
        # Retrieve
        with st.spinner("Retrieving relevant context..."):
            q_emb = embed_model.encode([question])
            q_emb = np.array(q_emb, dtype="float32")
            distances, indices = st.session_state.index.search(q_emb, top_k)

            picked = [st.session_state.chunks[i] for i in indices[0]]
            context = "\n\n".join(picked)[:max_context_chars]

        # Prompt
        prompt = f"""
You are a helpful assistant.
Answer the question ONLY using the CONTEXT below.
If the answer is not in the context, say: "I don't know based on the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

        # Generate
        try:
            with st.spinner("Generating answer (Ollama)..."):
                answer = ollama_generate(prompt, model_name=model_name)

            st.subheader("✅ Answer")
            st.write(answer)

            if show_sources:
                st.subheader("📌 Sources (Top Chunks)")
                for rank, chunk in enumerate(picked, start=1):
                    with st.expander(f"Chunk #{rank}"):
                        st.write(chunk)

        except requests.exceptions.RequestException as e:
            st.error("❌ Could not connect to Ollama. Make sure it's running.")
            st.code("ollama run mistral")
            st.write(str(e))

else:
    st.info("Upload a PDF to start. (Also ensure Ollama is running: `ollama run mistral`)")