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