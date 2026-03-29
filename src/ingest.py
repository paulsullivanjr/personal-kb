"""
Ingestion pipeline for the personal knowledge base.

Reads documents from data/raw/, splits them into chunks,
generates embeddings via Ollama, and stores everything in ChromaDB.

Usage:  python -m src.ingest
"""

import os
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -- Paths --
# Resolve relative to this file so it works regardless of where you run from
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"       # Drop your PDFs and .md files here
CHROMA_DIR = DATA_DIR / "chroma"  # ChromaDB stores its data here on disk

# -- Config --
COLLECTION_NAME = "personal_kb"   # Name of the ChromaDB collection
EMBED_MODEL = "nomic-embed-text"  # Ollama embedding model
CHUNK_SIZE = 1000                 # Max characters per chunk
CHUNK_OVERLAP = 200               # Overlap between chunks to preserve context at boundaries


def clean_text(text: str) -> str:
    """Remove surrogate and other invalid UTF-8 characters that PDFs sometimes contain."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def read_pdf(path: Path) -> str:
    """Extract all text from a PDF, joining pages with newlines."""
    reader = PdfReader(path)
    return clean_text("\n".join(page.extract_text() or "" for page in reader.pages))


def read_markdown(path: Path) -> str:
    """Read a markdown file as plain text."""
    return path.read_text(encoding="utf-8")


def load_documents() -> list[dict]:
    """
    Scan data/raw/ for supported files (.pdf, .md).
    Returns a list of dicts with 'source' (filename) and 'text' (content).
    Skips files that are empty or unsupported types.
    """
    docs = []
    for file in RAW_DIR.iterdir():
        if file.suffix == ".pdf":
            text = read_pdf(file)
        elif file.suffix == ".md":
            text = read_markdown(file)
        else:
            continue
        if text.strip():
            docs.append({"source": file.name, "text": text})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Split each document's text into smaller chunks for embedding.
    Each chunk gets a unique ID like "myfile.pdf::chunk0" so we can
    trace it back to its source document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "id": f"{doc['source']}::chunk{i}",
                "text": chunk,
                "source": doc["source"],
            })
    return chunks


def embed_and_store(chunks: list[dict]) -> None:
    """
    Generate vector embeddings for each chunk and store them in ChromaDB.

    - Uses PersistentClient so the DB is saved to disk at data/chroma/
    - Deletes the existing collection first so re-running gives a clean slate
    - Processes in batches of 50 to avoid sending too much to Ollama at once
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Wipe the old collection so re-ingesting replaces everything
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection didn't exist yet — that's fine
    collection = client.get_or_create_collection(COLLECTION_NAME)

    batch_size = 50
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]

        # Call Ollama to turn each text chunk into a vector
        response = ollama.embed(model=EMBED_MODEL, input=texts)
        embeddings = response.embeddings

        # Store the chunk text, its embedding, and metadata in ChromaDB
        collection.add(
            ids=[c["id"] for c in batch],
            documents=texts,
            metadatas=[{"source": c["source"]} for c in batch],
            embeddings=embeddings,
        )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")


def main():
    """Entry point: load -> chunk -> embed -> store."""
    print(f"Loading documents from {RAW_DIR} ...")
    docs = load_documents()
    if not docs:
        print("No PDF or Markdown files found in data/raw/. Nothing to ingest.")
        return

    print(f"Found {len(docs)} document(s). Chunking ...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunk(s). Embedding and storing ...")
    embed_and_store(chunks)
    print("Done.")


if __name__ == "__main__":
    main()
