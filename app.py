"""
Streamlit chat UI for the personal knowledge base.

Retrieves relevant chunks from ChromaDB based on the user's question,
sends them as context to llama3.1:8b via Ollama, and displays the response.

Usage:  streamlit run app.py
"""

import streamlit as st
import chromadb
import ollama
from pathlib import Path

# -- Config --
DATA_DIR = Path(__file__).resolve().parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "personal_kb"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1:8b"
N_RESULTS = 5  # Number of relevant chunks to retrieve per question


@st.cache_resource
def get_collection():
    """Load the ChromaDB collection once and reuse across rerenders."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def retrieve(question: str, collection) -> list[str]:
    """
    RAG step 1 — RETRIEVAL
    Converts the question into a vector, then searches ChromaDB for the
    chunks with the most similar vectors. This is how we find the parts
    of your documents that are relevant to the question.
    """
    # Turn the question into a vector using the same model we used at ingest time
    response = ollama.embed(model=EMBED_MODEL, input=[question])

    # Vector similarity search — ChromaDB compares this vector against all
    # stored chunk vectors and returns the closest matches
    results = collection.query(
        query_embeddings=response.embeddings,
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"],
    )
    return results


def ask(question: str, context_chunks: list[dict]) -> str:
    """
    RAG step 2 — AUGMENTATION + GENERATION
    Takes the retrieved chunks and injects them into the prompt as context.
    The LLM then generates an answer grounded in your actual documents
    rather than relying solely on its training data.
    """
    # AUGMENTATION: combine the retrieved chunks into a single context block
    # that we'll inject into the prompt
    context = "\n\n---\n\n".join(context_chunks)

    # Build the augmented prompt — the question alone would just use the LLM's
    # training data, but by prepending the retrieved context we "augment" it
    # with knowledge from your personal documents
    prompt = (
        "Use the following context to answer the question. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    # GENERATION: the LLM reads the augmented prompt and generates an answer
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


# -- UI --
st.title("Personal Knowledge Base")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if question := st.chat_input("Ask your knowledge base a question"):
    # Show the user's message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # This is where the RAG pipeline runs:
    #   1. RETRIEVE — find relevant chunks from your documents
    #   2. AUGMENT  — inject those chunks into the prompt as context
    #   3. GENERATE — LLM answers using that context
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            collection = get_collection()
            results = retrieve(question, collection)          # R — Retrieval
            chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            answer = ask(question, chunks)                    # A+G — Augmentation + Generation
        st.markdown(answer)

        # Show the retrieved chunks so you can see what the retrieval found
        with st.expander("Retrieved chunks (click to inspect)"):
            for i, (chunk, meta, dist) in enumerate(zip(chunks, metadatas, distances)):
                st.markdown(f"**Chunk {i + 1}** — `{meta['source']}` (distance: {dist:.4f})")
                st.text(chunk[:500])
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
