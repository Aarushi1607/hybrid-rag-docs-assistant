import os
import re
import pickle

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi

def load_documents(docs_path: str) -> list:
    """Load all .md files from a directory recursively."""
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.md",              
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    docs = loader.load()
    print(f"✓ Loaded {len(docs)} documents")
    print(f"  Total characters: {sum(len(d.page_content) for d in docs):,}")
    return docs

# corpus=collection of docs that the rag application will search
def inspect_corpus(docs: list):
    """Print stats about your loaded documents."""
    print("\n── Corpus Stats ──")
    for doc in docs[:5]:  # show first 5
        print(f"  File: {doc.metadata.get('source', 'unknown')}")
        print(f"  Length: {len(doc.page_content)} chars")
        print(f"  Preview: {doc.page_content[:100].strip()}")
        print()

def chunk_documents(docs: list, chunk_size: int = 512, chunk_overlap: int = 64) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",    # split on H2 headers first
            "\n### ",   # then H3
            "\n\n",     # then paragraphs
            "\n",       # then lines
            " ",        # then words
            ""          # then characters
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    print(f"✓ Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"  Avg chunk size: {sum(c.metadata['chunk_size'] for c in chunks) // len(chunks)} chars")
    print(f"  Smallest chunk: {min(c.metadata['chunk_size'] for c in chunks)} chars")
    print(f"  Largest chunk: {max(c.metadata['chunk_size'] for c in chunks)} chars")

    return chunks


def inspect_chunks(chunks: list, n: int = 3):
    """Show sample chunks."""
    print(f"\n── Sample Chunks ──")
    for chunk in chunks[:n]:
        print(f"  Source: {chunk.metadata.get('source', '?').split('/')[-1]}")
        print(f"  Size: {chunk.metadata['chunk_size']} chars")
        print(f"  Content: {chunk.page_content[:150].strip()}")
        print()

CHROMA_PATH = "./data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" #384 dimensional embeddings

def build_vector_store(chunks: list, persist_path: str = CHROMA_PATH) -> Chroma:
    """Embed chunks and store in ChromaDB."""

    # Check if already built
    if os.path.exists(persist_path) and os.listdir(persist_path):
        print("✓ ChromaDB already exists — loading from disk")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return Chroma(
            persist_directory=persist_path,
            embedding_function=embeddings
        )

    print(f"Building ChromaDB with {len(chunks)} chunks...")
    print("  (This takes a few minutes the first time)")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_path,
        collection_metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )

    print(f"✓ ChromaDB built and saved to {persist_path}")
    print(f"  Collection size: {vectorstore._collection.count()} vectors")
    return vectorstore

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def build_bm25_index(
    chunks: list,
    save_path: str = "./data/bm25_index.pkl"
):
    """Build and save a BM25 index for the supplied chunks."""
    print("Building BM25 index...")

    tokenized_corpus = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"✓ BM25 index built over {len(chunks)} chunks")

    return bm25


def save_chunks(
    chunks: list,
    save_path: str = "./data/chunks.pkl"
):
    """Save chunks for later retrieval."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✓ Chunks saved to {save_path}")


def load_chunks(
    save_path: str = "./data/chunks.pkl"
) -> list:
    """Load previously saved chunks."""
    with open(save_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    docs = load_documents("./docs/fastapi-docs/docs/en")

    chunks = chunk_documents(
        docs,
        chunk_size=512,
        chunk_overlap=64
    )

    save_chunks(chunks)

    vectorstore = build_vector_store(chunks)

    bm25 = build_bm25_index(chunks)

    print("\nIngestion complete. Ready to build retrieval.")
