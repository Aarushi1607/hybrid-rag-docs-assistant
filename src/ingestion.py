import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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



if __name__ == "__main__":
    docs = load_documents("./docs/fastapi-docs/docs/en")

    for size in [256, 512, 1024]:
        print(f"\n{'=' * 50}")
        print(f"CHUNK SIZE = {size}")

        chunks = chunk_documents(
            docs,
            chunk_size=size,
            chunk_overlap=size // 8
        )

        inspect_chunks(chunks, n=1)
