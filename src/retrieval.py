import re
import pickle

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi


CHROMA_PATH = "./data/chroma_db"
BM25_PATH = "./data/bm25_index.pkl"
CHUNKS_PATH = "./data/chunks.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_retrieval_components():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return vectorstore, bm25, chunks

def dense_retrieve(
    query: str,
    vectorstore: Chroma,
    k: int = 10
) -> list:
    """Retrieve the k most semantically similar chunks."""

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    return results

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"\b\w+\b", text.lower())

def sparse_retrieve(query: str,bm25: BM25Okapi,chunks: list,k: int = 10) -> list:
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    top_k_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [
        (chunks[i], scores[i])
        for i in top_k_indices
    ]


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
    top_n: int = 5
) -> list:

    fused_scores = {}
    documents = {}

    # Process dense retrieval results
    for rank, (doc, _) in enumerate(dense_results, start=1):
        chunk_id = doc.metadata["chunk_index"]

        if chunk_id not in fused_scores:
            fused_scores[chunk_id] = 0.0
            documents[chunk_id] = doc

        fused_scores[chunk_id] += dense_weight / (rrf_k + rank)

    # Process sparse retrieval results
    for rank, (doc, _) in enumerate(sparse_results, start=1):
        chunk_id = doc.metadata["chunk_index"]

        if chunk_id not in fused_scores:
            fused_scores[chunk_id] = 0.0
            documents[chunk_id] = doc

        fused_scores[chunk_id] += sparse_weight / (rrf_k + rank)

    # Sort chunks by their combined RRF scores
    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (documents[chunk_id], score)
        for chunk_id, score in ranked_chunks[:top_n]
    ]


if __name__ == "__main__":
    vectorstore, bm25, chunks = load_retrieval_components()

    query = "Path parameters predefined values"

    print("\n=== DENSE RESULTS (ChromaDB) ===")

    dense_results = dense_retrieve(
        query,
        vectorstore,
        k=5
    )

    for doc, score in dense_results:
        print(f"\n[{score:.3f}]")
        print(doc.page_content[:200].strip())

    print("\n=== SPARSE RESULTS (BM25) ===")

    sparse_results = sparse_retrieve(
        query,
        bm25,
        chunks,
        k=5
    )

    for doc, score in sparse_results:
        print(f"\n[{score:.3f}]")
        print(doc.page_content[:200].strip())
    
    print("\n=== HYBRID RESULTS (RRF) ===")

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        top_n=5
    )

    for doc, score in hybrid_results:
        print(f"\n[{score:.5f}]")
        print(doc.page_content[:200].strip())
