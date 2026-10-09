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
    print(f"✓ ChromaDB loaded ({vectorstore._collection.count()} vectors)")
    print(f"✓ BM25 + {len(chunks)} chunks loaded")

    return vectorstore, bm25, chunks


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"\b\w+\b", text.lower())

def dense_retrieve(query: str, vectorstore, k: int = 10,
                   min_score: float = 0.3) -> list:
    """
    Returns (doc, relevance_score) where HIGHER score = MORE relevant.
    Range: 0.0 to 1.0
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    results.sort(key=lambda x: x[1], reverse=True)

    # Filter out low-relevance chunks
    filtered = [(doc, score) for doc, score in results if score >= min_score]

    if not filtered:
        # Fallback: return top-3 even if below threshold
        # (better than passing nothing to the LLM)
        print(f"  ⚠ No chunks above threshold {min_score} — returning top-3 anyway")
        return results[:3]

    return filtered


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

def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    chunks: list,
    top_n: int = 5
) -> list:
    """
    Full hybrid pipeline: dense + sparse + RRF.
    This is what generation.py calls.
    Returns top_n (doc, score) tuples.
    """
    dense_results  = dense_retrieve(query, vectorstore, k=10)
    sparse_results = sparse_retrieve(query, bm25, chunks, k=10)

    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        top_n=top_n
    )

def compare_retrieval(
    query: str,
    vectorstore: Chroma,
    bm25: BM25Okapi,
    chunks: list,
    top_n: int = 5
) -> dict:
    """Compare dense-only retrieval with hybrid RRF retrieval."""

    dense_results = dense_retrieve(
        query, vectorstore, k=10
    )

    sparse_results = sparse_retrieve(
        query, bm25, chunks, k=10
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=0.7,
        sparse_weight=0.3,
        rrf_k=60,
        top_n=top_n
    )

    return {
        "dense": dense_results[:top_n],
        "hybrid": hybrid_results
    }



def evaluate_retrieval(
    results: list,
    relevant_chunk_ids: set,
    k: int = 5
) -> dict:
    """Calculate Precision@K and Recall@K using labeled chunk IDs."""
    retrieved_ids = [doc.metadata["chunk_index"] for doc, _ in results[:k]]

    relevant_retrieved = sum(
        cid in relevant_chunk_ids for cid in retrieved_ids
    )

    precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0.0
    recall    = relevant_retrieved / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0

    return {"precision_at_k": precision, "recall_at_k": recall}


def print_results(title: str, results: list, max_preview: int = 120):
    """Pretty-print retrieval results for debugging."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    for rank, (doc, score) in enumerate(results, 1):
        source   = doc.metadata.get("source", "?").split("\\")[-1]
        chunk_id = doc.metadata.get("chunk_index", "?")
        preview  = doc.page_content[:max_preview].replace("\n", " ").strip()
        print(f"\n  Rank {rank} | chunk_id={chunk_id} | score={score:.4f}")
        print(f"  Source: {source}")
        print(f"  Text:   {preview}...")


def find_relevant_chunks(query: str, vectorstore, bm25, chunks, top_n: int = 10):
    """
    Helper for building evaluation labels.
    Run this for each eval query, read the output, and manually
    mark which chunk_ids are truly relevant → put those in evaluation_queries.
    """
    results = hybrid_retrieve(query, vectorstore, bm25, chunks, top_n=top_n)

    print(f"\nQuery: '{query}'")
    print("Mark relevant chunks manually:\n")

    for rank, (doc, score) in enumerate(results, 1):
        chunk_id = doc.metadata.get("chunk_index")
        source   = doc.metadata.get("source", "?").split("\\")[-1]
        print(f"  [{rank}] chunk_id={chunk_id} | {source}")
        print(f"       {doc.page_content[:150].strip()}")
        print()

if __name__ == "__main__":
    vectorstore, bm25, chunks = load_retrieval_components()

    # find_relevant_chunks("Path parameters predefined values",
    #                       vectorstore, bm25, chunks)
    # find_relevant_chunks("How do query parameters work in FastAPI?",
    #                       vectorstore, bm25, chunks)
   
    evaluation_queries = [
        {
            "query": "Path parameters predefined values",
            "relevant_chunk_ids": {1631, 1603},
        },
        {
            "query": "How do query parameters work in FastAPI?",
            "relevant_chunk_ids": {1190, 1703, 1649, 1366},
        },
    ]

    for item in evaluation_queries:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]

        results = compare_retrieval(
            query,
            vectorstore,
            bm25,
            chunks,
            top_n=5,
        )

        print(f"\nQuery: {query}")

        for mode in ("dense", "hybrid"):
            metrics = evaluate_retrieval(
                results[mode],
                relevant_ids,
                k=5,
            )

            print(
                f"{mode.upper()}: "
                f"Precision@5={metrics['precision_at_k']:.2f}, "
                f"Recall@5={metrics['recall_at_k']:.2f}"
            )
        print_results("HYBRID results", results["hybrid"])
