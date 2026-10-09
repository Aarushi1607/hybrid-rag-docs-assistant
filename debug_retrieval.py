from src.retrieval import (
    load_retrieval_components,
    dense_retrieve,
    sparse_retrieve,
    reciprocal_rank_fusion,
)

def show_results(title, results):
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk_id = doc.metadata.get("chunk_index", "unknown")
        preview = doc.page_content[:250].replace("\n", " ")

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Score: {score:.6f}")
        print(f"Source: {source}")
        print(f"Text: {preview}")


if __name__ == "__main__":
    vectorstore, bm25, chunks = load_retrieval_components()

    query = "How do I define a POST endpoint in FastAPI?"

    dense_results = dense_retrieve(
        query, vectorstore, k=10
    )

    sparse_results = sparse_retrieve(
        query, bm25, chunks, k=10
    )

    hybrid_results = reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    dense_weight=0.5,
    sparse_weight=0.5,
    rrf_k=60,
    top_n=5,
    )

    show_results("DENSE RETRIEVAL", dense_results)
    show_results("SPARSE RETRIEVAL (BM25)", sparse_results)
    show_results("HYBRID RETRIEVAL (RRF)", hybrid_results)
    print("\n" + "=" * 80)
    print("FULL CONTENT OF CHUNK 3655")
    print("=" * 80)

    for chunk in chunks:
        if chunk.metadata.get("chunk_index") == 3655:
            print(chunk.page_content)
            break
    else:
        print("Chunk 3655 not found in chunks.pkl")

    
    print("\n" + "=" * 80)
    print("CHUNKS AROUND CHUNK 3655")
    print("=" * 80)

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_index")

        if chunk_id is not None and 3653 <= chunk_id <= 3657:
            print(f"\n--- CHUNK {chunk_id} ---")
            print(chunk.page_content)
