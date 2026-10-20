"""Print the numbers you can honestly quote for the project.

Run from the project root (the data paths are relative to it):
    python -m src.project_stats
"""
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.getcwd())

OLLAMA_MODEL = "qwen3:4b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def section(title):
    print(f"\n=== {title} ===")


def model_sizes():
    section("Models (pretrained - not trained by you)")
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMBED_MODEL)
        n = sum(p.numel() for p in m.parameters())
        print(f"Embedding model : {EMBED_MODEL}")
        print(f"  parameters    : {n/1e6:.1f}M, vector size {m.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"Embedding model : could not load ({e})")
    try:
        import ollama
        info = ollama.show(OLLAMA_MODEL)
        d = info["details"] if isinstance(info, dict) else info.details
        get = (lambda k: d.get(k)) if isinstance(d, dict) else (lambda k: getattr(d, k, None))
        print(f"LLM             : {OLLAMA_MODEL}")
        print(f"  parameters    : {get('parameter_size')}, quantization {get('quantization_level')}, family {get('family')}")
    except Exception as e:
        print(f"LLM             : could not query Ollama ({e})")


def corpus_stats():
    section("Knowledge base (your data, not training data)")
    try:
        with open("./data/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        sources = {str(c.metadata.get("source")) for c in chunks}
        avg = sum(len(c.page_content) for c in chunks) / len(chunks)
        print(f"Chunks indexed  : {len(chunks)}")
        print(f"Source files    : {len(sources)}")
        print(f"Avg chunk size  : {avg:.0f} characters")
    except Exception as e:
        print(f"Could not read data/chunks.pkl ({e})")


def eval_summary():
    section("Measured results (from evaluation.py)")
    files = sorted(glob.glob("./data/eval_results*.json"))
    if not files:
        print("No data/eval_results*.json found. Run: python -m src.evaluation")
        return
    for path in files:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        scored = [r for r in rows if r.get("score") is not None]
        neg = [r for r in scored if r.get("golden_answer") == "NOT_IN_CORPUS"]
        pos = [r for r in scored if r.get("golden_answer") != "NOT_IN_CORPUS"]
        print(f"\n{os.path.basename(path)}  ({len(rows)} questions, {len(scored)} scored)")
        if pos:
            print(f"  Answer quality (LLM judge) : {sum(r['score'] for r in pos)/len(pos):.2f} / 5 over {len(pos)} in-corpus questions")
            print(f"  Scored 4-5                 : {sum(r['score'] >= 4 for r in pos)}/{len(pos)}")
        if neg:
            ok = sum(r["score"] == 5 for r in neg)
            print(f"  Correct refusals           : {ok}/{len(neg)} out-of-corpus questions")
        lat = [r["latency_s"] for r in rows if "latency_s" in r]
        if lat:
            print(f"  Avg latency                : {sum(lat)/len(lat):.1f}s")


if __name__ == "__main__":
    model_sizes()
    corpus_stats()
    eval_summary()