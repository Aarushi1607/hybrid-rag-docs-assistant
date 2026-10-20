"""Find out WHY answers score low, so you fix the right layer.

Run from the project root:
    python -m src.diagnose                        # reads the newest data/eval_results*.json
    python -m src.diagnose --show                 # also prints what was retrieved for each failure
    python -m src.diagnose data/eval_results_dense_gate.json --show --mode dense
"""
import argparse
import glob
import json
import os


def is_fallback(text: str) -> bool:
    return "don't have enough information" in text.lower().replace("\u2019", "'")


def classify(row: dict) -> str:
    ans = row.get("generated_answer", "")
    if is_fallback(ans):
        if "best retrieval score" in ans.lower():
            return "refused: retrieval confidence gate"
        return "refused: answerability check"
    return "answered, but judged wrong/incomplete"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--show", action="store_true", help="print retrieved chunks for failures")
    ap.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    ap.add_argument("--max", type=int, default=6, help="max failures to show with --show")
    args = ap.parse_args()

    path = args.path or max(glob.glob("./data/eval_results*.json"), key=os.path.getmtime)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    pos = [r for r in rows if r.get("score") is not None and r.get("golden_answer") != "NOT_IN_CORPUS"]
    fails = [r for r in pos if r["score"] <= 3]
    print(f"File: {path}")
    print(f"{len(pos)} in-corpus questions scored, {len(fails)} scored <= 3\n")

    print("Why the failures failed:")
    groups = {}
    for r in fails:
        groups.setdefault(classify(r), []).append(r)
    for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(v):3}  {k}")
        if k.startswith("refused"):
            tops = sorted(round(float(r.get("confidence_score", 0)), 2) for r in v)
            print(f"       top dense scores of these: {tops}")

    print("\nAverage score by confidence label:")
    for label in ("high", "medium", "low", "none", "?"):
        s = [r["score"] for r in pos if r.get("confidence_label") == label]
        if s:
            print(f"  {label:7} {sum(s)/len(s):.2f}  ({len(s)} questions)")

    print("\nAverage score by question type:")
    types = sorted({r["type"] for r in pos})
    for t in types:
        s = [r["score"] for r in pos if r["type"] == t]
        print(f"  {t:15} {sum(s)/len(s):.2f}  ({len(s)} questions)")

    if args.show and fails:
        from src.retrieval import load_retrieval_components, retrieve_context, source_name
        vs, bm25, chunks = load_retrieval_components()
        for r in fails[: args.max]:
            print(f"\n{'='*70}\nQ{r['id']} [{r['type']}] score {r['score']}: {r['question']}")
            print(f"GOLDEN   : {r['golden_answer'][:200]}")
            print(f"GENERATED: {r['generated_answer'][:200]}")
            ctx, _ = retrieve_context(r["question"], vs, bm25, chunks, mode=args.mode, top_n=5)
            for i, (doc, score) in enumerate(ctx, 1):
                text = doc.page_content[:220].replace("\n", " ").strip()
                print(f"  [{i}] {source_name(doc)} ({score:.4f}): {text}")
        print("\nFor each failure ask: is the golden answer's information in the chunks above?")
        print("  no  -> retrieval/chunking problem;  yes -> generation, prompt or judge problem.")


if __name__ == "__main__":
    main()