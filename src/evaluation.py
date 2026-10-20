import argparse
import json
import os
import sys
import time

import ollama

import src.generation as generation
from src.generation import ask, warm_up, OLLAMA_MODEL, KEEP_ALIVE
from src.retrieval import load_retrieval_components


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

JUDGE_OPTIONS = {"temperature": 0, "num_predict": 150}


def is_fallback(answer: str) -> bool:
    """True if the answer is the 'not enough information' refusal."""
    return "don't have enough information" in answer.lower().replace("\u2019", "'")


def llm_judge(question: str, golden_answer: str, generated_answer: str) -> dict:
    """Score answer correctness 1-5 based on meaning, not exact wording.

    Returns score=None (not 0) when the judge itself fails, so judge errors
    are reported separately instead of dragging the average down.
    """

    judge_prompt = f"""You are an objective evaluator assessing a technical documentation assistant.

Question: {question}
Golden Answer: {golden_answer}
Generated Answer: {generated_answer}

Score the generated answer 1-5 based on MEANING, not exact wording:
5 - Correct and complete. Covers the same key points as the golden answer.
4 - Mostly correct. Covers the main point with minor gaps or extra detail.
3 - Partially correct. Gets some key points but misses important ones.
2 - Mostly wrong. Some relevant content but fundamentally incorrect.
1 - Wrong or hallucinated. Contradicts the golden answer or makes things up.

IMPORTANT: A correct answer phrased differently from the golden answer should still score 4-5.
If the generated answer says "I don't have enough information" but the golden answer exists, score 1.

Respond with only: {{"score": X, "reason": "one sentence"}}"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            format="json",
            options=JUDGE_OPTIONS,
            keep_alive=KEEP_ALIVE,
        )
        result = json.loads(response["message"]["content"])
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return {"score": None, "reason": "invalid score format"}
        return {
            "score": max(1, min(5, int(score))),
            "reason": str(result.get("reason", "")),
        }
    except Exception as e:
        return {"score": None, "reason": f"judge error: {e}"}


def run_evaluation(
    golden_path: str = "./data/golden_qa.json",
    mode: str = "hybrid",
    use_gate: bool = generation.USE_ANSWERABILITY_CHECK,
):
    """Run the full eval suite and print a detailed report."""

    generation.USE_ANSWERABILITY_CHECK = use_gate

    vectorstore, bm25, chunks = load_retrieval_components()
    try:
        warm_up()
    except Exception as e:
        print(f"⚠ Ollama warm-up failed: {e}")

    with open(golden_path, encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []

    for item in golden_set:
        print(f"Evaluating Q{item['id']}: {item['question'][:60]}...")

        start = time.perf_counter()
        try:
            result = ask(item["question"], vectorstore, bm25, chunks, mode=mode)
        except Exception as e:
            print(f"  ✗ ask() failed: {e}")
            results.append({
                "id": item["id"], "type": item["type"], "question": item["question"],
                "mode": mode, "score": None, "reason": f"ask error: {e}",
                "latency_s": round(time.perf_counter() - start, 2),
                "confidence_label": "?", "confidence_score": 0,
                "generated_answer": "", "golden_answer": item["golden_answer"][:300],
            })
            continue
        latency = time.perf_counter() - start

        if item["golden_answer"] == "NOT_IN_CORPUS":
            triggered = is_fallback(result["answer"])
            score_data = {
                "score": 5 if triggered else 1,
                "reason": "Correctly said I don't know" if triggered
                          else "Should have said I don't know but gave an answer",
            }
        else:
            score_data = llm_judge(
                item["question"],
                item["golden_answer"],
                result["answer"],
            )

        # Print low scores / judge failures immediately for debugging
        if score_data["score"] is None:
            print(f"  ✗ Not scored — {score_data['reason']}")
        elif score_data["score"] <= 2:
            print(f"  ⚠  Score {score_data['score']}/5 — {score_data['reason']}")
            print(f"     Generated: {result['answer'][:120]}")
            print(f"     Golden:    {item['golden_answer'][:120]}")

        results.append({
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "mode": result.get("retrieval_mode", mode),
            "score": score_data["score"],
            "reason": score_data["reason"],
            "latency_s": round(latency, 2),
            "confidence_label": result.get("confidence", {}).get("label", "?"),
            "confidence_score": result.get("confidence", {}).get("top_score", 0),
            "generated_answer": result["answer"][:300],
            "golden_answer": item["golden_answer"][:300],
        })

    scored = [r for r in results if r["score"] is not None]
    unscored = [r for r in results if r["score"] is None]

    print(f"\n{'='*50}")
    print("EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"Config          : mode={mode}, answerability_check={use_gate}, "
          f"model={OLLAMA_MODEL}, top_n={generation.CONTEXT_TOP_N}")
    print(f"Total questions : {len(results)} ({len(scored)} scored, {len(unscored)} not scored)")
    if scored:
        avg_score = sum(r["score"] for r in scored) / len(scored)
        print(f"Average score   : {avg_score:.2f} / 5")
    if results:
        avg_latency = sum(r["latency_s"] for r in results) / len(results)
        print(f"Average latency : {avg_latency:.1f}s per question")

    by_type = {}
    for r in scored:
        by_type.setdefault(r["type"], []).append(r["score"])

    print("\nBy question type:")
    for qtype, scores in by_type.items():
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {qtype:15} {bar} {avg:.2f} ({len(scores)} questions)")

    print("\nLow-scoring questions (≤ 2):")
    low = [r for r in scored if r["score"] <= 2]
    if low:
        for r in low:
            print(f"  Q{r['id']} [{r['type']}]: {r['question'][:60]}")
            print(f"    Score: {r['score']}/5 — {r['reason']}")
    else:
        print("  None!")

    if unscored:
        print("\nNot scored (errors):")
        for r in unscored:
            print(f"  Q{r['id']}: {r['reason']}")

    # One file per configuration so hybrid/dense and gate on/off can be compared.
    out_path = f"./data/eval_results_{mode}_{'gate' if use_gate else 'nogate'}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path.removeprefix('./')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set.")
    parser.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid")
    parser.add_argument("--no-gate", action="store_true",
                        help="skip the answerability LLM call (single-call generation)")
    parser.add_argument("--golden", default="./data/golden_qa.json")
    args = parser.parse_args()

    run_evaluation(args.golden, mode=args.mode, use_gate=args.gate or generation.USE_ANSWERABILITY_CHECK)