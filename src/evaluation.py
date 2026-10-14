import json
import ollama
from src.generation import ask
from src.retrieval import load_retrieval_components


def llm_judge(question: str, golden_answer: str, generated_answer: str) -> dict:
    """Score answer correctness 1-5 based on meaning, not exact wording."""

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

    response = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": judge_prompt}],
        format="json",
        options={"think": False}
    )

    try:
        result = json.loads(response['message']['content'])
        # Validate score is in range
        if not isinstance(result.get('score'), (int, float)):
            return {"score": 0, "reason": "invalid score format"}
        result['score'] = max(1, min(5, int(result['score'])))
        return result
    except Exception as e:
        return {"score": 0, "reason": f"parse error: {e}"}


def run_evaluation(golden_path: str = "./data/golden_qa.json"):
    """Run the full eval suite and print a detailed report."""

    vectorstore, bm25, chunks = load_retrieval_components()

    with open(golden_path) as f:
        golden_set = json.load(f)

    results = []

    for item in golden_set:
        print(f"Evaluating Q{item['id']}: {item['question'][:60]}...")

        result = ask(item['question'], vectorstore, bm25, chunks)

        if item['golden_answer'] == "NOT_IN_CORPUS":
            triggered = "don't have enough information" in result['answer'].lower()
            score_data = {
                "score": 5 if triggered else 1,
                "reason": "Correctly said I don't know" if triggered
                          else "Should have said I don't know but gave an answer"
            }
        else:
            score_data = llm_judge(
                item['question'],
                item['golden_answer'],
                result['answer']
            )

        # Print low scores immediately for debugging
        if score_data['score'] <= 2:
            print(f"  ⚠  Score {score_data['score']}/5 — {score_data['reason']}")
            print(f"     Generated: {result['answer'][:120]}")
            print(f"     Golden:    {item['golden_answer'][:120]}")

        results.append({
            "id": item['id'],
            "type": item['type'],
            "question": item['question'],
            "score": score_data['score'],
            "reason": score_data['reason'],
            "confidence_label": result.get('confidence', {}).get('label', '?'),
            "confidence_score": result.get('confidence', {}).get('top_score', 0),
            "generated_answer": result['answer'][:300],
            "golden_answer": item['golden_answer'][:300]
        })

    # ── Report ────────────────────────────────────────────────
    avg_score = sum(r['score'] for r in results) / len(results)

    by_type = {}
    for r in results:
        by_type.setdefault(r['type'], []).append(r['score'])

    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"Total questions : {len(results)}")
    print(f"Average score   : {avg_score:.2f} / 5")

    print(f"\nBy question type:")
    for qtype, scores in by_type.items():
        avg = sum(scores) / len(scores)
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {qtype:15} {bar} {avg:.2f} ({len(scores)} questions)")

    print(f"\nLow-scoring questions (≤ 2):")
    low = [r for r in results if r['score'] <= 2]
    if low:
        for r in low:
            print(f"  Q{r['id']} [{r['type']}]: {r['question'][:60]}")
            print(f"    Score: {r['score']}/5 — {r['reason']}")
    else:
        print("  None!")

    with open("./data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to data/eval_results.json")


if __name__ == "__main__":
    run_evaluation()
