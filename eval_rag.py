import argparse
import asyncio
import json
import os
from datetime import datetime

test_rag = None


def load_test_rag():
    global test_rag
    if test_rag is None:
        import test_rag as loaded_test_rag
        test_rag = loaded_test_rag
    return test_rag


def normalize_text(text):
    return "".join(str(text).lower().split())


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def deterministic_grade(answer, case):
    answer_norm = normalize_text(answer)
    must_contain = [normalize_text(item) for item in as_list(case.get("must_contain"))]
    any_contain = [normalize_text(item) for item in as_list(case.get("any_contain"))]
    not_contain = [normalize_text(item) for item in as_list(case.get("not_contain"))]

    missing = [item for item in must_contain if item not in answer_norm]
    any_hit = not any_contain or any(item in answer_norm for item in any_contain)
    forbidden_hit = [item for item in not_contain if item in answer_norm]

    passed = not missing and any_hit and not forbidden_hit
    reasons = []
    if missing:
        reasons.append(f"missing must_contain: {missing}")
    if not any_hit:
        reasons.append(f"none of any_contain matched: {any_contain}")
    if forbidden_hit:
        reasons.append(f"matched not_contain: {forbidden_hit}")

    return {
        "passed": passed,
        "method": "contains",
        "reason": "; ".join(reasons) if reasons else "matched expected keywords",
    }


async def llm_grade(answer, case):
    rag = load_test_rag()
    expected = case.get("expected") or {
        "must_contain": case.get("must_contain", []),
        "any_contain": case.get("any_contain", []),
        "not_contain": case.get("not_contain", []),
    }
    prompt = f"""
你是RAG评测员。请判断模型答案是否正确回答了问题。

问题：{case["question"]}
期望答案或评分规则：{expected}
模型答案：{answer}

请只返回JSON，不要输出其他内容：
{{
  "passed": true或false,
  "reason": "简短说明"
}}
"""
    raw = await rag.call_llm(prompt)
    parsed = rag.parse_llm_json(raw)
    if not parsed:
        return {
            "passed": False,
            "method": "llm",
            "reason": f"LLM judge returned invalid JSON: {raw}",
        }
    return {
        "passed": bool(parsed.get("passed")),
        "method": "llm",
        "reason": parsed.get("reason", ""),
    }


async def run_case(case, use_llm_judge=False):
    rag = load_test_rag()
    question = case["question"]
    search_result = await rag.tree_search(question)
    node_list = search_result.get("node_list", []) if search_result else []
    context, context_node_ids = rag.retrieve_context(question, node_list)
    answer = await rag.answer_question(question, context)
    grade = await llm_grade(answer, case) if use_llm_judge else deterministic_grade(answer, case)

    return {
        "id": case.get("id", question),
        "question": question,
        "passed": grade["passed"],
        "grade_method": grade["method"],
        "grade_reason": grade["reason"],
        "node_list": node_list,
        "context_node_list": context_node_ids,
        "answer": answer,
    }


async def main():
    parser = argparse.ArgumentParser(description="Batch evaluate test_rag.py RAG answers.")
    parser.add_argument("--cases", default="eval_cases.example.json", help="Path to eval cases JSON")
    parser.add_argument("--json-path", default=None, help="Override test_rag config JSON_PATH")
    parser.add_argument("--output", default=None, help="Path to save eval result JSON")
    parser.add_argument("--llm-judge", action="store_true", help="Use LLM as judge instead of keyword matching")
    parser.add_argument("--quiet-rag", action="store_true", help="Hide test_rag debug/context preview")
    args = parser.parse_args()

    rag = load_test_rag()

    if args.json_path:
        rag.config["JSON_PATH"] = args.json_path
    if args.quiet_rag:
        rag.config["SHOW_DEBUG_INFO"] = False
        rag.config["SHOW_CONTEXT_PREVIEW"] = False

    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if not await rag.load_document():
        raise SystemExit(1)

    results = []
    for index, case in enumerate(cases, 1):
        print(f"\n[{index}/{len(cases)}] {case['question']}")
        result = await run_case(case, use_llm_judge=args.llm_judge)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}: {result['grade_reason']}")
        print(f"Answer: {result['answer']}")

    passed_count = sum(1 for item in results if item["passed"])
    summary = {
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "accuracy": passed_count / len(results) if results else 0,
    }

    payload = {
        "summary": summary,
        "config": {
            "json_path": rag.config["JSON_PATH"],
            "cases": args.cases,
            "judge": "llm" if args.llm_judge else "contains",
        },
        "results": results,
    }

    output_path = args.output
    if not output_path:
        os.makedirs("results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"rag_eval_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Eval Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
