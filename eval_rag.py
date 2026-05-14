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


async def llm_grade(answer, case, contains_result=None):
    rag = load_test_rag()
    expected = case.get("expected") or {
        "must_contain": case.get("must_contain", []),
        "any_contain": case.get("any_contain", []),
        "not_contain": case.get("not_contain", []),
    }
    contains_note = ""
    if contains_result:
        contains_note = f"""
关键词初筛结果：
- passed: {contains_result["passed"]}
- reason: {contains_result["reason"]}
"""
    prompt = f"""
你是RAG评测员。请判断模型答案是否正确回答了问题。

问题：{case["question"]}
期望答案或评分规则：{expected}
模型答案：{answer}
{contains_note}

评分要求：
1. 重点判断语义是否正确，不要只做关键词匹配。
2. 如果答案用同义表达、近义表达或合理改写表达了同一含义，应判为通过。
3. 如果问题要求“文档没有就说明没有”，模型明确表示文档未提供、未提到、无法确定，且没有编造具体事实，应判为通过。
4. 如果答案虽然包含关键词，但核心事实错误、张冠李戴、编造文档未提供的信息，应判为不通过。
5. not_contain 表示高风险幻觉线索，但如果答案是在否定这些内容，请结合语义判断。

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


async def grade_answer(answer, case, judge_mode):
    contains_result = deterministic_grade(answer, case)
    if judge_mode == "contains":
        return contains_result
    if judge_mode == "llm":
        return await llm_grade(answer, case, contains_result=contains_result)
    if contains_result["passed"]:
        return {
            "passed": True,
            "method": "hybrid:contains",
            "reason": contains_result["reason"],
        }
    llm_result = await llm_grade(answer, case, contains_result=contains_result)
    return {
        "passed": llm_result["passed"],
        "method": "hybrid:llm",
        "reason": llm_result["reason"],
    }


async def run_case(case, judge_mode="hybrid"):
    rag = load_test_rag()
    question = case["question"]
    search_result = await rag.tree_search(question)
    node_list = search_result.get("node_list", []) if search_result else []
    context, context_node_ids = rag.retrieve_context(question, node_list)
    answer = await rag.answer_question(question, context)
    grade = await grade_answer(answer, case, judge_mode)

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
    parser.add_argument(
        "--judge",
        choices=["contains", "llm", "hybrid"],
        default="hybrid",
        help="Evaluation mode: contains=keyword matching, llm=LLM judge, hybrid=keywords first then LLM for failed cases",
    )
    parser.add_argument("--llm-judge", action="store_true", help="Deprecated alias for --judge llm")
    parser.add_argument("--quiet-rag", action="store_true", help="Hide test_rag debug/context preview")
    args = parser.parse_args()
    if args.llm_judge:
        args.judge = "llm"

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
        result = await run_case(case, judge_mode=args.judge)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} [{result['grade_method']}]: {result['grade_reason']}")
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
            "judge": args.judge,
        },
        "results": results,
    }

    output_path = args.output
    if not output_path:
        os.makedirs("results/evals", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results/evals", f"rag_eval_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== Eval Summary ===")
    print(f"Total: {summary['total']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Saved: {output_path}")

    failed_results = [item for item in results if not item["passed"]]
    if failed_results:
        print("\n=== Failed Cases ===")
        for index, item in enumerate(failed_results, 1):
            print(f"\n[{index}] {item['id']}")
            print(f"Question: {item['question']}")
            print(f"Reason: {item['grade_reason']}")
            print(f"Answer: {item['answer']}")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
