# YOLOv5 PageIndex RAG Evaluation Notes

## 1. 测试目标

验证基于 PageIndex 结构化 JSON 的 RAG 问答效果，并测试自动化评测脚本的稳定性。

## 2. 测试对象

- 原始文档：Yolov5.pdf
- PageIndex 输出：results/Yolov5_structure.json
- 测试用例：eval_cases.example.json
- 评测脚本：eval_rag.py
- 问答脚本：test_rag.py

## 3. 当前 RAG 流程

```text
用户问题
 ↓
tree_search(question)
 ↓
retrieve_context(question, node_list)
 ↓
answer_question(question, context)
 ↓
grade_answer(answer, case)

## 4. 评测方式
hybrid，关键词初筛 + LLM 复判

## 5. 结果：
31/32，96.88%

## 6. 唯一失败：
dataset_source

## 7. 失败原因：
数字汇总错误，模型把 2376 误写成总数，没有明确给出 4171