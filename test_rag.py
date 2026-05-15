
import json
import os
import asyncio
import sys
import re
import unicodedata
from openai import AsyncOpenAI
from dotenv import load_dotenv


# 用户问题
#   ↓
# classify_query_type 本地判断问题类型
#   ↓
# tree_search
#   ├─ 元信息/数字问题：本地检索
#   └─ 普通问题：LLM 选 node_id
#   ↓
# retrieve_context 本地提取上下文
#   ↓
# answer_question
#   ├─ 特定作者/数值问题：本地直接回答
#   └─ 普通问题：LLM 基于上下文生成答案


# 加载环境变量
load_dotenv()

# 全局配置
config = {
    "JSON_PATH": "./results/indexes/Yolov5_structure.json",
    "MODEL_NAME": "deepseek-chat",
    "MAX_NODES": 3,
    "METADATA_NODE_COUNT": 2,
    "EXPAND_CONTEXT": True,
    "EXPAND_NEIGHBORS": 1,
    "EXPAND_PARENT": True,
    "TEMPERATURE": 0,
    "SHOW_CONTEXT_PREVIEW": True,
    "CONTEXT_PREVIEW_LENGTH": 800,
    "SHOW_DEBUG_INFO": True
}

# 初始化LLM客户端
client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 全局变量
global_doc = None  # 现在存储的是整个文档对象
global_node_map = None  # 预先生成的node_id到节点的映射
global_parent_map = {}
global_node_order = []

# ====================== Unicode清理函数 ======================
def clean_unicode(text):
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'[\ud800-\udfff]', '�', text)
    
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except:
        return ''.join([c for c in text if (32 <= ord(c) <= 126) or (0x4e00 <= ord(c) <= 0x9fff)])

def clean_node(node):
    """递归清理单个节点中的所有字符串"""
    for key, value in node.items():
        if isinstance(value, str):
            node[key] = clean_unicode(value)
        elif isinstance(value, dict):
            node[key] = clean_node(value)
        elif isinstance(value, list):
            for i in range(len(value)):
                if isinstance(value[i], str):
                    value[i] = clean_unicode(value[i])
                elif isinstance(value[i], dict):
                    value[i] = clean_node(value[i])
    return node
# ==============================================================

# ====================== 工具函数（完全重写，匹配你的JSON结构）======================
def count_nodes(nodes):
    """统计节点数组中的总节点数"""
    count = 0
    for node in nodes:
        count += 1
        if "nodes" in node and node["nodes"]:  # 注意：你的子节点字段是"nodes"，不是"children"！
            count += count_nodes(node["nodes"])
    return count

def count_nodes_with_text(nodes):
    """统计有text字段的节点数量"""
    count = 0
    for node in nodes:
        if "text" in node and node["text"].strip():
            count += 1
        if "nodes" in node and node["nodes"]:
            count += count_nodes_with_text(node["nodes"])
    return count

def create_node_mapping(nodes, node_map=None):
    """创建node_id到节点的映射字典"""
    if node_map is None:
        node_map = {}
    
    for node in nodes:
        if "node_id" in node:
            node_map[node["node_id"]] = node
        
        if "nodes" in node and node["nodes"]:
            create_node_mapping(node["nodes"], node_map)
    
    return node_map

def build_node_indexes(nodes):
    """构建节点映射、父节点映射和文档顺序列表"""
    node_map = {}
    parent_map = {}
    node_order = []

    def walk(items, parent_id=None):
        for node in items:
            node_id = node.get("node_id")
            if node_id:
                node_map[node_id] = node
                parent_map[node_id] = parent_id
                node_order.append(node_id)
            if node.get("nodes"):
                walk(node["nodes"], node_id)

    walk(nodes)
    return node_map, parent_map, node_order

def remove_fields_from_nodes(nodes, fields):
    """递归移除节点数组中指定的字段"""
    for node in nodes:
        for field in fields:
            if field in node:
                del node[field]
        
        if "nodes" in node and node["nodes"]:
            remove_fields_from_nodes(node["nodes"], fields)
    
    return nodes

def get_node_content(node):
    """获取节点及其所有子节点的完整文本内容"""
    content_parts = []
    
    # 先添加当前节点的text
    if "text" in node and node["text"].strip():
        content_parts.append(node["text"].strip())
    
    # 递归添加所有子节点的内容（注意：子节点字段是"nodes"）
    if "nodes" in node and node["nodes"]:
        for child in node["nodes"]:
            child_content = get_node_content(child)
            if child_content.strip():
                content_parts.append(child_content)
    
    return "\n\n".join(content_parts)

def iter_nodes(nodes):
    """扁平遍历所有节点"""
    for node in nodes:
        yield node
        if "nodes" in node and node["nodes"]:
            yield from iter_nodes(node["nodes"])

def parse_llm_json(text):
    """兼容 ```json ... ```、前后解释文字等常见LLM输出格式"""
    text = clean_unicode(text).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

def normalize_node_id(raw_id):
    """把 1、'1'、'node_id_1'、'0001' 等形式统一成真实node_id"""
    if raw_id is None:
        return None

    raw_id = str(raw_id).strip()
    if raw_id in global_node_map:
        return raw_id

    number_match = re.search(r"\d+", raw_id)
    if number_match:
        padded = number_match.group(0).zfill(4)
        if padded in global_node_map:
            return padded

    return None

def normalize_node_list(raw_node_list):
    """过滤不存在的节点，并去重"""
    normalized = []
    for raw_id in raw_node_list or []:
        node_id = normalize_node_id(raw_id)
        if node_id and node_id not in normalized:
            normalized.append(node_id)
    return normalized

def extract_query_terms(query):
    """提取中英文检索词；中文长句额外切成2-4字片段提高召回"""
    terms = []
    for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", query):
        part = part.lower().strip()
        if len(part) < 2:
            continue
        terms.append(part)
        if re.fullmatch(r"[\u4e00-\u9fff]+", part) and len(part) > 2:
            for size in range(2, min(4, len(part)) + 1):
                terms.extend(part[i:i + size] for i in range(len(part) - size + 1))

    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    return unique_terms

def keyword_score(query, node):
    """本地兜底检索：按问题关键词在标题/摘要/正文中的命中情况打分"""
    query_terms = extract_query_terms(query)
    if not query_terms:
        return 0

    title = str(node.get("title", "")).lower()
    summary = str(node.get("summary", "")).lower()
    text = str(node.get("text", "")).lower()

    score = 0
    for term in query_terms:
        if term in title:
            score += 8
        if term in summary:
            score += 4
        if term in text:
            score += 1
    return score

def classify_query_type(query):
    """按问题形态选择检索/回答策略"""
    if is_metadata_query(query):
        return "metadata"

    numeric_terms = [
        "指标", "表", "表格", "对比", "比较", "最大", "最小", "浮动", "波动",
        "差值", "提升", "降低", "多少", "P", "R", "mAP", "accuracy", "precision"
    ]
    negative_terms = ["是否", "有没有", "未", "不是", "只使用", "没有"]

    if any(term in query for term in numeric_terms):
        return "numeric_table"
    if any(term in query for term in negative_terms):
        return "negative_check"
    return "general"

def table_keyword_score(query, node):
    """表格/指标问题额外偏向包含表格、数字和指标名的节点"""
    base = keyword_score(query, node)
    text = str(node.get("text", ""))
    summary = str(node.get("summary", ""))
    title = str(node.get("title", ""))
    haystack = f"{title}\n{summary}\n{text}"

    table_terms = ["表", "指标", "P", "R", "mAP", "mAP@.5", "mAP@.5:.95", "实验", "对比"]
    score = base
    for term in table_terms:
        if term in haystack:
            score += 5
    score += min(len(re.findall(r"\b0\.\d+\b|\b\d+\.\d+\b", haystack)), 30)
    return score

def fallback_tree_search(query):
    """当LLM没有返回有效node_id时，用本地关键词检索兜底"""
    scored_nodes = []
    query_type = classify_query_type(query)
    for node in iter_nodes(global_doc["structure"]):
        node_id = node.get("node_id")
        if not node_id:
            continue
        score = table_keyword_score(query, node) if query_type == "numeric_table" else keyword_score(query, node)
        if score > 0:
            scored_nodes.append((score, node_id))

    scored_nodes.sort(reverse=True)
    node_list = [node_id for _, node_id in scored_nodes[:config["MAX_NODES"]]]
    return {
        "thinking": "LLM没有返回有效节点，已使用本地关键词命中标题、摘要和正文进行兜底检索。",
        "node_list": node_list
    }

def numeric_table_tree_search(query):
    """表格/指标问题优先检索包含指标和数字的节点"""
    scored_nodes = []
    for node in iter_nodes(global_doc["structure"]):
        node_id = node.get("node_id")
        if not node_id:
            continue
        score = table_keyword_score(query, node)
        if score > 0:
            scored_nodes.append((score, node_id))

    scored_nodes.sort(reverse=True)
    node_list = [node_id for _, node_id in scored_nodes[:max(config["MAX_NODES"], 4)]]
    return {
        "thinking": "这是表格、指标或数值对比问题。已优先检索包含实验表格、mAP、P/R等指标和大量数字的节点。",
        "node_list": node_list
    }

def is_metadata_query(query):
    """作者、题名、学校等文档级元信息通常在首页，不一定进入章节摘要"""
    metadata_terms = [
        "作者", "谁写", "谁的", "姓名", "题目", "题名", "标题",
        "学校", "学院", "专业", "导师", "指导教师", "毕业论文",
        # 专利首页元数据通常不在正文目录节点中，命中这些词时优先检索文档开头节点。
        "授权公告号", "公告号", "申请号", "申请公布号", "公布号",
        "申请日", "授权公告日", "优先权", "专利权人", "发明人",
        "代理机构", "代理师", "专利代理师", "Int.Cl", "分类号",
        "国际专利分类", "审查员", "发明名称"
    ]
    return any(term in query for term in metadata_terms)

def metadata_tree_search(query):
    """元信息问题优先检索首页/前言节点"""
    node_list = []
    for node in iter_nodes(global_doc["structure"]):
        node_id = node.get("node_id")
        if node_id and node_id not in node_list:
            node_list.append(node_id)
        if len(node_list) >= config["METADATA_NODE_COUNT"]:
            break

    return {
        "thinking": "这是作者、题名、学校等文档级元信息问题。此类信息通常位于首页、摘要页或页眉中，因此优先检索文档开头节点。",
        "node_list": node_list
    }

def try_answer_author_from_context(query, context):
    """从常见中文论文页眉格式中直接抽取作者"""
    if not any(term in query for term in ["作者", "谁写", "谁的", "姓名"]):
        return None

    explicit_patterns = [
        r"作者[：:\s]+([\u4e00-\u9fff]{2,4})",
        r"姓\s*名[：:\s]+([\u4e00-\u9fff]{2,4})",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, context)
        if match:
            return f"论文作者是{match.group(1)}。"

    label_words = {"摘要", "关键词", "题目", "题名", "标题", "学院", "专业", "学校"}
    title_words = ("基于", "研究", "设计", "检测", "分析", "系统", "方法")
    for line in context.splitlines():
        match = re.search(r"^\s*([\u4e00-\u9fff]{2,4})[：:]\s*([^。\n]{4,80})", line)
        if not match:
            continue
        name, title = match.group(1), match.group(2)
        if name in label_words:
            continue
        if any(word in title for word in title_words):
            return f"论文作者是{name}。"
    return None

def extract_table6_metric_rows(context):
    """从综合实验表6中抽取 P/R/mAP 指标行"""
    table_match = re.search(r"表\s*6(?P<body>.*?)(?:根据上表|5\.3|图15|图16)", context, flags=re.S)
    if table_match:
        segment = table_match.group("body")
    else:
        marker = re.search(r"NWD\s+CBAM\s+Biformer\s+C2f\s+P\s+R\s+mAP@\.5\s+mAP@\.5:\.95", context)
        segment = context[marker.start(): marker.start() + 2000] if marker else context

    rows = []
    for line in segment.splitlines():
        floats = [float(item) for item in re.findall(r"(?<!\d)(?:0|1)\.\d+", line)]
        if len(floats) >= 4:
            rows.append(floats[-4:])
    return rows

def try_answer_numeric_table(query, context):
    """用代码处理一部分表格/指标类问题，避免LLM只做关键词抽取"""
    is_variation_question = any(term in query for term in ["浮动", "波动", "变化幅度", "差异最大"])
    asks_largest = any(term in query for term in ["最大", "最多", "最明显"])
    asks_metric = any(term in query for term in ["指标", "P", "R", "mAP"])
    if not (is_variation_question and asks_largest and asks_metric):
        return None

    rows = extract_table6_metric_rows(context)
    if len(rows) < 2:
        return None

    metric_names = ["P", "R", "mAP@.5", "mAP@.5:.95"]
    stats = []
    for col_idx, metric in enumerate(metric_names):
        values = [row[col_idx] for row in rows]
        min_value = min(values)
        max_value = max(values)
        diff = max_value - min_value
        stats.append((diff, metric, min_value, max_value))

    stats.sort(reverse=True, key=lambda item: item[0])
    diff, metric, min_value, max_value = stats[0]
    detail = "；".join(
        f"{name}: {low:.3f}-{high:.3f}, 差值{delta:.3f}"
        for delta, name, low, high in stats
    )
    return (
        f"根据最终综合实验表中的数值计算，浮动最大的是 {metric}。"
        f"它的最小值为 {min_value:.3f}，最大值为 {max_value:.3f}，差值约为 {diff:.3f}。"
        f"各指标范围为：{detail}。"
    )
# ==============================================================

async def call_llm(prompt, temperature=None):
    if temperature is None:
        temperature = config["TEMPERATURE"]
    
    prompt = clean_unicode(prompt)
    
    response = await client.chat.completions.create(
        model=config["MODEL_NAME"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

async def tree_search(query):
    """基于树结构的推理式检索（已适配你的JSON结构）"""
    query_type = classify_query_type(query)
    if query_type == "metadata":
        return metadata_tree_search(query)
    if query_type == "numeric_table":
        return numeric_table_tree_search(query)

    # 创建一个轻量级的树副本，只保留标题和摘要
    tree_light = remove_fields_from_nodes(json.loads(json.dumps(global_doc["structure"])), fields=['text'])
    
    search_prompt = f"""
你是一个专业的文档检索专家。给定一个问题和一个文档的树状结构，
每个节点包含node_id、title和summary字段。

你的任务：
1. 仔细分析问题的核心需求
2. 遍历整个文档树，找出所有最可能包含答案的节点
3. 按照相关性从高到低排序，最多返回{config["MAX_NODES"]}个节点

问题：{query}

文档树结构：
{json.dumps(tree_light, indent=2, ensure_ascii=False)}

请严格按照以下JSON格式回复，不要输出任何其他内容：
{{
    "thinking": "详细说明你的思考过程",
    "node_list": ["node_id_1", "node_id_2", "node_id_3"]
}}
"""
    
    result = await call_llm(search_prompt)
    parsed = parse_llm_json(result)
    if not parsed:
        print("❌ LLM返回的JSON格式错误，原始输出：")
        print("-" * 50)
        print(result)
        print("-" * 50)
        return fallback_tree_search(query)

    parsed["node_list"] = normalize_node_list(parsed.get("node_list", []))
    if not parsed["node_list"]:
        return fallback_tree_search(query)

    return parsed

def extract_context(node_ids):
    """提取指定节点的完整上下文内容"""
    context = []
    
    for node_id in node_ids:
        node = global_node_map.get(node_id)
        if not node:
            continue
            
        # 获取节点及其所有子节点的完整内容
        full_content = get_node_content(node)
        full_content = clean_unicode(full_content)
        
        if config["SHOW_DEBUG_INFO"]:
            print(f"🔍 调试：节点 {node_id} ({node['title']}) 提取到 {len(full_content)} 个字符")
        
        if full_content.strip():
            context.append(f"【{node['title']}】(第{node['start_index']}-{node['end_index']}页)\n{full_content}")
        else:
            if config["SHOW_DEBUG_INFO"]:
                print(f"⚠️  调试：节点 {node_id} 没有找到任何文本内容")
    
    return "\n\n---\n\n".join(context)

def expand_node_ids(node_ids):
    """扩展父节点和相邻节点，降低检索边界切断表格/结论的概率"""
    if not config.get("EXPAND_CONTEXT", True):
        return node_ids

    expanded = []
    order_index = {node_id: idx for idx, node_id in enumerate(global_node_order)}

    def add(node_id):
        if node_id and node_id in global_node_map and node_id not in expanded:
            expanded.append(node_id)

    for node_id in node_ids:
        if config.get("EXPAND_PARENT", True):
            add(global_parent_map.get(node_id))

        neighbor_count = int(config.get("EXPAND_NEIGHBORS", 0))
        idx = order_index.get(node_id)
        if idx is not None and neighbor_count > 0:
            start = max(0, idx - neighbor_count)
            end = min(len(global_node_order), idx + neighbor_count + 1)
            for neighbor_id in global_node_order[start:end]:
                add(neighbor_id)

        add(node_id)

    return expanded

def retrieve_context(query, node_ids):
    """根据问题类型提取上下文；复杂问题会自动扩大节点范围"""
    query_type = classify_query_type(query)
    if query_type in {"numeric_table", "comparison", "negative_check"}:
        node_ids = expand_node_ids(node_ids)
    return extract_context(node_ids), node_ids

async def answer_question(query, context):
    if not context.strip():
        return "❌ 错误：没有提取到任何上下文内容"

    direct_answer = try_answer_author_from_context(query, context)
    if direct_answer:
        return direct_answer

    numeric_answer = try_answer_numeric_table(query, context)
    if numeric_answer:
        return numeric_answer
    
    answer_prompt = f"""
请严格基于以下上下文回答问题，不要编造任何上下文之外的信息。
如果上下文中有明确答案，请直接给出答案。
如果问题需要比较、排序、计算差值、判断“最大/最小/浮动/提升”等，而上下文提供了相关表格或数字，请基于上下文中的数字进行推理和计算；不要因为结论没有被原文直接写出就回答找不到。
进行数值比较时，请列出你使用的关键数字、简单计算过程和结论。
如果问题涉及数量、比例、实验结果或数据集规模，请严格根据上下文列出原始数字，并在需要时进行简单计算。不要把单项数量误认为总数。
如果问题询问论文作者、题名等元信息，注意识别首页、摘要页、页眉中的格式，例如“姓名：论文题名”通常表示该姓名是作者。
只有当上下文既没有直接答案，也没有足够数字或事实可推导答案时，才回答"根据提供的上下文无法找到答案"。

问题：{query}

上下文：
{context}

答案：
"""
    
    if config["SHOW_DEBUG_INFO"]:
        print(f"\n📤 发送给LLM的完整上下文长度：{len(context)} 字符")
    
    return await call_llm(answer_prompt)

def print_help():
    print("\n📖 可用命令：")
    print("  输入任意问题 - 进行RAG问答")
    print("  help / ?     - 显示此帮助信息")
    print("  config       - 显示当前配置")
    print("  set <key> <value> - 修改配置")
    print("               例如：set MAX_NODES 5")
    print("               例如：set SHOW_DEBUG_INFO False")
    print("  reload       - 重新加载当前JSON文件")
    print("  clear        - 清屏")
    print("  quit / exit  - 退出程序\n")

def print_config():
    print("\n⚙️ 当前配置：")
    for key, value in config.items():
        print(f"  {key} = {value}")
    print()

def terminal_text_width(text):
    """估算终端显示宽度，用于中文输入时正确移动光标"""
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width

def redraw_editable_line(prompt, buffer, cursor, state):
    """重绘当前输入行，并把光标放回编辑位置"""
    text = "".join(buffer)
    rendered = prompt + text
    current_width = terminal_text_width(rendered)
    last_width = state.get("last_width", 0)

    sys.stdout.write("\r" + rendered)
    if last_width > current_width:
        sys.stdout.write(" " * (last_width - current_width))

    sys.stdout.write("\r" + rendered)
    right_width = terminal_text_width("".join(buffer[cursor:]))
    if right_width:
        sys.stdout.write(f"\033[{right_width}D")
    sys.stdout.flush()
    state["last_width"] = current_width

def editable_input(prompt):
    """支持左右方向键、Home/End、Delete/Backspace 的输入函数"""
    if os.name != "nt" or not sys.stdin.isatty():
        try:
            import readline  # noqa: F401
        except ImportError:
            pass
        return input(prompt)

    import msvcrt

    buffer = []
    cursor = 0
    state = {"last_width": terminal_text_width(prompt)}
    sys.stdout.write(prompt)
    sys.stdout.flush()

    while True:
        char = msvcrt.getwch()

        if char == "\x03":
            raise KeyboardInterrupt
        if char in ("\r", "\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            return "".join(buffer)
        if char == "\b":
            if cursor > 0:
                del buffer[cursor - 1]
                cursor -= 1
                redraw_editable_line(prompt, buffer, cursor, state)
            continue
        if char == "\x01":  # Ctrl+A
            cursor = 0
            redraw_editable_line(prompt, buffer, cursor, state)
            continue
        if char == "\x05":  # Ctrl+E
            cursor = len(buffer)
            redraw_editable_line(prompt, buffer, cursor, state)
            continue
        if char == "\x15":  # Ctrl+U
            del buffer[:cursor]
            cursor = 0
            redraw_editable_line(prompt, buffer, cursor, state)
            continue
        if char in ("\x00", "\xe0"):
            key = msvcrt.getwch()
            if key == "K" and cursor > 0:  # Left
                cursor -= 1
            elif key == "M" and cursor < len(buffer):  # Right
                cursor += 1
            elif key == "G":  # Home
                cursor = 0
            elif key == "O":  # End
                cursor = len(buffer)
            elif key == "S" and cursor < len(buffer):  # Delete
                del buffer[cursor]
            redraw_editable_line(prompt, buffer, cursor, state)
            continue
        if char == "\t":
            char = "    "
        if char.isprintable():
            for item in char:
                buffer.insert(cursor, item)
                cursor += 1
            redraw_editable_line(prompt, buffer, cursor, state)

async def load_document():
    """加载整个文档对象（已适配你的JSON结构）"""
    global global_doc, global_node_map, global_parent_map, global_node_order
    try:
        print(f"📂 正在加载文档：{config['JSON_PATH']}...")
        with open(config["JSON_PATH"], "r", encoding="utf-8", errors="replace") as f:
            global_doc = json.load(f)
        
        print("🧹 正在清理文档中的无效字符...")
        global_doc["structure"] = [clean_node(node) for node in global_doc["structure"]]
        
        # 预先生成node_id到节点的映射、父节点映射和文档顺序
        print("🗺️  正在构建节点索引...")
        global_node_map, global_parent_map, global_node_order = build_node_indexes(global_doc["structure"])
        
        total_nodes = count_nodes(global_doc["structure"])
        nodes_with_text = count_nodes_with_text(global_doc["structure"])
        
        print(f"✅ 文档加载完成")
        print(f"   文档名称：{global_doc.get('doc_name', '未知')}")
        print(f"   文档描述：{global_doc.get('doc_description', '无')[:100]}...")
        print(f"   总节点数：{total_nodes}")
        print(f"   有text字段的节点数：{nodes_with_text}")
        if nodes_with_text == 0:
            print("⚠️  当前结构文件没有正文text，回答会缺少上下文。请重新生成结构：")
            print("   python run_pageindex.py --pdf_path uploads/Yolov5.pdf --if-add-node-text yes")
        
        return True
    except Exception as e:
        print(f"❌ 加载文档失败：{e}")
        import traceback
        traceback.print_exc()
        return False

async def process_query(query):
    """处理用户的查询"""
    global global_doc, global_node_map
    
    if global_doc is None or global_node_map is None:
        print("❌ 文档未加载，请先使用reload命令加载")
        return
    
    print(f"\n🔍 正在检索与问题相关的节点...")
    search_result = await tree_search(query)
    if not search_result:
        return
    
    print("\n🧠 === 检索推理过程 ===")
    print(search_result["thinking"])
    
    print("\n📑 === 检索到的节点 ===")
    for i, node_id in enumerate(search_result["node_list"], 1):
        node = global_node_map.get(node_id)
        if node:
            print(f"{i}. Node {node_id}: {node['title']} (第{node['start_index']}-{node['end_index']}页)")
    
    print("\n📄 正在提取上下文...")
    context, context_node_ids = retrieve_context(query, search_result["node_list"])
    if context_node_ids != search_result["node_list"]:
        print("\n📎 === 扩展后的上下文节点 ===")
        for i, node_id in enumerate(context_node_ids, 1):
            node = global_node_map.get(node_id)
            if node:
                print(f"{i}. Node {node_id}: {node['title']} (第{node['start_index']}-{node['end_index']}页)")
    
    if config["SHOW_CONTEXT_PREVIEW"] and context.strip():
        print(f"\n📝 === 上下文预览（前{config['CONTEXT_PREVIEW_LENGTH']}字）===")
        preview = context[:config["CONTEXT_PREVIEW_LENGTH"]] + "..." if len(context) > config["CONTEXT_PREVIEW_LENGTH"] else context
        print(preview)
    
    print("\n🤖 正在生成答案...")
    answer = await answer_question(query, context)
    
    print("\n✅ === 最终答案 ===")
    print(answer)
    print("\n" + "="*80 + "\n")

async def main():
    print("="*80)
    print("📚 PageIndex 交互式 RAG 测试工具（最终版）")
    print("="*80)
    
    if len(sys.argv) > 1:
        config["JSON_PATH"] = sys.argv[1]
    
    if not await load_document():
        return
    
    print_help()
    
    # 主循环
    while True:
        try:
            query = editable_input("💬 请输入你的问题（输入help查看帮助）：").strip()
            
            if not query:
                continue
                
            # 处理命令
            if query.lower() in ["quit", "exit", "q"]:
                print("👋 再见！")
                break
            elif query.lower() in ["help", "?"]:
                print_help()
                continue
            elif query.lower() == "config":
                print_config()
                continue
            elif query.lower() == "reload":
                await load_document()
                continue
            elif query.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif query.lower().startswith("set "):
                parts = query.split(maxsplit=2)
                if len(parts) == 3:
                    key = parts[1].upper()
                    value = parts[2]
                    
                    if key in config:
                        if isinstance(config[key], bool):
                            value = value.lower() in ["true", "yes", "1"]
                        elif isinstance(config[key], int):
                            try:
                                value = int(value)
                            except ValueError:
                                print(f"❌ {key} 必须是整数")
                                continue
                        
                        config[key] = value
                        print(f"✅ 已设置 {key} = {value}")
                        
                        if key == "JSON_PATH":
                            print("💡 请输入 'reload' 命令加载新的文档")
                    else:
                        print(f"❌ 未知配置项：{key}")
                else:
                    print("❌ 用法：set <key> <value>")
                continue
            
            # 处理普通查询
            await process_query(query)
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到Ctrl+C，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{e}")
            import traceback
            traceback.print_exc()
            print("请重试或输入quit退出\n")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
