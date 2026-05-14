import os
import json
import copy
import math
import random
import re
from .utils import *
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):    
    title=item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title':title, 'page_number': None}
    
    
    page_number = item['physical_index']
    page_text = page_list[page_number-start_index][0]

    
    prompt = f"""
    Your job is to check if the given section appears or starts in the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    Reply format:
    {{
        
        "thinking": <why do you think the section appears or starts in the page_text>
        "answer": "yes or no" (yes if the section appears or starts in the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if 'answer' in response:
        answer = response['answer']
    else:
        answer = 'no'
    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, model=None, logger=None):    
    prompt = f"""
    You will be given the current section title and the current page_text.
    Your job is to check if the current section starts in the beginning of the given page_text.
    If there are other contents before the current section title, then the current section does not start in the beginning of the given page_text.
    If the current section title is the first content in the given page_text, then the current section starts in the beginning of the given page_text.

    Note: do fuzzy matching, ignore any space inconsistency in the page_text.

    The given section title is {title}.
    The given page_text is {page_text}.
    
    reply format:
    {{
        "thinking": <why do you think the section appears or starts in the page_text>
        "start_begin": "yes or no" (yes if the section starts in the beginning of the page_text, no otherwise)
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = await llm_acompletion(model=model, prompt=prompt)
    response = extract_json(response)
    if logger:
        logger.info(f"Response: {response}")
    return response.get("start_begin", "no")







#并发检查每个目录标题是否出现在其 physical_index 对应页的开头，并给目录项加上 appear_start 字段，为后续准确切分章节范围做准备。
#为什么？  因为后面要把扁平目录变成树，并计算章节范围。  这时候需要知道一个目录标题是不是“真的出现在页面开头”。
#如果这个章节不是从这一页开头开始，切分正文时要更谨慎。  如果标题确实在该页开头附近，后续可以更放心地把这一页作为章节开始页。
async def check_title_appearance_in_start_concurrent(structure, page_list, model=None, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")
    
    # skip items without physical_index
    #处理没有物理页码的目录项：如果某个目录项连 physical_index 都没有，那就没法去对应页检查标题，所以直接标记"appear_start": "no"

    #一开始没有页码的项会被过滤掉，但“页码越界后被置为 None”的项仍可能留下来，所以这里还要再处理一次。
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, model=model, logger=logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


def toc_detector_single_page(content, model=None):
    prompt = f"""
    Your job is to detect if there is a table of content provided in the given text.

    Given text: {content}

    return the following JSON format:
    {{
        "thinking": <why do you think there is a table of content in the given text>
        "toc_detected": "<yes or no>",
    }}

    Directly return the final JSON structure. Do not output anything else.
    Please note: abstract,summary, notation list, figure list, table list, etc. are not table of contents."""

    response = llm_completion(model=model, prompt=prompt)
    # print('response', response)
    json_content = extract_json(response)    
    return json_content['toc_detected']


def check_if_toc_extraction_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a partial document  and a  table of contents.
    Your job is to check if the  table of contents is complete, which it contains all the main sections in the partial document.

    Reply format:
    {{
        "thinking": <why do you think the table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Document:\n' + content + '\n Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']


def check_if_toc_transformation_is_complete(content, toc, model=None):
    prompt = f"""
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['completed']

def extract_toc_content(content, model=None):
    prompt = f"""
    Your job is to extract the full table of contents from the given text, replace ... with :

    Given text: {content}

    Directly return the full table of contents content. Do not output anything else."""

    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    if if_complete == "yes" and finish_reason == "finished":
        return response
    
    chat_history = [
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": response},    
    ]
    prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
    new_response, finish_reason = llm_completion(model=model, prompt=prompt, chat_history=chat_history, return_finish_reason=True)
    response = response + new_response
    if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    attempt = 0
    max_attempts = 5

    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > max_attempts:
            raise Exception('Failed to complete table of contents after maximum retries')

        chat_history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        prompt = f"""please continue the generation of table of contents , directly output the remaining part of the structure"""
        new_response, finish_reason = llm_completion(model=model, prompt=prompt, chat_history=chat_history, return_finish_reason=True)
        response = response + new_response
        if_complete = check_if_toc_transformation_is_complete(content, response, model)
    
    return response

def detect_page_index(toc_content, model=None):
    print('start detect_page_index')
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)
    return json_content['page_index_given_in_toc']

#从指定的目录页中提取目录文本，并判断这个目录文本里是否包含目录页码。
# 输入：
# page_list 是整个 PDF 每一页的文本内容列表。
# toc_page_list 是目录页在 page_list 中的下标列表 
# toc_page_list = [1] 表示目录在：page_list[1]也就是 PDF 的第 2 个物理页。
# model 是传给 detect_page_index(...) 的模型对象或模型名称。

# 根据 toc_page_list 从 page_list 中取出目录页文本，把多页目录拼成一个字符串，清洗掉目录中的点线格式，
# 然后判断这个目录中是否包含页码，最后返回目录文本和是否有页码的标记。

#这样做的目的通常是为了让后续模型或规则更容易识别：
def toc_extractor(page_list, toc_page_list, model):
    def transform_dots_to_colon(text):
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces
        text = re.sub(r'(?:\. ){5,}\.?', ': ', text)
        return text
    
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += page_list[page_index][0]
    toc_content = transform_dots_to_colon(toc_content)
    has_page_index = detect_page_index(toc_content, model=model)#判断目录中是否包含页码
    
    return {
        "toc_content": toc_content, #拼接后的目录文本
        "page_index_given_in_toc": has_page_index  #目录中是否有页码，通常是 "yes" 或 "no"
    }
    #返回一个字典
#举例：
 # 输入
# page_list = [
#     ["封面内容"],
#     ["目录\n第一章 绪论 ........ 1\n第二章 方法 ........ 8\n"],
#     ["第一章 绪论\n这里是正文内容……"],
#     ["第二章 方法\n这里是正文内容……"],
# ]

# toc_page_list = [1]

# result = toc_extractor(page_list, toc_page_list, model)
# 函数内部处理过程

# 先取出目录页：

# page_list[1][0]

# 得到：

# 目录
# 第一章 绪论 ........ 1
# 第二章 方法 ........ 8

# 然后把点线替换成冒号：

# 目录
# 第一章 绪论 :  1
# 第二章 方法 :  8

# 接着调用：

# detect_page_index(toc_content, model=model)

# 判断出目录里有页码：

# has_page_index = "yes"
# 输出
# {
#     "toc_content": "目录\n第一章 绪论 :  1\n第二章 方法 :  8\n",
#     "page_index_given_in_toc": "yes"
# }




def toc_index_extractor(toc, content, model=None):
    print('start toc_index_extractor')
    toc_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nTable of contents:\n' + str(toc) + '\nDocument pages:\n' + content
    response = llm_completion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return json_content



def toc_transformer(toc_content, model=None):
    print('start toc_transformer')
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        last_complete = extract_json(last_complete)
        cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
        return cleaned_response
    
    last_complete = get_json_content(last_complete)
    attempt = 0
    max_attempts = 5
    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > max_attempts:
            raise Exception('Failed to complete toc transformation after maximum retries')
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position+2]
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format: 

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

        if new_complete.startswith('```json'):
            new_complete =  get_json_content(new_complete)
            last_complete = last_complete+new_complete

        if_complete = check_if_toc_transformation_is_complete(toc_content, last_complete, model)
        

    last_complete = extract_json(last_complete)

    cleaned_response=convert_page_to_int(last_complete['table_of_contents'])
    return cleaned_response
    


#从指定页开始，连续扫描 PDF 页面，找出目录页所在的页码列表。
# 参数含义：
# start_page_index：从第几页开始检查，注意这里是 0-based 页码。
# page_list：PDF 每页的内容列表，通常 page_list[i][0] 是第 i 页文本。
# opt：配置对象，里面用到了：
# opt.toc_check_page_num：最多检查到第几页；
# opt.model：用于调用模型判断页面是否是目录。
# logger：日志对象，可选。
def find_toc_pages(start_page_index, page_list, opt, logger=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index
    
    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        #判断当前页是否为目录
        detected_result = toc_detector_single_page(page_list[i][0],model=opt.model)
        #两种情况 一种当前页是目录 则把页码加入列表 如果不是目录并且上一页是目录 代表连续目录页已经结束了，于是停止扫描。
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1
    
    if not toc_page_list and logger:
        logger.info('No toc found')
        
    return toc_page_list
    #find_toc_pages() 会从某一页开始，用模型逐页判断是否为目录页，并返回一段连续的目录页索引。
    #返回的是一个列表 里面放的是目录所在的页码
    #比如[1, 2]  表示目录在 page_list[1] 和 page_list[2]。


def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)  
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data

def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue
    
    if not differences:
        return None
    
    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1
    
    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]
    
    return most_common

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']
    
    return data


#按照token进行分组
def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):    
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        # merge all pages into one text
        page_text = "".join(page_contents)
        return [page_text]
    
    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))
    
    print('divide page_list to groups', len(subsets))
    return subsets

#把一段正文片段 part 和当前目录结构 structure 发给大模型，让模型判断每个目录标题是否在这段正文里开始，并补上对应的物理页码。
# part：PDF 正文的一部分，里面已经带了页码标签，比如 <physical_index_5>。
# structure：结构化后的目录列表，比如每一项有 title、structure、physical_index。
# model：调用的大模型对象或模型名。

def add_page_number_to_toc(part, structure, model=None):
    fill_prompt_seq = """
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X. 

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format. 
        [
            {
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            },
            ...
        ]    
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else."""

    prompt = fill_prompt_seq + f"\n\nCurrent Partial Document:\n{part}\n\nGiven Structure\n{json.dumps(structure, indent=2)}\n"
    current_json_raw = llm_completion(model=model, prompt=prompt)
    json_result = extract_json(current_json_raw)
    
    for item in json_result:
        if 'start' in item:
            del item['start']
    return json_result
    #返回的也是带 physical_index 的结构化目录

def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text

### add verify completeness
def generate_toc_continue(toc_content, part, model=None):
    print('start generate_toc_continue')
    prompt = """
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. \
    
    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            },
            ...
        ]    

    Directly return the additional part of the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part + '\nPrevious tree structure\n:' + json.dumps(toc_content, indent=2)
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if finish_reason == 'finished':
        return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')
    
### add verify completeness “无目录模式”里的第一步目录生成函数。
#它接收一段正文 part，让 LLM 从这段正文里抽取章节结构，生成初始目录，并标出每个章节开始的物理页码。
#输入： part 一组带有<physical_index_X> 标签的页面正文   model调用哪个LLM模型
def generate_toc_init(part, model=None):
    print('start generate_toc_init')
    prompt = """
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. 

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            
        ],


    Directly return the final JSON structure. Do not output anything else."""

    prompt = prompt + '\nGiven text\n:' + part
    response, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

    if finish_reason == 'finished':
         return extract_json(response)
    else:
        raise Exception(f'finish reason: {finish_reason}')



#把正文一页一页喂给 LLM，让模型从正文里自己识别章节标题，并生成带页码的目录结构。
# page_list
#   ↓
# 给每页正文加 <physical_index_X> 标签
#   ↓
# 按 token 长度分组
#   ↓
# 第一组：生成初始目录
#   ↓
# 后续组：继续补充目录
#   ↓
# 把 physical_index 字符串转成整数
#   ↓
# 返回目录列表

def process_no_toc(page_list, start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))#这里算的是 加了 <physical_index_X> 标签后的文本 token 数。
    group_texts = page_list_to_group_text(page_contents, token_lengths)#按照token分组
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number= generate_toc_init(group_texts[0], model)#第一组生成初始目录
    #对后续分组继续补充目录：
    for group_text in group_texts[1:]:
        toc_with_page_number_additional = generate_toc_continue(toc_with_page_number, group_text, model)    
        toc_with_page_number.extend(toc_with_page_number_additional)
    logger.info(f'generate_toc: {toc_with_page_number}')
    #对后续分组继续补充目录：
    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')
    toc_with_page_number = dedupe_repeated_toc_items(toc_with_page_number, logger=logger)

    return toc_with_page_number


#新加的去重清洗 
def dedupe_repeated_toc_items(toc_items, logger=None):
    """Remove repeated continuation nodes generated for the same section.

    In long no-TOC documents, the model sometimes repeats the same
    structure/title on later pages to indicate continuation of one section.
    Keeping those duplicates makes verification fail and can overwrite nodes
    during tree conversion, so keep the first occurrence only.
    """
    cleaned_items = []
    seen = set()
    removed_items = []
    for item in toc_items:
        key = (
            str(item.get('structure', '')).strip(),
            str(item.get('title', '')).strip(),
        )
        if key[0] and key[1] and key in seen:
            removed_items.append(item)
            continue
        seen.add(key)
        cleaned_items.append(item)
    if removed_items and logger:
        logger.info({
            'dedupe_repeated_toc_items_removed_count': len(removed_items),
            'dedupe_repeated_toc_items_removed': removed_items,
        })
    return cleaned_items



# 该函数针对：PDF 有目录文本，但目录项本身没有可用的逻辑页码/目录页码；为了给目录项补上对应的 PDF 物理页码，
# 函数把正文页加上 <physical_index_x> 标签，再让模型根据目录标题和正文内容/正文标题进行匹配，最终返回带 physical_index 的结构化目录。
#目录里没页码，重点是全文搜索标题出现在哪个 physical_index
#输入
# toc_content    从 PDF 目录页里提取出的目录文本。
# toc_page_list  目录页所在页码列表。
# page_list      PDF 每页内容列表
# start_index    物理页码从几开始标。
# model          调用哪个 LLM。
# logger         日志记录器。

#举例：
# PDF第1页：封面
# PDF第2页：摘要
# PDF第3页：目录
# PDF第4页：1 绪论
# PDF第5页：1.1 研究背景
# PDF第6页：1.2 国内外研究现状
# PDF第8页：2 材料与方法

def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, model=None, logger=None):
    page_contents=[]
    token_lengths=[]
    toc_content = toc_transformer(toc_content, model)#目录文本变成结构化列表
    #类似这种的
    # [
    # {"title": "第一章 绪论"},
    # {"title": "1.1 研究背景"},
    # {"title": "1.2 研究意义"},
    # {"title": "第二章 方法"},
    # {"title": "2.1 数据来源"},
    # {"title": "2.2 模型设计"}
    # ]
    logger.info(f'toc_transformer: {toc_content}')

    #给每页正文加物理页码标签  关键  start_index表示page_list[0] 在原始 PDF 里的真实物理页码。
    #依赖 start_index 告诉它正文第一页在原 PDF 中的真实物理页码。
    for page_index in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_index}>\n{page_list[page_index-start_index][0]}\n<physical_index_{page_index}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    #如果 PDF 很长，不能一次塞给模型，就分成几组。
    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number=copy.deepcopy(toc_content)
    #如果 PDF 很长，不能一次塞给模型，就分成几组。
    for group_text in group_texts:
        toc_with_page_number = add_page_number_to_toc(group_text, toc_with_page_number, model)
    logger.info(f'add_page_number_to_toc: {toc_with_page_number}')
    #把字符串页码转成整数
    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number#输出带有 physical_index 的目录列表。






#如果PDF有目录，并且目录里有页码
#把目录页码转换成真实 PDF 物理页码 physical_index
#目录里有页码，重点是算 offset
def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, model=None, logger=None):
    toc_with_page_number = toc_transformer(toc_content, model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))
    
    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_index in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_index+1}>\n{page_list[page_index][0]}\n<physical_index_{page_index+1}>\n\n"

    toc_with_physical_index = toc_index_extractor(toc_no_page_number, main_content, model)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    logger.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    
    #现在前面的流程已经得到一个目录列表 toc_with_page_number，但其中有一项缺少 physical_index
    #尽量补齐 physical_index 后的目录列表
    toc_with_page_number = process_none_page_numbers(toc_with_page_number, page_list, model=model)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number#返回的是已经加好偏移量后的目录列表

    # [
    # {
    #     "structure": "1",
    #     "title": "绪论",
    #     "physical_index": 4
    # },
    # {
    #     "structure": "1.1",
    #     "title": "研究背景",
    #     "physical_index": 5
    # },
    # {
    #     "structure": "2",
    #     "title": "方法",
    #     "physical_index": 8
    # }
    # ]



#目录项里有些章节没有 physical_index 页码，需要补页码的情况。
##check if needed to process none page numbers
# 遍历每个目录项
#    ↓
# 发现某个 item 没有 physical_index
#    ↓
# 找它前一个有 physical_index 的目录项
#    ↓
# 找它后一个有 physical_index 的目录项
#    ↓
# 只在这个页码范围内搜索标题
#    ↓
# 让 LLM 判断标题出现在哪个 physical_index
#    ↓
# 把结果写回 item["physical_index"]

def process_none_page_numbers(toc_items, page_list, start_index=1, model=None):
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break
            
            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_index in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_index - start_index
                if list_index >= 0 and list_index < len(page_list):
                    page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            del item_copy['page']
            result = add_page_number_to_toc(page_contents, item_copy, model)
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                del item['page']
    
    return toc_items#最终返回补完缺失 physical_index 后的目录列表。



#检查PDF 前若干页里有没有目录 
#输入：page_listPDF 每一页的内容列表   opt 是一个配置对象
#check_toc() 会先在 PDF 前部找目录页；如果找到目录，就提取目录并判断有没有页码；
# 如果第一次目录没页码，它会继续往后找更完整的目录，直到找到带页码的目录或超过检查范围。
#核心流程：
# 从 PDF 的前几页寻找目录页；如果没找到目录，就返回空目录；如果找到目录，就用 toc_extractor 抽取目录结构，并判断目录中是否有页码；
# 如果有页码，直接返回；如果没有页码，就继续往后找是否还有其他目录页带页码；如果最终都没有找到带页码的目录，就返回第一次抽取到的无页码目录。
def check_toc(page_list, opt=None):
    toc_page_list = find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)#没找到目录
    #没有目录  自然也没有目录页码
    if len(toc_page_list) == 0:
        print('no toc found')
        return {'toc_content': None, 'toc_page_list': [], 'page_index_given_in_toc': 'no'}
    #有目录的话也要分有没有页码这两种情况
    else:
        print('toc found')
        toc_json = toc_extractor(page_list, toc_page_list, opt.model)

        if toc_json['page_index_given_in_toc'] == 'yes':
            print('index found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        else:
            current_start_index = toc_page_list[-1] + 1 #如果当前目录没有页码，就继续往后找更多目录页
            #因为目录可能跨多页，或者第一次只找到了一部分目录页，而后面的目录页可能带页码。
            
            #处理的情况：前面几页看起来是目录，但没有页码，比如是引言；可能真正带页码的目录页还在后面。
            # 循环条件有三个：
            # 当前抽到的目录没有页码；
            # 还没有超过 PDF 总页数；
            # 还没有超过允许检查的最大页数 opt.toc_check_page_num。
            while (toc_json['page_index_given_in_toc'] == 'no' and 
                   current_start_index < len(page_list) and 
                   current_start_index < opt.toc_check_page_num):
                
                #尝试找后续目录页
                #find_toc_pages返回目录所在的页码列表
                additional_toc_pages = find_toc_pages(
                    start_page_index=current_start_index,
                    page_list=page_list,
                    opt=opt
                )
                
                #如果找不到更多目录页就停止查找
                if len(additional_toc_pages) == 0:
                    break
                
                # 根据 toc_page_list 从 page_list 中取出目录页文本，把多页目录拼成一个字符串，清洗掉目录中的点线格式，
                # 然后判断这个目录中是否包含页码，最后返回目录文本和是否有页码的标记。
                additional_toc_json = toc_extractor(page_list, additional_toc_pages, opt.model)
                if additional_toc_json['page_index_given_in_toc'] == 'yes':
                    print('index found')
                    return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

                else:
                    current_start_index = additional_toc_pages[-1] + 1
            print('index not found')
            #如果一直没找到带页码的目录，就返回第一次抽到的目录
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}
            #返回一个字典包含：
            # {
            # "toc_content": 抽取出来的目录内容或 None,  这个目录内容有没有页码 看page_index_given_in_toc字段就知道了
            # "toc_page_list": 目录页所在的物理页下标列表,
            # "page_index_given_in_toc": 目录中是否包含页码，值为 "yes" 或 "no"       
            # }






################### fix incorrect toc #########################################################
async def single_toc_item_index_fixer(section_title, content, model=None):
    toc_extractor_prompt = """
    You are given a section title and several pages of a document, your job is to find the physical index of the start page of the section in the partial document.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    Reply in a JSON format:
    {
        "thinking": <explain which page, started and closed by <physical_index_X>, contains the start of this section>,
        "physical_index": "<physical_index_X>" (keep the format)
    }
    Directly return the final JSON structure. Do not output anything else."""

    prompt = toc_extractor_prompt + '\nSection Title:\n' + str(section_title) + '\nDocument pages:\n' + content
    response = await llm_acompletion(model=model, prompt=prompt)
    json_content = extract_json(response)    
    return convert_physical_index_to_int(json_content['physical_index'])



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, model=None, logger=None):
    print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}
    
    end_index = len(page_list) + start_index - 1
    
    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']
        
        # Check if list_index is valid
        if list_index < 0 or list_index >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }
        
        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1
        
        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices and i >= 0 and i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index
        
        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            # Add bounds checking to prevent IndexError
            page_list_idx = page_index - start_index
            if page_list_idx >= 0 and page_list_idx < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[page_list_idx][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
            else:
                continue
        content_range = ''.join(page_contents)
        
        physical_index_int = await single_toc_item_index_fixer(incorrect_item['title'], content_range, model)
        
        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index, model)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            print(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    logger.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    logger.info(f'invalid_results: {invalid_results}')

    return toc_with_page_number, invalid_results


#修复错误项
async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, model=None, logger=None):
    print('start fix_incorrect_toc')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        print(f"Fixing {len(current_incorrect)} incorrect results")
        
        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, model, logger)
                
        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break
    
    return current_toc, current_incorrect



#验证目录项的页码准不准。
#该函数会抽查或全查目录项标题是否真的出现在它标注的物理页附近，用来判断当前目录页码是否可靠。

################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, model=None):
    print('start verify_toc')
    # Find the last non-None physical_index
    last_physical_index = None
    #先找最后一个有效页码：从后往前看
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break
    
    # Early return if we don't have valid physical indices
    # 如果没有任何有效页码，准确率直接算 0；
    # 如果最后一个目录项的页码还不到全文一半，也直接算 0。
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []
    
    # Determine which items to check
    #决定检查哪些目录项
    if N is None:
        print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    #准备检查列表  给每个目录项加一个 list_index 
    #list_index 是原来在 list_result 里的位置，后面修复错误页码时用。
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_result
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    #并发检查标题是否真的出现在对应页附近。
    tasks = [
        check_title_appearance(item, page_list, start_index, model)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)
    # 返回的是一种给每个目录项添加了一个answer字段
    # results = [
    # {
    #     "title": "第一章 绪论",
    #     "physical_index": 2,
    #     "list_index": 0,
    #     "answer": "yes"
    # },
    # {
    #     "title": "第二章 方法",
    #     "physical_index": 5,
    #     "list_index": 1,
    #     "answer": "no"
    # },
    # {
    #     "title": "第三章 实验",
    #     "physical_index": 8,
    #     "list_index": 2,
    #     "answer": "yes"
    # }
    # ]

    
    # Process results
    #统计正确数量和错误项
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)
    
    # Calculate accuracy
    #计算准确率
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results#返回的是准确率和目录页码不可靠的目录项




# PageIndex 建树过程中的 目录处理控制器，负责选择目录处理方式、验证结果、修复错误，以及在失败时自动降级重试。
# 尝试生成目录结构
#    ↓
# 过滤无效页码
#    ↓
# 校验目录页码是否正确
#    ↓
# 如果准，直接返回
#    ↓
# 如果有小错，尝试修复
#    ↓
# 如果太差，换一种方法重新处理

#输入
# page_list: PDF每页文本和token数
# mode: 当前处理模式
# toc_content: 从PDF里提取出来的目录文本
# toc_page_list: 哪些页是目录页
# start_index: 从第几页开始处理，默认1
# opt: 配置，比如模型名、toc_check_page_num
# logger: 日志

#支持三种模式
# 1. PDF有目录，而且目录里有页码
# 2. PDF有目录，但目录里没页码
# 3. PDF没有目录，直接让模型从正文生成目录

#流程：
#  PDF 内容生成统一的目录项。
#  删除没有页码的目录项。
#  修正不合法页码。
#  验证并修复目录页码。

#输出：
# 最终返回的是比较可信的、带 physical_index 的扁平目录列表
# 后面的建树逻辑再把它转换成真正的层级树，并补上 start_index / end_index。


################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, logger=None):
    # 外层只负责第一次选择：有带页码目录就用目录页码，否则不用目录。
    # meta_processor 内部负责失败后的降级：带页码目录失败 → 无页码目录匹配 → 全文生成目录。
    print(mode)
    print(f'start_index: {start_index}')
    
    #有目录 有页码
    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, model=opt.model, logger=logger)
    #有目录 没页码 这是第一种模式的备选 主要只在“目录页码不可靠”的降级场景中被调用。
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = process_toc_no_page_numbers(toc_content, toc_page_list, page_list, model=opt.model, logger=logger)
    #没目录 
    else:
        toc_with_page_number = process_no_toc(page_list, start_index=start_index, model=opt.model, logger=logger)

    #删除本来就没有页码的项。     
    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None] 
    
    #修正页码边界
    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number, 
        len(page_list), 
        start_index=start_index, 
        logger=logger
    )
    #校验目录是否准确：  检查目录项的标题，是否真的出现在对应的页附近
    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, model=opt.model)
     
    #写日志  
    logger.info({
        'mode': 'process_toc_with_page_numbers',
        'accuracy': accuracy,
        'incorrect_results': incorrect_results
    })
    #如果完全正确，说明目录页码全部验证通过，直接返回。
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    #如果准确率还可以，但有少量错误：
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results,start_index=start_index, max_attempts=3, model=opt.model, logger=logger)
        return toc_with_page_number
    #如果准确率太低：说明当前方法不靠谱，要换一种更保守的方法
    else:
        #虽然目录里看起来有页码，但用起来不准，那我不信它的页码了，重新根据正文推断。  
        # 降级   process_toc_with_page_numbers——>process_toc_no_page_numbers   process_toc_no_page_numbers->process_no_toc
        if mode == 'process_toc_with_page_numbers':
            return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt, logger=logger)
        #目录也不可靠了，干脆当作没有目录，让模型从正文重新生成。
        elif mode == 'process_toc_no_page_numbers':
            return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt, logger=logger)
        else:
            raise Exception('Processing failed')#说明建树失败。
        
 
 #具体的递归拆分函数
#  process_large_node_recursively() 是一个“递归细分大章节”的函数：如果某个节点页数和 token 数都太大，
#  就在这个节点范围内重新跑无目录解析，生成更细的子节点，然后对子节点继续重复这个过程。
#它针对的是目录项对应文本过长的问题；根因通常是原始目录太粗、生成目录太粗、目录项漏检导致跨度过大，或者章节本身确实很长。
#流程：
# 看当前节点是不是太大
#     不大：直接返回
#     太大：截取这个节点对应页面
#         让 LLM 在这段正文里重新生成子目录
#         检查子目录标题位置
#         转成子树挂到当前节点下面
#         如果子节点还大，继续递归拆
async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]#取出该节点的覆盖页面
    #因为 start_index / end_index 是从 1 开始的物理页码，Python 切片是从 0 开始，所以开始位置要减 1。
    # 也就是说page_list[0]表示的是pdf第一页的内容

    token_num = sum([page[1] for page in node_page_list])#计算token总数
    
    #判断这个节点是否“大到需要继续拆
    #同时满足页数超过 max_page_num_each_node       token 数超过 max_token_num_each_node  才会继续拆
    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        #强制使用meta_processor中的proces_no_doc的逻辑，让 LLM 根据该节点范围内的正文重新生成子目录。
        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        

        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, model=opt.model, logger=logger)
        #注意这里传的是完整 page_list，不是 node_page_list。
        #因为 physical_index 是原 PDF 的全局页码，所以要用完整 PDF 页面来查。


 
        # Filter out items with None physical_index before post_processing
        #在进行post_processing之前过滤掉没有物理页码的节点
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]
        
        #LLM重新生成的节点的第一个标题是当前节点的标题  那么在形成树状列表的时候就要跳过第一个节点 其他节点组成当前节点的子节点
        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():    
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            #更新当前节点结束页：也就是让父节点自己的正文范围停在第一个子节点开始处。
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        
        #LLM重新生成的节点的第一个标题不是当前节点的标题  则所有生成项都作为当前节点的子节点， 
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']
        
    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)
    
    return node

#PDF建树的核心流程
async def tree_parser(page_list, opt, doc=None, logger=None):

    check_toc_result = check_toc(page_list, opt)
    logger.info(check_toc_result)

    #根据有没有可用目录选择处理方式
    # 如果 PDF 有目录，并且目录里自带页码：
    # 用“带页码目录处理逻辑”建结构

    # 否则：
    # 用“无目录处理逻辑”让 LLM 从正文生成结构

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        #PDF 自己有目录，而且目录有页码，那就利用原始目录来建结构。
        toc_with_page_number = await meta_processor(  #await等待一个异步函数执行完成，并拿到它的返回结果。
            page_list, #全文页面内容；
            mode='process_toc_with_page_numbers', 
            start_index=1, 
            toc_content=check_toc_result['toc_content'], 
            toc_page_list=check_toc_result['toc_page_list'], 
            opt=opt,
            logger=logger)
    else:
        ## 没有可直接使用的带页码目录，则让 LLM 根据全文内容自己生成目录结构。
        toc_with_page_number = await meta_processor(
            page_list, 
            mode='process_no_toc', 
            start_index=1, 
            opt=opt,
            logger=logger)
    #如果第一个正式章节不是从第一页开始，它会补一个Preface   为什么？因为后面建树时，会根据相邻目录项的 physical_index 推断每个节点的范围。
    #如果正文第一个章节不是从 PDF 第 1 页开始，就补一个 Preface 节点，把前面的页面纳入结构中，避免丢失文档开头内容。
    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    #检查标题真实出现位置   返回的目录结构会多一个字段 appear_start 表示章节标题是不是在所属页的开头
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, model=opt.model, logger=logger)
    
    #过滤无效节点
    # Filter out items with None physical_index before post_processings
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]
    
    #把扁平目录变成树 这里输出的就是带 structure title start_index end_index nodes 的树状目录列表  每一个目录项变成一个节点node
    toc_tree = post_processing(valid_toc_items, len(page_list))

    #递归拆分过大的节点
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)# 并发运行 tasks 里的所有异步任务，并等它们全部结束。
    #每个子节点如果需要继续拆分，就各自调用 LLM / 做检查。gather 可以让这些异步处理并发进行，提高速度。

    
    return toc_tree#返回一个带页码范围的树结构 就是json的核心结构


#返回pdf文件的完整树状目录列表
def page_index_main(doc, opt=None):
    logger = JsonLogger(doc)#拿 PDF 文件名创建一个日志记录器
    
    #判断pdf和路径是否存在
    is_valid_pdf = (
        (isinstance(doc, str) and os.path.isfile(doc) and doc.lower().endswith(".pdf")) or 
        isinstance(doc, BytesIO)
    )
    if not is_valid_pdf:
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    print('Parsing PDF...')
    #抽取PDF的每一页文本和token数 生成一个page_list
    # [
    # ("第一页文本", 1200),
    # ("第二页文本", 950),
    # ...
    # ]
    page_list = get_page_tokens(doc, model=opt.model)
    #当前建树、找目录、匹配标题这些逻辑，主要依赖的是页面文本，所以经常只看 [0]。token 数一般用于控制分组、避免超过模型上下文长度。

    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum([page[1] for page in page_list])})

    async def page_index_builder():
        structure = await tree_parser(page_list, opt, doc=doc, logger=logger)
        #根据配置给树的各节点补充额外信息，然后整理输出格式
        if opt.if_add_node_id == 'yes':
            write_node_id(structure) # 给节点加 node_id。  
        if opt.if_add_node_text == 'yes':
            add_node_text(structure, page_list)#给节点加原文 text。
        #给节点加摘要 summary，必须先生成正文才可以，所以有两种情况，如果只要求摘要字段的话
        #在生成摘要之后，就要把text字段删掉
        if opt.if_add_node_summary == 'yes':
            if opt.if_add_node_text == 'no': 
                add_node_text(structure, page_list)
            await generate_summaries_for_structure(structure, model=opt.model)
            if opt.if_add_node_text == 'no':
                remove_structure_text(structure)
            if opt.if_add_doc_description == 'yes':#如果命令行参数有让生成简述
                # Create a clean structure without unnecessary fields for description generation
                clean_structure = create_clean_structure_for_description(structure)
                #让 LLM 根据文档树结构，生成一句文档简介。
                doc_description = generate_doc_description(clean_structure, model=opt.model)

                #整理字段顺序
                structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
                return {
                    'doc_name': get_pdf_name(doc),
                    'doc_description': doc_description,
                    'structure': structure,
                }
        structure = format_structure(structure, order=['title', 'node_id', 'start_index', 'end_index', 'summary', 'text', 'nodes'])
        return {
            'doc_name': get_pdf_name(doc),
            'structure': structure,
        }

    return asyncio.run(page_index_builder())
    #最终输出的是一个 Python 字典 dict，里面包含：
    # {
    # "doc_name": PDF 文件名,
    # "structure": 文档目录树
    # }
    #比较典型的最终输出
    # {
    # "doc_name": "example.pdf",
    # "structure": [
    #     {
    #         "title": "Preface",
    #         "start_index": 1,
    #         "end_index": 2
    #     },
    #     {
    #         "title": "第一章 绪论",
    #         "node_id": "1",
    #         "start_index": 3,
    #         "end_index": 10,
    #         "summary": "本章介绍研究背景、研究意义和论文结构。",
    #         "nodes": [
    #             {
    #                 "title": "1.1 研究背景",
    #                 "node_id": "1.1",
    #                 "start_index": 3,
    #                 "end_index": 5,
    #                 "summary": "本节介绍相关研究背景。"
    #             },
    #             {
    #                 "title": "1.2 研究意义",
    #                 "node_id": "1.2",
    #                 "start_index": 6,
    #                 "end_index": 10,
    #                 "summary": "本节说明研究的理论和实践意义。"
    #             }
    #         ]
    #     }
    # ]
    # }












def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None, max_token_num_each_node=None,
               if_add_node_id=None, if_add_node_summary=None, if_add_doc_description=None, if_add_node_text=None):
    
    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return page_index_main(doc, opt)


#检查目录项里的 physical_index 有没有超过 PDF 实际页数；如果超过，就把它清空成 None。并不是删除
#核心逻辑：max_allowed_page = page_list_length + start_index - 1

def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    if not toc_with_page_number:
        return toc_with_page_number
    
    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []
    
    #遍历目录项
    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                if logger:
                    logger.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")
    
    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")
        
    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")
     
    return toc_with_page_number
