from __future__ import annotations

import json
import re
import PyPDF2

try:
    from .utils import get_number_of_pages, remove_fields
except ImportError:
    from utils import get_number_of_pages, remove_fields


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_pages(pages: str) -> list[int]:
    """Parse a pages string like '5-7', '3,8', or '12' into a sorted list of ints."""
    result = []
    for part in pages.split(','):
        part = part.strip()
        if '-' in part:
            start, end = int(part.split('-', 1)[0].strip()), int(part.split('-', 1)[1].strip())
            if start > end:
                raise ValueError(f"Invalid range '{part}': start must be <= end")
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def _count_pages(doc_info: dict) -> int:
    """Return total page count for a PDF document."""
    if doc_info.get('page_count'):
        return doc_info['page_count']
    if doc_info.get('pages'):
        return len(doc_info['pages'])
    return get_number_of_pages(doc_info['path'])


def _get_pdf_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """Extract text for specific PDF pages (1-indexed). Prefer cached pages, fallback to PDF."""
    cached_pages = doc_info.get('pages')
    if cached_pages:
        page_map = {p['page']: p['content'] for p in cached_pages}
        return [
            {'page': p, 'content': page_map[p]}
            for p in page_nums if p in page_map
        ]
    path = doc_info['path']
    with open(path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total = len(pdf_reader.pages)
        valid_pages = [p for p in page_nums if 1 <= p <= total]
        return [
            {'page': p, 'content': pdf_reader.pages[p - 1].extract_text() or ''}
            for p in valid_pages
        ]


def _get_md_page_content(doc_info: dict, page_nums: list[int]) -> list[dict]:
    """
    For Markdown documents, 'pages' are line numbers.
    Find nodes whose line_num falls within [min(page_nums), max(page_nums)] and return their text.
    """
    min_line, max_line = min(page_nums), max(page_nums)
    results = []
    seen = set()

    def _traverse(nodes):
        for node in nodes:
            ln = node.get('line_num')
            if ln and min_line <= ln <= max_line and ln not in seen:
                seen.add(ln)
                results.append({'page': ln, 'content': node.get('text', '')})
            if node.get('nodes'):
                _traverse(node['nodes'])

    _traverse(doc_info.get('structure', []))
    results.sort(key=lambda x: x['page'])
    return results


def _iter_nodes(nodes):
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        yield node
        if node.get('nodes'):
            yield from _iter_nodes(node['nodes'])


def _infer_page_count(doc_info: dict) -> int:
    for key in ('page_count', 'total_page_number'):
        value = doc_info.get(key)
        if isinstance(value, int) and value > 0:
            return value

    candidates = []
    for node in _iter_nodes(doc_info.get('structure', [])):
        for key in ('start_index', 'end_index'):
            try:
                value = int(node.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                candidates.append(value)
    return max(candidates) if candidates else 0


def _front_text(doc_info: dict, max_nodes: int = 3) -> str:
    parts = []
    for node in _iter_nodes(doc_info.get('structure', [])):
        text = node.get('text')
        if isinstance(text, str) and text.strip():
            parts.append(text)
        if len(parts) >= max_nodes:
            break
    return '\n'.join(parts)


def _clean_field(value):
    value = re.sub(r'[\u00a0\u200b\u2011\u2010\u2013\u2014‑]', '-', str(value or ''))
    value = re.sub(r'[ \t]+', ' ', value)
    value = re.sub(r'\s*\.\s*', '.', value)
    value = re.sub(r'\s+', ' ', value)
    return value.strip(' ;,，、')


def _match_field(text, pattern, flags=0):
    match = re.search(pattern, text, flags)
    if not match:
        return None
    return _clean_field(match.group(1))


def _match_pages(text, label):
    match = re.search(rf'{label}\s*(\d+)\s*页', text)
    return int(match.group(1)) if match else None


def _split_people(value):
    if not value:
        return []
    cleaned = _clean_field(value)
    parts = re.split(r'[\s,，、;；]+', cleaned)
    return [part for part in parts if part]


def extract_patent_metadata(doc_info: dict, text: str = None) -> dict:
    """
    Extract common Chinese patent front-page metadata without calling an LLM.

    The function is intentionally conservative: missing fields are returned as None
    rather than guessed. It works best on PageIndex JSON files whose first node
    contains the patent front page text.
    """
    source_text = text if text is not None else _front_text(doc_info)
    source_text = source_text or ''

    cn_numbers = re.findall(r'\bCN\s*\d+\s*[A-Z]\b', source_text)
    cn_numbers = [_clean_field(item) for item in cn_numbers]
    grant_number = next((item for item in cn_numbers if item.endswith(' B')), None)

    publication_number = _match_field(
        source_text,
        r'申请公布号\s*([A-Z]{2}\s*\d+\s*[A-Z])'
    )
    if not publication_number:
        publication_number = next((item for item in cn_numbers if item.endswith(' A')), None)

    grant_date = _match_field(source_text, r'\(45\)\s*授权公告日\s*([0-9]{4}\.[0-9]{2}\.[0-9]{2})')
    if not grant_date and grant_number:
        pattern = re.escape(grant_number).replace(r'\ ', r'\s*')
        grant_date = _match_field(source_text, pattern + r'\s*([0-9]{4}\.[0-9]{2}\.[0-9]{2})')

    title = _match_field(
        source_text,
        r'\(54\)\s*发明名称\s*(.*?)(?=\(\s*57\s*\)\s*摘要|\(57\)\s*摘要)',
        flags=re.S
    )

    applicant = _match_field(
        source_text,
        r'\(73\)\s*(?:专利权人|申请人)\s*(.*?)(?=\s*地址|\(\s*72\s*\)|\(72\))',
        flags=re.S
    )
    applicant_address = _match_field(
        source_text,
        r'地址\s*(.*?)(?=\(\s*72\s*\)|\(72\))',
        flags=re.S
    )

    inventors_raw = _match_field(
        source_text,
        r'\(72\)\s*发明人\s*(.*?)(?=\(\s*74\s*\)|\(74\))',
        flags=re.S
    )

    return {
        'doc_name': doc_info.get('doc_name'),
        'page_count': _infer_page_count(doc_info),
        'title': title,
        'application_number': _match_field(source_text, r'\(21\)\s*申请号\s*([0-9A-Za-z.\s]+)'),
        'publication_number': publication_number,
        'grant_number': grant_number,
        'application_date': _match_field(source_text, r'\(22\)\s*申请日\s*([0-9]{4}\.[0-9]{2}\.[0-9]{2})'),
        'publication_date': _match_field(source_text, r'\(43\)\s*申请公布日\s*([0-9]{4}\.[0-9]{2}\.[0-9]{2})'),
        'grant_date': grant_date,
        'priority_number': _match_field(source_text, r'\(30\)\s*优先权数据\s*([0-9/,\s]+)(?=\s*[0-9]{4}\.)'),
        'priority_date': _match_field(source_text, r'\(30\)\s*优先权数据\s*[0-9/,\s]+([0-9]{4}\.[0-9]{2}\.[0-9]{2})'),
        'applicant': applicant,
        'patentee': applicant,
        'applicant_address': applicant_address,
        'inventors': _split_people(inventors_raw),
        'agency': _match_field(source_text, r'\(74\)\s*专利代理机构\s*(.*?)(?=专利代理师|\(51\))', flags=re.S),
        'agents': _split_people(_match_field(source_text, r'专利代理师\s*(.*?)(?=\(51\)|Int\.Cl\.)', flags=re.S)),
        'claims_pages': _match_pages(source_text, '权利要求书'),
        'description_pages': _match_pages(source_text, '说明书'),
        'sequence_listing_pages': _match_pages(source_text, '序列表'),
        'drawings_pages': _match_pages(source_text, '附图'),
    }


# ── Tool functions ────────────────────────────────────────────────────────────

def get_document(documents: dict, doc_id: str) -> str:
    """Return JSON with document metadata: doc_id, doc_name, doc_description, type, status, page_count (PDF) or line_count (Markdown)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    result = {
        'doc_id': doc_id,
        'doc_name': doc_info.get('doc_name', ''),
        'doc_description': doc_info.get('doc_description', ''),
        'type': doc_info.get('type', ''),
        'status': 'completed',
    }
    if doc_info.get('type') == 'pdf':
        result['page_count'] = _count_pages(doc_info)
    else:
        result['line_count'] = doc_info.get('line_count', 0)
    return json.dumps(result)


def get_patent_metadata(documents: dict, doc_id: str) -> str:
    """Return structured Chinese patent front-page metadata extracted without an LLM."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    return json.dumps(extract_patent_metadata(doc_info), ensure_ascii=False)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """Return tree structure JSON with text fields removed (saves tokens)."""
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})
    structure = doc_info.get('structure', [])
    structure_no_text = remove_fields(structure, fields=['text'])
    return json.dumps(structure_no_text, ensure_ascii=False)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Retrieve page content for a document.

    pages format: '5-7', '3,8', or '12'
    For PDF: pages are physical page numbers (1-indexed).
    For Markdown: pages are line numbers corresponding to node headers.

    Returns JSON list of {'page': int, 'content': str}.
    """
    doc_info = documents.get(doc_id)
    if not doc_info:
        return json.dumps({'error': f'Document {doc_id} not found'})

    try:
        page_nums = _parse_pages(pages)
    except (ValueError, AttributeError) as e:
        return json.dumps({'error': f'Invalid pages format: {pages!r}. Use "5-7", "3,8", or "12". Error: {e}'})

    try:
        if doc_info.get('type') == 'pdf':
            content = _get_pdf_page_content(doc_info, page_nums)
        else:
            content = _get_md_page_content(doc_info, page_nums)
    except Exception as e:
        return json.dumps({'error': f'Failed to read page content: {e}'})

    return json.dumps(content, ensure_ascii=False)
