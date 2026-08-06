"""引用绑定与引用标记清理。"""

import re
from collections.abc import Mapping

from ..contracts import EvidenceItem

_LEGAL_CITE = re.compile(r"\[诗\d+-(?:appr|note)-\d+\]")
_CITE_LIKE = re.compile(r"\[诗\d+[^\]]*\]")
_SESSION_CITE = re.compile(r"\[诗(\d+)-((?:appr|note)-\d+)\]")
_DANGLING_DEGRADED_NOTICE = (
    "（部分解读未能匹配到可靠出处，已尽力修正，请谨慎参考）"
)


def list_dangling_citations(evidence: list[EvidenceItem]) -> list[str]:
    """返回按答案引用顺序排列的“诗号-段号 + 原因”简明清单。"""
    dangling: list[str] = []
    for item in evidence:
        if not isinstance(item, dict) or item.get("dangling") is not True:
            continue
        citation = item.get("citation") or item.get("evidence_id") or "未知引用"
        reason = item.get("reason") or "无法匹配出处"
        dangling.append(f"[{citation}]：{reason}")
    return dangling


def _append_degraded_notice(answer: str) -> str:
    """保留悬空正文和 marker，仅在尾部追加一次诚实提示。"""
    stripped = answer.rstrip()
    if stripped.endswith(_DANGLING_DEGRADED_NOTICE):
        return stripped
    separator = "\n" if stripped else ""
    return f"{stripped}{separator}{_DANGLING_DEGRADED_NOTICE}"


def _strip_invalid_citation_markers(answer: str) -> str:
    """静默剥离正文中形似引用但不符合合法格式的 marker。"""
    cleaned = _CITE_LIKE.sub(
        lambda match: (
            match.group(0)
            if _LEGAL_CITE.fullmatch(match.group(0))
            else ""
        ),
        answer,
    )
    # 清理剥离后形成的双空格、中文句中的孤立空格和行尾空格。
    cleaned = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", cleaned)
    cleaned = re.sub(
        r"(?<=[\u3400-\u9fff，。！？；：、])[ \t]+"
        r"(?=[\u3400-\u9fff，。！？；：、])",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[ \t]+(?=[，。！？；：、,.!?;:])", "", cleaned)
    return re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)


def collect_evidence(
    answer: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
    *,
    cached_details: Mapping[str, dict] | None = None,
) -> list[EvidenceItem]:
    """★ 引用绑定核心:
    1. 从 answer 抽出所有 [诗N-appr-x]/[诗N-note-x] 引用;
    2. 用 N 经 session_poems 找 poem_id;
    3. 用 poem_id + "#" + 诗内短 id 定位完整证据块。
    """
    session_poems = session_poems or {}
    citations = _extract_session_citations(answer)

    # 完整 evidence_id 全局唯一，因此索引不会再被跨诗短 id 覆盖。
    index = _build_evidence_index(
        trajectory, cached_details=cached_details
    )

    evidence: list[EvidenceItem] = []
    for poem_number, short_evidence_id in citations:
        poem_id = session_poems.get(poem_number)
        if poem_id is None:
            evidence.append(
                _dangling_evidence(
                    poem_number,
                    short_evidence_id,
                    reason="引用了未取详情的诗",
                )
            )
            continue

        full_evidence_id = f"{poem_id}#{short_evidence_id}"
        block = index.get(full_evidence_id)
        if block is not None:
            evidence.append({**block, "poem_number": poem_number})
        else:
            evidence.append(
                _dangling_evidence(
                    poem_number,
                    short_evidence_id,
                    poem_id=poem_id,
                    reason="段编号不存在",
                )
            )
    return evidence


def _extract_session_citations(answer: str) -> list[tuple[int, str]]:
    """提取引用并去重，同时保留它们在答案中的首次出现顺序。"""
    citations: list[tuple[int, str]] = []
    for poem_number, short_evidence_id in _SESSION_CITE.findall(answer):
        citation = (int(poem_number), short_evidence_id)
        if citation not in citations:
            citations.append(citation)
    return citations


def _dangling_evidence(
    poem_number: int,
    short_evidence_id: str,
    *,
    reason: str,
    poem_id: str | None = None,
) -> EvidenceItem:
    """统一构造两类悬空引用，保留模型原始的会话引用以便排查。"""
    return {
        "evidence_id": (
            f"{poem_id}#{short_evidence_id}"
            if poem_id is not None
            else short_evidence_id
        ),
        "text": None,
        "poem_id": poem_id,
        "poem_number": poem_number,
        "citation": f"诗{poem_number}-{short_evidence_id}",
        "dangling": True,
        "reason": reason,
    }


def _build_evidence_index(
    trajectory: list,
    *,
    cached_details: Mapping[str, dict] | None = None,
) -> dict:
    """遍历轨迹里所有 get_poem_detail 观察，建完整 evidence_id → 块索引。
    缓存先进入索引，本轮合法观察随后覆盖同一作品的缓存内容。"""
    index: dict[str, dict] = {}
    if cached_details is not None:
        if not isinstance(cached_details, Mapping):
            raise TypeError("cached_details 必须是 poem_id → detail 的映射")
        for poem_id, detail in cached_details.items():
            if not isinstance(poem_id, str):
                continue
            _add_detail_to_evidence_index(
                index, detail, expected_poem_id=poem_id
            )

    for step in trajectory:
        if not isinstance(step, dict):
            continue
        if step.get("action") != "get_poem_detail":
            continue
        obs = step.get("observation")
        _add_detail_to_evidence_index(index, obs)
    return index


def _add_detail_to_evidence_index(
    index: dict[str, dict],
    detail,
    *,
    expected_poem_id: str | None = None,
) -> None:
    """安全忽略不符合详情/evidence 基本结构的缓存或轨迹条目。"""
    if not isinstance(detail, dict) or "error" in detail:
        return
    poem_id = detail.get("poem_id")
    title = detail.get("title")
    if (
        not isinstance(poem_id, str)
        or not poem_id
        or (expected_poem_id is not None and poem_id != expected_poem_id)
        or not isinstance(title, str)
    ):
        return
    collections = []
    for key in ("appreciation", "annotations"):
        items = detail.get(key)
        if not isinstance(items, list):
            return
        validated = []
        for item in items:
            if not isinstance(item, dict):
                return
            full_id = item.get("evidence_id")
            text = item.get("text")
            if (
                not isinstance(full_id, str)
                or not full_id.startswith(f"{poem_id}#")
                or not isinstance(text, str)
            ):
                return
            validated.append((full_id, text))
        collections.append(validated)

    for items in collections:
        for full_id, text in items:
            block = {
                "evidence_id": full_id,
                "text": text,
                "poem_id": poem_id,
                "title": title,
            }
            index[full_id] = block
            # 语料中的注释 ID 沿用 anno-*，对外合法 marker 使用 note-*。
            if "#anno-" in full_id:
                index[full_id.replace("#anno-", "#note-", 1)] = block
