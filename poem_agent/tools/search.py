"""诗文检索工具契约。"""

from __future__ import annotations

from ..retrieval import retrieve_poems


_EMPTY_CONDITIONS_ERROR = (
    "未提供任何检索条件（author/dynasty/title/query 均为空）。请检查："
    "① 用户是否点名作者/朝代/篇名→填对应硬条件；"
    "② 是否描述主题/意象/情感→填 query；"
    "③ 若无需检索则改用其他方式回应。"
    "请重新提取参数后再调用，勿用全空参数重复调用。"
)


def _normalize_optional_condition(name: str, value: str | None) -> str | None:
    """校验并去除可选检索条件两端空白。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是字符串或 None")
    normalized = value.strip()
    return normalized or None


def search_poems(
    query: str | None = None,
    author: str | None = None,
    dynasty: str | None = None,
    title: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """按结构化硬条件与语义意图统一检索诗文。"""
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= 20
    ):
        raise ValueError("top_k 必须是 1–20 的整数（不接受布尔值）")

    normalized = {
        "query": _normalize_optional_condition("query", query),
        "author": _normalize_optional_condition("author", author),
        "dynasty": _normalize_optional_condition("dynasty", dynasty),
        "title": _normalize_optional_condition("title", title),
    }
    if all(value is None for value in normalized.values()):
        raise ValueError(_EMPTY_CONDITIONS_ERROR)

    return retrieve_poems(top_k=top_k, **normalized)
