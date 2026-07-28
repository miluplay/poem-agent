"""诗文检索工具契约。"""

from __future__ import annotations

from ..retrieval import hybrid_search


def search_poems(query: str, top_k: int = 5) -> list[dict]:
    """混合检索诗文，融合标题、正文/赏析语义和标签信号。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k 必须是正整数")

    return hybrid_search(query.strip(), top_k)
