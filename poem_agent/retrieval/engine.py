"""统一诗文检索管线。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from .. import store


_ROOT = Path(__file__).resolve().parents[2]
_CHROMA_PATH = _ROOT / "chroma"
_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_CONTENT_COLLECTION = "poem_content"
_APPRECIATION_COLLECTION = "poem_appreciation"
_SEMANTIC_TOP_N = 20


def _build_tag_vocabulary() -> frozenset[str]:
    """从当前诗库动态生成标签词表。"""
    tags: set[str] = set()
    for poem in store.load_poems():
        for tag in poem.get("tags", []):
            if isinstance(tag, str) and tag:
                tags.add(tag)
    return frozenset(tags)


# 词表随进程启动从 poems.json 生成；新增数据后重启即可自动纳入。
_TAGS = _build_tag_vocabulary()


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    """懒加载并复用与建索引相同的 BGE 模型。"""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_chroma_client() -> Any:
    """复用本地持久化 Chroma 客户端。"""
    import chromadb

    return chromadb.PersistentClient(path=str(_CHROMA_PATH))


def _extract_query_tags(query: str) -> set[str]:
    """识别 query 中显式出现的标签，只作为软排序信号。"""
    return {tag for tag in _TAGS if tag in query}


def _calculate_tag_score(poem: dict, query_tags: set[str]) -> float:
    """计算标签软分：诗命中的查询标签数 / 查询中出现的标签总数。"""
    if not query_tags:
        return 0.0
    poem_tags = {tag for tag in poem.get("tags", []) if isinstance(tag, str)}
    return len(poem_tags & query_tags) / len(query_tags)


def _fuse_score(semantic_score: float, tag_score: float) -> float:
    """融合语义分与标签分，权重集中在这里便于后续调参。"""
    return 0.6 * semantic_score + 0.4 * tag_score


def _query_collection(
    collection_name: str,
    query_embedding: list[float],
    candidate_ids: frozenset[str] | None,
) -> dict:
    """查询单个 collection，并用元数据过滤把召回限制在候选池内。"""
    collection = _get_chroma_client().get_collection(name=collection_name)
    result_count = min(_SEMANTIC_TOP_N, collection.count())
    if result_count == 0 or candidate_ids == frozenset():
        return {"metadatas": [[]], "distances": [[]]}

    query_args: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": result_count,
        "include": ["metadatas", "distances"],
    }
    if candidate_ids is not None:
        query_args["where"] = {"poem_id": {"$in": sorted(candidate_ids)}}
    return collection.query(**query_args)


def _semantic_scores(
    query: str, candidate_ids: frozenset[str] | None = None
) -> dict[str, float]:
    """在候选池内检索正文和赏析，并按 poem_id 对两路相似度取最大值。"""
    model = _get_embedding_model()
    embedding = model.encode(
        [_QUERY_PREFIX + query],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_embedding = embedding[0].tolist()

    scores: dict[str, float] = {}
    for collection_name in (_CONTENT_COLLECTION, _APPRECIATION_COLLECTION):
        result = _query_collection(
            collection_name, query_embedding, candidate_ids
        )
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for metadata, distance in zip(metadatas, distances):
            poem_id = metadata.get("poem_id") if metadata else None
            if not isinstance(poem_id, str):
                continue
            # 两个 collection 都使用 cosine；Chroma 返回 distance = 1 - similarity。
            similarity = 1.0 - float(distance)
            scores[poem_id] = max(scores.get(poem_id, float("-inf")), similarity)
    return scores


def _partition_title_matches(
    poems: list[dict], title: str
) -> tuple[list[dict], list[dict]]:
    """按既有标题归一规则拆分精确与部分命中。"""
    normalized_title = store._normalize_title(title)
    if not normalized_title:
        return [], []
    exact_matches: list[dict] = []
    partial_matches: list[dict] = []

    for poem in poems:
        poem_title = poem.get("title")
        if not isinstance(poem_title, str):
            continue
        normalized_poem_title = store._normalize_title(poem_title)
        if normalized_poem_title == normalized_title:
            exact_matches.append(poem)
        elif normalized_title in normalized_poem_title:
            partial_matches.append(poem)

    return exact_matches, partial_matches


def _hard_filter(
    poems: list[dict],
    *,
    author: str | None,
    dynasty: str | None,
    title: str | None,
) -> list[dict]:
    """独立确定标题匹配层级，再与作者、朝代硬条件取交集。"""
    title_candidates = poems
    if title is not None:
        exact_matches, partial_matches = _partition_title_matches(poems, title)
        # 精确标题具有排他性；只有全库完全没有精确命中时才启用部分匹配容错。
        title_candidates = exact_matches if exact_matches else partial_matches

    candidates = [
        poem
        for poem in title_candidates
        if (
            author is None
            or (
                isinstance(poem.get("author"), str)
                and poem["author"].strip() == author
            )
        )
        and (
            dynasty is None
            or (
                isinstance(poem.get("dynasty"), str)
                and poem["dynasty"].strip() == dynasty
            )
        )
    ]
    return candidates


def _lightweight_result(poem: dict, score: float | None) -> dict:
    """把库内详情裁成统一工具的轻量候选结构。"""
    return {
        "poem_id": poem["poem_id"],
        "title": poem["title"],
        "author": poem["author"],
        "dynasty": poem["dynasty"],
        "score": score,
    }


def retrieve_poems(
    *,
    query: str | None = None,
    author: str | None = None,
    dynasty: str | None = None,
    title: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """执行硬过滤 → 候选池 → 软排序 → 截断的统一检索管线。"""
    poems = store.load_poems()
    candidates = _hard_filter(
        poems, author=author, dynasty=dynasty, title=title
    )
    if not candidates:
        return []

    if query is None:
        return [
            _lightweight_result(poem, None) for poem in candidates[:top_k]
        ]

    candidate_ids = frozenset(
        poem["poem_id"]
        for poem in candidates
        if isinstance(poem.get("poem_id"), str)
    )
    semantic_scores = _semantic_scores(
        query,
        None if len(candidates) == len(poems) else candidate_ids,
    )
    query_tags = _extract_query_tags(query)
    poem_order = {
        poem["poem_id"]: index for index, poem in enumerate(poems)
    }

    ranked = []
    for poem in candidates:
        ranked.append(
            _lightweight_result(
                poem,
                _fuse_score(
                    semantic_scores.get(poem["poem_id"], 0.0),
                    _calculate_tag_score(poem, query_tags),
                ),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item["score"],
            poem_order[item["poem_id"]],
        )
    )
    return ranked[:top_k]
