"""混合检索算法实现。"""

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
_MIN_SUBSTRING_TITLE_LENGTH = 2


def _build_vocabularies() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """从当前诗库动态生成作者、朝代和标签词表。"""
    authors: set[str] = set()
    dynasties: set[str] = set()
    tags: set[str] = set()

    for poem in store.load_poems():
        author = poem.get("author")
        dynasty = poem.get("dynasty")
        if isinstance(author, str) and author:
            authors.add(author)
        if isinstance(dynasty, str) and dynasty:
            dynasties.add(dynasty)
        for tag in poem.get("tags", []):
            if isinstance(tag, str) and tag:
                tags.add(tag)

    return frozenset(authors), frozenset(dynasties), frozenset(tags)


# 词表随进程启动从 poems.json 生成；新增数据后重启即可自动纳入。
_AUTHORS, _DYNASTIES, _TAGS = _build_vocabularies()


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


def _extract_query_terms(
    query: str,
) -> tuple[set[str], set[str], set[str]]:
    """用子串包含匹配识别查询中的作者、朝代和标签。"""
    return (
        {author for author in _AUTHORS if author in query},
        {dynasty for dynasty in _DYNASTIES if dynasty in query},
        {tag for tag in _TAGS if tag in query},
    )


def _passes_hard_filters(
    poem: dict, query_authors: set[str], query_dynasties: set[str]
) -> bool:
    """作者和朝代一旦在查询中出现，就作为硬过滤条件。"""
    return (
        (not query_authors or poem.get("author") in query_authors)
        and (not query_dynasties or poem.get("dynasty") in query_dynasties)
    )


def _calculate_tag_score(poem: dict, query_tags: set[str]) -> float:
    """计算标签软分：诗命中的查询标签数 / 查询中出现的标签总数。"""
    if not query_tags:
        return 0.0
    poem_tags = {tag for tag in poem.get("tags", []) if isinstance(tag, str)}
    return len(poem_tags & query_tags) / len(query_tags)


def _fuse_score(semantic_score: float, tag_score: float) -> float:
    """融合语义分与标签分，权重集中在这里便于后续调参。"""
    return 0.6 * semantic_score + 0.4 * tag_score


def _query_collection(collection_name: str, query_embedding: list[float]) -> dict:
    """查询单个 collection，空 collection 直接返回空结果。"""
    collection = _get_chroma_client().get_collection(name=collection_name)
    result_count = min(_SEMANTIC_TOP_N, collection.count())
    if result_count == 0:
        return {"metadatas": [[]], "distances": [[]]}
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=result_count,
        include=["metadatas", "distances"],
    )


def _semantic_scores(query: str) -> dict[str, float]:
    """分别检索正文和赏析，并按 poem_id 对两路余弦相似度取最大值。"""
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
        result = _query_collection(collection_name, query_embedding)
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


def _is_exact_title_match(
    normalized_title: str, normalized_query: str
) -> bool:
    """判断已归一化的标题与查询是否完全相等。"""
    return bool(normalized_query) and normalized_title == normalized_query


def _is_partial_title_match(
    normalized_title: str, normalized_query: str
) -> bool:
    """判断已归一化的标题是否包含查询，但排除完全相等。"""
    return (
        bool(normalized_query)
        and normalized_title != normalized_query
        and normalized_query in normalized_title
    )


def _title_matches(query: str) -> tuple[list[dict], list[dict]]:
    """查找完整检索使用的精确/部分标题命中，复用标题归一化规则。"""
    normalized_query = store._normalize_title(query)
    exact_matches: list[dict] = []
    partial_matches: list[dict] = []

    for poem in store.load_poems():
        title = poem.get("title")
        if not isinstance(title, str):
            continue
        normalized_title = store._normalize_title(title)
        if _is_exact_title_match(normalized_title, normalized_query):
            exact_matches.append(poem)
        elif _is_partial_title_match(normalized_title, normalized_query):
            partial_matches.append(poem)

    return exact_matches, partial_matches


def _is_content_title_match(
    normalized_title: str, normalized_query: str
) -> bool:
    """判断标题能否作为 query 中的内容线索，单字标题一律排除。"""
    return (
        len(normalized_title) >= _MIN_SUBSTRING_TITLE_LENGTH
        and normalized_title != normalized_query
        and normalized_title in normalized_query
    )


def _content_title_matches(
    query: str, query_authors: set[str]
) -> list[dict]:
    """用 query 中的标题子串定位诗，并用已识别作者核对候选。"""
    normalized_query = store._normalize_title(query)
    matches: list[dict] = []

    for poem in store.load_poems():
        title = poem.get("title")
        if not isinstance(title, str):
            continue
        normalized_title = store._normalize_title(title)
        if _is_content_title_match(normalized_title, normalized_query):
            matches.append(poem)

    # 查询提到作者时始终核对，避免唯一标题也因作者不符而误短路。
    if query_authors:
        matches = [
            poem for poem in matches if poem.get("author") in query_authors
        ]
    return matches


def _title_shortcut_result(poem: dict) -> list[dict]:
    """构造唯一标题定位时的短路返回值。"""
    return [
        {
            "poem_id": poem["poem_id"],
            "title": poem["title"],
            "author": poem["author"],
            "dynasty": poem["dynasty"],
            "score": 1.0,
            "matched_by": "title",
        }
    ]


def hybrid_search(query: str, top_k: int) -> list[dict]:
    """融合标题、正文/赏析语义和标签信号检索诗文。"""
    exact_title_matches, partial_title_matches = _title_matches(query)
    if len(exact_title_matches) == 1:
        return _title_shortcut_result(exact_title_matches[0])

    poems_by_id = {poem["poem_id"]: poem for poem in store.load_poems()}
    query_authors, query_dynasties, query_tags = _extract_query_terms(query)
    content_title_matches = _content_title_matches(query, query_authors)
    if len(content_title_matches) == 1:
        return _title_shortcut_result(content_title_matches[0])

    exact_title_ids = {
        poem["poem_id"] for poem in exact_title_matches
    }
    partial_title_ids = {
        poem["poem_id"] for poem in partial_title_matches
    }
    title_ids = exact_title_ids | partial_title_ids
    semantic_scores = _semantic_scores(query)

    candidates: list[dict] = []
    candidate_ids = title_ids | set(semantic_scores)
    for poem_id in candidate_ids:
        poem = poems_by_id.get(poem_id)
        if poem is None or not _passes_hard_filters(
            poem, query_authors, query_dynasties
        ):
            continue

        is_exact_title_match = poem_id in exact_title_ids
        is_partial_title_match = poem_id in partial_title_ids
        is_title_match = is_exact_title_match or is_partial_title_match
        semantic_score = 1.0 if is_title_match else semantic_scores[poem_id]
        tag_score = _calculate_tag_score(poem, query_tags)
        candidates.append(
            {
                "poem_id": poem_id,
                "title": poem["title"],
                "author": poem["author"],
                "dynasty": poem["dynasty"],
                "score": _fuse_score(semantic_score, tag_score),
                "matched_by": (
                    "title"
                    if is_title_match
                    else "tag"
                    if tag_score > 0
                    else "semantic"
                ),
                "_title_match_rank": (
                    0
                    if is_exact_title_match
                    else 1
                    if is_partial_title_match
                    else 2
                ),
            }
        )

    # 精确标题、部分标题、语义结果依次置顶；层内沿用原有次级排序。
    poem_order = {
        poem["poem_id"]: index for index, poem in enumerate(store.load_poems())
    }
    candidates.sort(
        key=lambda item: (
            item["_title_match_rank"],
            -item["score"],
            poem_order[item["poem_id"]],
        )
    )
    for item in candidates:
        del item["_title_match_rank"]
    return candidates[:top_k]
