"""诗文数据的加载与基础查询。"""

import json
import math
from functools import lru_cache
from pathlib import Path
import re
import unicodedata


_DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "poems.json"


@lru_cache(maxsize=1)
def load_poems(path: str | Path = _DEFAULT_DATA_PATH) -> list[dict]:
    """读取 913 首,返回 list[PoemDetail]。"""
    with Path(path).open(encoding="utf-8") as file:
        poems = json.load(file)
    if not isinstance(poems, list):
        raise ValueError(f"诗文数据应为 JSON 数组，实际为 {type(poems).__name__}")
    return poems


def _normalize_title(title: str) -> str:
    """统一标题的 Unicode 形式，并忽略空白和最外层书名号。"""
    normalized = unicodedata.normalize("NFKC", title)
    normalized = re.sub(r"\s+", "", normalized)
    while (
        len(normalized) >= 2
        and (normalized[0], normalized[-1]) in {("《", "》"), ("〈", "〉")}
    ):
        normalized = normalized[1:-1]
    return normalized


def find_by_title(title: str) -> dict | None:
    """按标题匹配一首诗，忽略书名号及标题中的空白。"""
    if not isinstance(title, str):
        return None
    target = _normalize_title(title)
    if not target:
        return None
    return next(
        (
            poem
            for poem in load_poems()
            if isinstance(poem.get("title"), str)
            and _normalize_title(poem["title"]) == target
        ),
        None,
    )


def get_by_id(poem_id: str) -> dict | None:
    """按 poem_id 取整首。"""
    if not isinstance(poem_id, str) or not poem_id:
        return None
    return next(
        (poem for poem in load_poems() if poem.get("poem_id") == poem_id),
        None,
    )


def reference_count_baseline(poems: list[dict] | None = None) -> dict:
    """按全语料 nearest-rank 下四分位数计算两类参考量基线。"""
    corpus = load_poems() if poems is None else poems
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("参考量基线需要非空诗文列表")

    appreciation_counts: list[int] = []
    annotation_counts: list[int] = []
    for index, poem in enumerate(corpus):
        if not isinstance(poem, dict):
            raise ValueError(f"诗文[{index}] 必须是对象")
        appreciation = poem.get("appreciation")
        annotations = poem.get("annotations")
        if not isinstance(appreciation, list) or not isinstance(
            annotations, list
        ):
            raise ValueError(
                f"诗文[{index}] 的 appreciation/annotations 必须是列表"
            )
        if any(not isinstance(item, dict) for item in appreciation) or any(
            not isinstance(item, dict) for item in annotations
        ):
            raise ValueError(
                f"诗文[{index}] 的 appreciation/annotations 条目必须是对象"
            )
        appreciation_counts.append(len(appreciation))
        annotation_counts.append(len(annotations))

    rank = math.ceil(0.25 * len(corpus))
    return {
        "method": "corpus_lower_quartile",
        "appreciation_threshold": sorted(appreciation_counts)[rank - 1],
        "annotation_threshold": sorted(annotation_counts)[rank - 1],
        "aggregate_ratio_threshold": 0.6,
        "comparison": "strictly_greater",
    }
