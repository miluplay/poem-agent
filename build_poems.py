#!/usr/bin/env python3
"""筛选 chinese-gushiwen 数据并生成统一的 PoemDetail 列表。"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INPUT_DIR = Path("data/guwen")
OUTPUT_PATH = Path("data/poems.json")
MAX_APPRECIATION_PARAGRAPH_LENGTH = 500
FEATURED_TAGS = {
    "唐诗三百首",
    "宋词三百首",
    "古文观止",
    "宋词精选",
    "初中文言文",
    "高中文言文",
    "初中古诗",
    "高中古诗",
    "小学古诗",
}
PROSE_TAGS = {"古文观止", "初中文言文", "高中文言文", "文言文"}


def nonempty(value: Any) -> bool:
    """字符串去除首尾空白后是否非空。"""
    return isinstance(value, str) and bool(value.strip())


def load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    """读取文件中首尾相接的多个 JSON 对象（源文件不是 JSON 数组）。"""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        match = re.search(r"\S", text[position:])
        if not match:
            break
        position += match.start()
        value, position = decoder.raw_decode(text, position)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: 期待 JSON 对象，实际为 {type(value).__name__}")
        records.append(value)
    return records


def passes_hard_filter(poem: dict[str, Any]) -> bool:
    """任务定义的四项硬筛选条件。"""
    content = poem.get("content")
    shangxi = poem.get("shangxi")
    return (
        nonempty(content)
        and len(content.strip()) >= 4
        and nonempty(shangxi)
        and len(shangxi.strip()) >= 50
        and nonempty(poem.get("translation"))
        and all(nonempty(poem.get(field)) for field in ("title", "writer", "dynasty"))
    )


def priority_score(poem: dict[str, Any]) -> int:
    """计算精选标签、注释和标签丰富度得分。"""
    tags = poem.get("type") if isinstance(poem.get("type"), list) else []
    score = 3 if FEATURED_TAGS.intersection(tags) else 0
    score += int(nonempty(poem.get("remark")))
    score += int(len(tags) >= 3)
    return score


def rank_key(poem: dict[str, Any]) -> tuple[int, int]:
    """得分降序，得分相同时赏析长度降序。"""
    return priority_score(poem), len(poem["shangxi"])


def split_annotations(remark: Any, poem_id: str) -> list[dict[str, str]]:
    if not nonempty(remark):
        return []
    lines = [line.strip() for line in remark.splitlines() if line.strip()]
    return [
        {"evidence_id": f"{poem_id}#anno-{index}", "text": text}
        for index, text in enumerate(lines)
    ]


def split_appreciation(shangxi: str, poem_id: str) -> list[dict[str, str]]:
    """按空行分段，同时兼容空行中含全角或半角空白。"""
    paragraphs = [
        paragraph.strip(" \t\r\n\u3000")
        for paragraph in re.split(r"\n[ \t\u3000]*\n+", shangxi)
    ]
    return [
        {"evidence_id": f"{poem_id}#appr-{index}", "text": text}
        for index, text in enumerate(paragraph for paragraph in paragraphs if paragraph)
    ]


def transform(poem: dict[str, Any]) -> dict[str, Any]:
    poem_id = poem["_id"]["$oid"]
    tags = poem.get("type") if isinstance(poem.get("type"), list) else []
    return {
        "poem_id": poem_id,
        "title": poem["title"],
        "author": poem["writer"],
        "dynasty": poem["dynasty"],
        "content": poem["content"],
        "annotations": split_annotations(poem.get("remark"), poem_id),
        "translation": poem["translation"],
        "appreciation": split_appreciation(poem["shangxi"], poem_id),
        "tags": tags,
        "audio_url": poem.get("audioUrl") or "",
        "source": "chinese-gushiwen",
    }


def passes_quality_filter(poem: dict[str, Any]) -> bool:
    """排除无注释、无标签或存在超长赏析自然段的作品。"""
    return (
        bool(poem["annotations"])
        and bool(poem["tags"])
        and all(
            len(paragraph["text"]) <= MAX_APPRECIATION_PARAGRAPH_LENGTH
            for paragraph in poem["appreciation"]
        )
    )


def distribution(values: Iterable[int]) -> str:
    numbers = list(values)
    if not numbers:
        return "无数据"
    return (
        f"最小 {min(numbers)} / 中位 {statistics.median(numbers):g} / "
        f"最大 {max(numbers)} / 平均 {statistics.mean(numbers):.2f}"
    )


def genre_coverage(poems: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "唐诗": sum(p["dynasty"].startswith("唐") for p in poems),
        "宋词": sum(
            p["dynasty"].startswith("宋")
            and any("词" in tag for tag in p["tags"])
            for p in poems
        ),
        "文言文": sum(bool(PROSE_TAGS.intersection(p["tags"])) for p in poems),
    }


def suspicious_content(content: str) -> bool:
    """标记替换字符、NUL、同一非空白字符连续四次等常见脏数据。"""
    return bool(re.search(r"[\ufffd\x00]|(\S)\1{3,}", content))


def print_report(
    total: int,
    eligible: int,
    overlong: int,
    without_annotations: int,
    without_tags: int,
    poems: list[dict[str, Any]],
) -> None:
    dynasty_counts = Counter(poem["dynasty"] for poem in poems)
    with_remark = sum(bool(poem["annotations"]) for poem in poems)
    paragraph_counts = [len(poem["appreciation"]) for poem in poems]
    paragraph_lengths = [
        len(paragraph["text"])
        for poem in poems
        for paragraph in poem["appreciation"]
    ]
    annotation_counts = [len(poem["annotations"]) for poem in poems]
    suspicious = [
        poem["title"]
        for poem in poems
        if any(len(p["text"]) > 500 for p in poem["appreciation"])
        or suspicious_content(poem["content"])
    ]

    print("=== 数据门禁报告 ===")
    print(f"源数据总数: {total}")
    print(f"通过硬筛选: {eligible}")
    print(f"剔除赏析单段超过 {MAX_APPRECIATION_PARAGRAPH_LENGTH} 字: {overlong}")
    print(f"剔除无注释: {without_annotations}")
    print(f"剔除无标签: {without_tags}")
    print(f"最终入选: {len(poems)}")
    print("体裁覆盖: " + "，".join(f"{k} {v}" for k, v in genre_coverage(poems).items()))
    print("朝代分布: " + "，".join(f"{k} {v}" for k, v in dynasty_counts.most_common()))
    ratio = with_remark / len(poems) if poems else 0
    print(f"有 remark: {with_remark}/{len(poems)} ({ratio:.2%})")
    print("appreciation 段数: " + distribution(paragraph_counts))
    print("appreciation 单段字符数: " + distribution(paragraph_lengths))
    print("annotations 条数: " + distribution(annotation_counts))
    print(f"疑似质量问题: {len(suspicious)} 首")
    print("标题: " + ("、".join(suspicious) if suspicious else "无"))


def main() -> None:
    paths = sorted(INPUT_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"未找到输入文件: {INPUT_DIR}/*.json")
    raw = [record for path in paths for record in load_concatenated_json(path)]
    eligible = [poem for poem in raw if passes_hard_filter(poem)]
    mapped = [transform(poem) for poem in eligible]
    overlong = sum(
        any(
            len(paragraph["text"]) > MAX_APPRECIATION_PARAGRAPH_LENGTH
            for paragraph in poem["appreciation"]
        )
        for poem in mapped
    )
    without_annotations = sum(not poem["annotations"] for poem in mapped)
    without_tags = sum(not poem["tags"] for poem in mapped)
    selected = [
        raw_poem
        for raw_poem, mapped_poem in zip(eligible, mapped)
        if passes_quality_filter(mapped_poem)
    ]
    poems = [transform(poem) for poem in sorted(selected, key=rank_key, reverse=True)]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(poems, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report(
        len(raw),
        len(eligible),
        overlong,
        without_annotations,
        without_tags,
        poems,
    )


if __name__ == "__main__":
    main()
