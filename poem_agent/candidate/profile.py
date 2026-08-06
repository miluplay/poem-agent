"""Candidate Pool 的纯参考量画像与结论计算。"""

from __future__ import annotations

from statistics import mean, median


def _poem_dimension(count: int, threshold: int) -> dict:
    return {
        "count": count,
        "label": "sufficient" if count >= threshold else "limited",
    }


def _summary(values: list[int]) -> dict:
    if not values:
        return {"total": 0, "min": None, "max": None, "median": None, "mean": None}
    return {
        "total": sum(values),
        "min": min(values),
        "max": max(values),
        "median": median(values),
        "mean": mean(values),
    }


def _aggregate_poem_dimension(rows: list[dict], dimension: str) -> dict:
    values = [row[dimension]["count"] for row in rows]
    sufficient_ids = [
        row["poem_id"]
        for row in rows
        if row[dimension]["label"] == "sufficient"
    ]
    limited_ids = [
        row["poem_id"]
        for row in rows
        if row[dimension]["label"] == "limited"
    ]
    count = len(rows)
    ratio = len(sufficient_ids) / count if count else None
    return {
        **_summary(values),
        "sufficient_count": len(sufficient_ids),
        "limited_count": len(limited_ids),
        "sufficient_ratio": ratio,
        "limited_poem_ids": limited_ids,
        "label": (
            "not_evaluated"
            if ratio is None
            else "sufficient" if ratio > 0.6 else "limited"
        ),
    }


def _overall_dimension(
    poem_rows: list[dict], target_rows: list[dict], dimension: str
) -> dict:
    evaluated = [
        row for row in target_rows
        if row[dimension]["label"] != "not_evaluated"
    ]
    sufficient = [
        row["target_id"]
        for row in evaluated
        if row[dimension]["label"] == "sufficient"
    ]
    limited = [
        row["target_id"]
        for row in evaluated
        if row[dimension]["label"] == "limited"
    ]
    ratio = len(sufficient) / len(evaluated) if evaluated else None
    return {
        **_summary([row[dimension]["count"] for row in poem_rows]),
        "sufficient_target_count": len(sufficient),
        "limited_target_count": len(limited),
        "sufficient_target_ratio": ratio,
        "limited_target_ids": limited,
        "label": (
            "not_evaluated"
            if ratio is None
            else "sufficient" if ratio > 0.6 else "limited"
        ),
    }


def _build_reference_verdict(
    coverage: dict, overall: dict, has_unavailable_detail: bool
) -> str:
    if overall["poem_count"] == 0:
        if has_unavailable_detail:
            return "候选详情不可用，参考量无法评估。"
        return "尚未读取作品详情，参考量未评估。"
    appr = overall["appreciation"]["label"]
    anno = overall["annotations"]["label"]
    description = {
        ("sufficient", "sufficient"): "赏析与注释参考量均充足",
        ("sufficient", "limited"): "赏析参考量充足，注释参考量较少",
        ("limited", "sufficient"): "赏析参考量较少，注释参考量充足",
        ("limited", "limited"): "赏析与注释参考量均较少",
    }[(appr, anno)]
    if (
        coverage["status"] == "all_covered"
        and not has_unavailable_detail
    ):
        return f"全部可用 targets 已取得详情；{description}。"
    suffix = (
        "；另有候选详情不可用"
        if has_unavailable_detail
        else ""
    )
    if coverage["status"] == "all_covered":
        return (
            f"全部剩余可用 targets 已取得详情；"
            f"{description}{suffix}。"
        )
    return f"仅部分可用 targets 已取得详情；已读部分{description}{suffix}。"



