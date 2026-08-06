"""Candidate Pool 的查询编译、检索状态与固定 verdict。"""

from __future__ import annotations

from .. import store
from .models import THEME_SEPARATOR, Target


def _main_query(target: Target) -> dict:
    return {
        "query": THEME_SEPARATOR.join(target.themes) or None,
        "author": target.author,
        "dynasty": target.dynasty,
        "title": target.title,
    }


def _diagnostic_query(target: Target) -> dict | None:
    query = THEME_SEPARATOR.join(target.themes) or None
    if target.title is not None and (
        target.author is not None or target.dynasty is not None
    ):
        return {
            "query": query, "author": None, "dynasty": None,
            "title": target.title,
        }
    if (
        target.title is None
        and target.author is not None
        and target.dynasty is not None
    ):
        return {
            "query": query, "author": target.author,
            "dynasty": None, "title": None,
        }
    return None


def _target_status(
    target: Target,
    main_ids: list[str],
    diagnostic_ids: list[str],
    candidates: dict[str, dict],
) -> tuple[str, dict | None]:
    if not any((target.author, target.dynasty, target.title)):
        return "not_applicable", None
    if main_ids:
        strict_ids = [
            poem_id for poem_id in main_ids
            if _strictly_matches(target, candidates[poem_id])
        ]
        if strict_ids:
            return "matched", None
        if target.title is not None:
            return "partial_match", {
                "requested_title": target.title,
                "partial_title_candidate_ids": list(main_ids),
            }
        return "missing", None
    if diagnostic_ids:
        conflicts = []
        for poem_id in diagnostic_ids:
            item = candidates[poem_id]
            differences = {}
            for name in ("author", "dynasty"):
                expected = getattr(target, name)
                actual = item.get(name)
                if expected is not None and actual != expected:
                    differences[name] = {
                        "expected": expected, "actual": actual
                    }
            conflicts.append({"poem_id": poem_id, "differences": differences})
        return "conflict", {"diagnostic_conflicts": conflicts}
    return "missing", None


def _strictly_matches(target: Target, candidate: dict) -> bool:
    return (
        (target.author is None or candidate.get("author") == target.author)
        and (target.dynasty is None or candidate.get("dynasty") == target.dynasty)
        and (
            target.title is None
            or (
                isinstance(candidate.get("title"), str)
                and store._normalize_title(candidate["title"]) == target.title
            )
        )
    )


def _build_verdict(target_results: list[dict]) -> str:
    statuses = [item["status"] for item in target_results]
    if "conflict" in statuses:
        return "请求不符：条件诊断发现一个或多个 target 的作者或朝代与语料不一致。"
    if "partial_match" in statuses and "missing" in statuses:
        return "部分满足：存在标题部分匹配 target，另有 target 未命中。"
    if "partial_match" in statuses:
        return "标题部分匹配：至少一个 target 仅找到标题部分匹配候选。"
    if all(status == "missing" for status in statuses):
        return "未命中：所有 target 在允许的检索与诊断路径中均无结果。"
    if "missing" in statuses:
        return "部分满足：部分 target 已命中，另有 target 未命中。"
    if all(status == "not_applicable" for status in statuses):
        if any(item["retrieval"] == "found" for item in target_results):
            return "已取得主题排序候选，但主题覆盖待评估。"
        return "未命中：主题检索未取得候选，主题覆盖仍待评估。"
    if "not_applicable" in statuses:
        return "部分满足：结构化 target 已命中，主题覆盖仍待评估。"
    return "全部命中：所有 target 的结构化条件均得到严格匹配。"

