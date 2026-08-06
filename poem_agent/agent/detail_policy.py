"""详情读取的授权、覆盖与 finish 就绪策略。"""

from __future__ import annotations

from ..contracts import TargetDetailChecklistItem
from ..session import AgentSession


def build_target_detail_checklist(
    session: AgentSession,
    activated_details: dict[str, dict],
) -> list[TargetDetailChecklistItem]:
    """生成当前 active 分析 target 的确定性本轮详情覆盖清单。"""
    if not isinstance(session, AgentSession):
        raise TypeError("session 必须是 AgentSession")
    if not isinstance(activated_details, dict):
        raise TypeError("activated_details 必须是对象")
    candidate_pool = session.candidate_pool
    resolved = session.consolidated_request
    if candidate_pool is None or resolved is None:
        return []

    analytical_target_ids = {
        target_id
        for task in resolved.tasks
        if task.type in {"appreciate", "compare"}
        for target_id in task.target_ids
    }
    results_by_target = {
        result["target_id"]: result
        for result in candidate_pool.target_results
    }
    cached_poem_ids = [
        item["poem_id"] for item in session.cached_poems_snapshot()
    ]
    checklist = []
    for target in candidate_pool.targets:
        target_id = target.target_id
        if target_id not in analytical_target_ids:
            continue
        activated_poem_ids = _unique_poem_ids(
            poem_id
            for poem_id, detail in activated_details.items()
            if poem_id not in candidate_pool.failed_candidate_ids
            and target_id in candidate_pool.target_ids_for(poem_id)
            and candidate_pool.is_loaded(poem_id)
            and _detail_can_produce_evidence(detail)
        )
        activatable_cached_poem_ids = _unique_poem_ids(
            poem_id
            for poem_id in cached_poem_ids
            if poem_id not in activated_details
            and poem_id not in candidate_pool.failed_candidate_ids
            and target_id in candidate_pool.target_ids_for(poem_id)
            and _detail_can_produce_evidence(
                session.cached_detail(poem_id) or {}
            )
        )
        target_result = results_by_target.get(target_id, {})
        visible_candidate_poem_ids = _unique_poem_ids(
            poem_id
            for poem_id in target_result.get("visible_candidate_ids", [])
            if poem_id not in candidate_pool.failed_candidate_ids
            and target_id in candidate_pool.target_ids_for(poem_id)
        )
        checklist.append(
            {
                "target_id": target_id,
                "covered": bool(activated_poem_ids),
                "activated_poem_ids": activated_poem_ids,
                "activatable_cached_poem_ids": (
                    activatable_cached_poem_ids
                ),
                "visible_candidate_poem_ids": (
                    visible_candidate_poem_ids
                ),
            }
        )
    return checklist


def finish_readiness_error(
    finish_payload: dict,
    session: AgentSession,
    activated_details: dict[str, dict],
) -> str | None:
    """阻止零本轮详情覆盖的 partial/sufficient 分析提前结束。"""
    assessment = finish_payload["analysis_assessment"]
    if assessment["level"] not in {"sufficient", "partial"}:
        return None
    candidate_pool = session.candidate_pool
    resolved = session.consolidated_request
    if candidate_pool is None or resolved is None:
        return "finish 就绪门槛失败：当前没有已提交的完整请求与 Candidate Pool"

    analytical_target_ids = {
        target_id
        for task in resolved.tasks
        if task.type in {"appreciate", "compare"}
        for target_id in task.target_ids
    }
    assessed_target_ids = set(assessment["target_ids"])
    if analytical_target_ids:
        missing_scope = sorted(analytical_target_ids - assessed_target_ids)
        if missing_scope:
            return (
                "finish 就绪门槛失败：sufficient/partial 的 target_ids 必须覆盖"
                f"当前分析任务 active targets，缺少 target IDs: {missing_scope}"
            )
    else:
        task_types = sorted({task.type for task in resolved.tasks})
        return (
            "finish 就绪门槛失败：当前完整请求仅包含 "
            f"{task_types}，不能申报 sufficient/partial 主体文学分析；"
            "请使用 not_applicable，或先修正完整请求。"
        )

    checklist = build_target_detail_checklist(session, activated_details)
    missing_rows = [
        row
        for row in checklist
        if row["target_id"] in assessed_target_ids and not row["covered"]
    ]
    if missing_rows:
        details = "；".join(
            _format_missing_target_options(row) for row in missing_rows
        )
        return (
            "finish 就绪门槛失败：以下 target 缺少本轮成功取得或显式"
            f"激活且可产生合法 evidence 的详情：{details}"
        )
    return None


def detail_coverage_priority_error(
    poem_id: str,
    session: AgentSession,
    activated_details: dict[str, dict],
) -> str | None:
    """有可补齐 target 时，阻止继续只读取已覆盖 target 的作品。"""
    checklist = build_target_detail_checklist(session, activated_details)
    missing_rows = [
        row
        for row in checklist
        if not row["covered"]
        and (
            row["activatable_cached_poem_ids"]
            or row["visible_candidate_poem_ids"]
        )
    ]
    if not missing_rows:
        return None
    missing_target_ids = {row["target_id"] for row in missing_rows}
    missing_option_ids = {
        poem_id
        for row in missing_rows
        for key in (
            "activatable_cached_poem_ids",
            "visible_candidate_poem_ids",
        )
        for poem_id in row[key]
    }
    candidate_pool = session.candidate_pool
    requested_target_ids = set(
        candidate_pool.target_ids_for(poem_id)
        if candidate_pool is not None
        else []
    )
    if poem_id in missing_option_ids:
        return None
    covered_target_ids = {
        row["target_id"] for row in checklist if row["covered"]
    }
    if not requested_target_ids & covered_target_ids:
        return None
    details = "；".join(
        _format_missing_target_options(row) for row in missing_rows
    )
    return (
        "详情覆盖优先级错误：仍有可补齐但未覆盖的分析 target，"
        f"当前 poem_id={poem_id!r} 不能补齐这些 target。"
        f"未覆盖 target IDs={sorted(missing_target_ids)}；{details}。"
        "下一次 get_poem_detail 必须优先选择上述缓存或 visible ID；"
        "缓存 ID 即使不在 visible 中也可直接激活。"
    )


def _unique_poem_ids(poem_ids) -> list[str]:
    return list(dict.fromkeys(poem_ids))


def _format_missing_target_options(row: dict) -> str:
    suffix = (
        ""
        if (
            row["activatable_cached_poem_ids"]
            or row["visible_candidate_poem_ids"]
        )
        else "；当前不可补齐"
    )
    return (
        f"target {row['target_id']} 缺少本轮详情；"
        f"可激活缓存 IDs={row['activatable_cached_poem_ids']}；"
        f"可读取 visible IDs={row['visible_candidate_poem_ids']}"
        f"{suffix}"
    )


def _detail_can_produce_evidence(detail: dict) -> bool:
    return (
        isinstance(detail, dict)
        and (
            bool(detail.get("appreciation"))
            or bool(detail.get("annotations"))
        )
    )
