"""手写 agent 循环。第一条纵线:一步 finish 版。
★ 这是你要亲手写透的核心,面试会被追问每个环节。"""

import json
import os
import re

from ..analysis_support import (
    AnalysisAssessmentProtocolError,
    append_required_support_notice,
    evaluate_analysis_support,
    force_fallback_analysis_support,
    validate_analysis_assessment,
)
from ..candidate_pool import CandidatePool
from ..request import (
    ConsolidatedRequestProtocolError,
    normalize_consolidated_request,
    request_semantics_error,
    render_resolved_request,
)
from ..session import AgentSession
from ..tools import TOOLS
from ..trust import (
    _append_degraded_notice,
    _strip_invalid_citation_markers,
    answer_integrity_fallback,
    collect_evidence,
    is_answer_suspiciously_incomplete,
    list_dangling_citations,
)
from .display import (
    _print_final_separator,
    _print_observation,
    _print_step_decision,
    _print_step_error,
)
from .observation import (
    _session_poem_number,
    _summarize_observation,
)
from .prompts import (
    SYSTEM_INSTRUCTION,
    build_force_finish_prompt,
    build_prompt,
)


DETAILS_PER_ACTIVE_TARGET = 2
MAX_FRAMEWORK_STEPS = 2
MAX_RECOVERY_STEPS = 2
# v0.6 兼容常量；不再驱动循环。
MAX_PRODUCTIVE_STEPS = 6
MAX_TOTAL_STEPS = 8
MAX_STEPS = MAX_PRODUCTIVE_STEPS

REPEATED_ACTION_OBSERVATION = (
    "你刚才已经用相同参数执行过 {action},结果没有变化。请不要重复,"
    "请基于 Candidate Pool 和已有详情继续或结束作答。"
)
STALLED_ACTION_OBSERVATION = (
    "你已多次调用工具但未获得有效结果,请停止调用,"
    "基于 Candidate Pool 和已有信息作答或告知用户找不到。"
)
_DETAIL_ACTION_FIELDS = frozenset(
    {"poem_id", "target_id", "target_ids", "fields"}
)


def build_target_detail_checklist(
    session: AgentSession,
    activated_details: dict[str, dict],
) -> list[dict]:
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


def _unique_poem_ids(poem_ids) -> list[str]:
    return list(dict.fromkeys(poem_ids))


def normalize_detail_action_input(action_input) -> str:
    """只提取 canonical poem_id；已知上下文字段不参与任何授权。"""
    if not isinstance(action_input, dict):
        raise ValueError("action_input 必须是对象")
    unknown = set(action_input) - _DETAIL_ACTION_FIELDS
    if unknown:
        raise ValueError(
            "包含未知字段: " + "、".join(sorted(unknown))
        )
    if "poem_id" not in action_input:
        raise ValueError("缺少 poem_id")
    poem_id = action_input["poem_id"]
    if not isinstance(poem_id, str) or not poem_id.strip():
        raise ValueError("poem_id 必须是非空字符串")
    return poem_id.strip()


def run_agent(
    user_query: str,
    llm,
    verbose: bool = False,
    *,
    session: AgentSession | None = None,
) -> dict:
    """运行一个用户轮次；显式 Session 原地复用，省略时使用临时会话。"""
    if session is not None and not isinstance(session, AgentSession):
        raise TypeError("session 必须是 AgentSession 或 None")
    current_session = session if session is not None else AgentSession()
    verbose = verbose or _env_flag_enabled("POEM_AGENT_VERBOSE")
    trajectory: list[dict] = []
    activated_details: dict[str, dict] = {}
    detail_steps = 0
    framework_steps = 0
    recovery_steps = 0
    total_steps = 0
    request_action_succeeded = False

    while recovery_steps < MAX_RECOVERY_STEPS:
        candidate_pool = current_session.candidate_pool
        active_count = len(candidate_pool.targets) if candidate_pool else 0
        detail_limit = active_count * DETAILS_PER_ACTIVE_TARGET
        total_limit = detail_limit + MAX_FRAMEWORK_STEPS + MAX_RECOVERY_STEPS
        if total_steps >= total_limit:
            break
        prompt = _build_session_prompt(
            user_query,
            trajectory,
            current_session,
            activated_details,
            request_phase_complete=request_action_succeeded,
        )
        decision = parse_decision(llm.generate(prompt))
        total_steps += 1

        if decision is None:
            error = "格式非法,请返回 JSON:{thought, action, action_input}"
            trajectory.append({"error": error})
            recovery_steps += 1
            if verbose:
                _print_step_error(total_steps, error)
            continue

        if verbose:
            _print_step_decision(total_steps, decision)

        if decision["action"] == "finish":
            try:
                finish_payload = validate_analysis_assessment(
                    decision["action_input"], current_session.candidate_pool
                )
            except AnalysisAssessmentProtocolError as exc:
                error = f"finish 协议错误: {exc}"
                trajectory.append({"error": error})
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{error}")
                continue
            readiness_error = _finish_readiness_error(
                finish_payload,
                current_session,
                activated_details,
            )
            if readiness_error is not None:
                candidate_pool = current_session.candidate_pool
                observation = {
                    "error": readiness_error,
                    "visible_candidate_ids": (
                        candidate_pool.visible_candidate_ids()
                        if candidate_pool is not None
                        else []
                    ),
                    "activatable_cached_poem_ids": [
                        item["poem_id"]
                        for item in current_session.cached_poems_snapshot()
                    ],
                }
                trajectory.append(
                    {
                        "thought": decision.get("thought", ""),
                        "action": "finish",
                        "input": decision["action_input"],
                        "observation": observation,
                    }
                )
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{readiness_error}")
                continue
            framework_steps += 1
            if verbose:
                _print_final_separator()
            regenerate = lambda feedback: _extract_finish_payload(
                llm.generate(_append_regeneration_feedback(prompt, feedback)),
                current_session.candidate_pool,
            )
            result = _unified_final_check(
                finish_payload,
                None,
                regenerate,
                trajectory,
                current_session.session_poems_snapshot(),
                current_session.candidate_pool,
                cached_details=activated_details,
                verbose=verbose,
            )
            return _finalize_round(user_query, result, current_session)

        if decision["action"] in {
            "initialize_candidate_pool",
            "update_candidate_pool",
        }:
            action = decision["action"]
            error = None
            if request_action_succeeded:
                observation = _closed_request_phase_observation(
                    current_session
                )
                trajectory.append(
                    {
                        "thought": decision.get("thought", ""),
                        "action": action,
                        "input": decision["action_input"],
                        "observation": observation,
                    }
                )
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{observation['error']}")
                continue
            elif action == "initialize_candidate_pool" and current_session.candidate_pool:
                error = "Candidate Pool 已存在，必须使用 update_candidate_pool"
            elif action == "update_candidate_pool" and current_session.candidate_pool is None:
                error = "Candidate Pool 尚未初始化，不能 update_candidate_pool"
            try:
                request = (
                    normalize_consolidated_request(decision["action_input"])
                    if error is None
                    else None
                )
            except ConsolidatedRequestProtocolError as exc:
                error = f"Candidate Pool 协议错误: 完整请求协议错误: {exc}"
            if error is None:
                error = request_semantics_error(
                    user_query,
                    current_session.consolidated_request,
                    request,
                )
            if error is not None:
                trajectory.append({"error": error})
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{error}")
                continue

            if action == "initialize_candidate_pool":
                current_session.initialize_request(request)
            else:
                current_session.update_request(request)
            request_action_succeeded = True
            framework_steps += 1
            observation = {
                "resolved_request": current_session.request_snapshot(),
                "candidate_pool": current_session.candidate_pool.model_snapshot(),
            }
            trajectory.append(
                {
                    "thought": decision.get("thought", ""),
                    "action": action,
                    "input": decision["action_input"],
                    "observation": observation,
                }
            )
            if verbose:
                _print_observation("完整请求与 Candidate Pool 已原子提交")
            continue

        if decision["action"] != "get_poem_detail":
            error = f"未知工具 {decision['action']}"
            trajectory.append({"error": error})
            recovery_steps += 1
            if verbose:
                _print_observation(f"错误：{error}")
            continue

        action_input = decision["action_input"]
        poem_id = action_input.get("poem_id")
        error = None
        candidate_pool = current_session.candidate_pool
        legal_ids = (
            candidate_pool.visible_candidate_ids()
            if candidate_pool is not None
            else []
        )
        if candidate_pool is None:
            error = "Candidate Pool 尚未初始化，不能读取详情"
        else:
            try:
                poem_id = normalize_detail_action_input(action_input)
            except ValueError as exc:
                error = f"详情协议错误: {exc}"
        if error is None and poem_id in activated_details:
            error = "该作品详情已在本轮成功观察，不能重复调用"
        elif error is None and detail_steps >= detail_limit:
            error = f"本轮详情额度已耗尽（上限 {detail_limit}）"
        elif (
            error is None
            and poem_id in candidate_pool.failed_candidate_ids
        ):
            error = "该作品详情此前连续失败，已被隔离"
        cached_detail = (
            current_session.cached_detail(poem_id)
            if error is None and isinstance(poem_id, str)
            else None
        )
        if error is None and cached_detail is None and poem_id not in legal_ids:
            error = "详情协议错误: poem_id 不属于当前可见未读窗口"
        if error is None:
            error = _detail_coverage_priority_error(
                poem_id,
                current_session,
                activated_details,
            )

        if error is not None:
            target_detail_checklist = build_target_detail_checklist(
                current_session, activated_details
            )
            observation = {
                "error": error,
                "poem_id": poem_id,
                "visible_candidate_ids": legal_ids,
                "activatable_cached_poem_ids": [
                    item["poem_id"]
                    for item in current_session.cached_poems_snapshot()
                ],
                "target_detail_checklist": target_detail_checklist,
            }
            trajectory.append(
                {
                    "thought": decision.get("thought", ""),
                    "action": decision["action"],
                    "input": action_input,
                    "observation": observation,
                }
            )
            recovery_steps += 1
            if verbose:
                _print_observation(
                    f"错误：{error}；当前新候选 visible IDs: {legal_ids}；"
                    "缓存命中不要求 visible，当前可激活缓存 IDs: "
                    f"{observation['activatable_cached_poem_ids']}"
                )
            continue

        if cached_detail is not None:
            observation = cached_detail
        else:
            tool = TOOLS["get_poem_detail"]
            automatically_retried = False
            try:
                observation = tool(poem_id=poem_id)
            except Exception:
                automatically_retried = True
                observation = tool(poem_id=poem_id)
            if (
                not automatically_retried
                and isinstance(observation, dict)
                and observation.get("error") == "not_found"
            ):
                observation = tool(poem_id=poem_id)

        if (
            isinstance(observation, dict)
            and observation.get("error") == "not_found"
        ):
            recovery = candidate_pool.recover_failed_detail(poem_id)
            observation = {
                "error": "not_found_after_retry",
                "poem_id": poem_id,
                "recovery": recovery,
                "visible_candidate_ids": (
                    candidate_pool.visible_candidate_ids()
                ),
            }
            recovery_steps += 1
        elif (
            not isinstance(observation, dict)
            or "error" in observation
            or observation.get("poem_id") != poem_id
        ):
            # 非契约错误不伪装成 not_found 或正常进展。
            raise RuntimeError("get_poem_detail 返回了非法结果")
        else:
            if cached_detail is None:
                current_session.add_detail(observation)
            activated_details[poem_id] = observation
            detail_steps += 1

        trajectory_step = {
            "thought": decision.get("thought", ""),
            "action": decision["action"],
            "input": decision["action_input"],
            "observation": observation,
        }
        trajectory.append(trajectory_step)
        if verbose:
            _print_observation(
                _summarize_observation(
                    observation,
                    session_poems=current_session.session_poems_snapshot(),
                    concise=True,
                )
            )

    result = force_finish(
        user_query,
        llm,
        trajectory,
        current_session.session_poems_snapshot(),
        candidate_pool=current_session.candidate_pool,
        session=current_session,
        cached_details=activated_details,
        request_phase_complete=request_action_succeeded,
        verbose=verbose,
    )
    return _finalize_round(user_query, result, current_session)


def _build_session_prompt(
    user_query: str,
    trajectory: list,
    session: AgentSession,
    activated_details: dict[str, dict] | None = None,
    *,
    request_phase_complete: bool = False,
) -> str:
    resolved = session.consolidated_request
    history = _history_for_prompt(session)
    return build_prompt(
        user_query,
        trajectory,
        session.session_poems_snapshot(),
        candidate_pool=(
            session.candidate_pool.model_snapshot()
            if session.candidate_pool is not None
            else None
        ),
        history=history,
        resolved_request=session.request_snapshot(),
        rendered_request=(
            render_resolved_request(resolved) if resolved is not None else None
        ),
        cached_poems=session.cached_poems_snapshot(),
        target_detail_checklist=build_target_detail_checklist(
            session, activated_details or {}
        ),
        request_phase_complete=request_phase_complete,
    )


def _history_for_prompt(session: AgentSession) -> list[dict]:
    """只从 Session 快照取历史，并移除当前已 frozen 的 target 元数据。"""
    active_ids = (
        {target.target_id for target in session.candidate_pool.targets}
        if session.candidate_pool is not None
        else set()
    )
    history = session.history_snapshot()
    for history_round in history:
        history_round["targets"] = [
            target
            for target in history_round.get("targets", [])
            if target.get("target_id") in active_ids
        ]
    return history


def _finalize_round(
    user_query: str, result: dict, session: AgentSession
) -> dict:
    public_result = _with_candidate_pool(result, session.candidate_pool)
    session.append_history(user_query, public_result)
    return public_result


def _closed_request_phase_observation(session: AgentSession) -> dict:
    """为阶段关闭后的重复请求动作生成带当前状态的恢复观察。"""
    candidate_pool = session.candidate_pool
    resolved = session.consolidated_request
    active_target_ids = (
        [target.target_id for target in candidate_pool.targets]
        if candidate_pool is not None
        else []
    )
    task_types = (
        list(dict.fromkeys(task.type for task in resolved.tasks))
        if resolved is not None
        else []
    )
    visible_ids = (
        candidate_pool.visible_candidate_ids()
        if candidate_pool is not None
        else []
    )
    cached_ids = [
        item["poem_id"] for item in session.cached_poems_snapshot()
    ]
    error = (
        "本轮请求动作已经成功，请求解析阶段已关闭；"
        f"当前 active target IDs={active_target_ids}，任务类型={task_types}。"
        "禁止再次 initialize/update，也禁止再次扩写、替换或增加 target。"
        f"下一步请读取 visible poem IDs={visible_ids} 或显式激活缓存 "
        f"poem IDs={cached_ids}，也可在条件满足时 finish。"
    )
    return {
        "error": error,
        "active_target_ids": active_target_ids,
        "task_types": task_types,
        "visible_candidate_ids": visible_ids,
        "activatable_cached_poem_ids": cached_ids,
    }


def _finish_readiness_error(
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


def _detail_coverage_priority_error(
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
        # search/read/verify-only target 保持原有合法读取行为。
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


def _action_signature(action: str, action_input: dict) -> tuple[str, str]:
    """将工具参数稳定序列化,使键顺序不同的等价调用也能被识别。"""
    normalized_input = json.dumps(
        action_input,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return action, normalized_input


def detect_repeated_action(
    action: str,
    action_input: dict,
    seen_actions: set[tuple[str, str]],
    stalled_action_counts: dict[str, int],
) -> str | None:
    """检测完全重复及连续多次无进展的动作,返回应回填的干预观察。"""
    if _action_signature(action, action_input) in seen_actions:
        return REPEATED_ACTION_OBSERVATION.format(action=action)
    if stalled_action_counts.get(action, 0) >= 3:
        return STALLED_ACTION_OBSERVATION
    return None


def _observation_made_progress(action: str, observation) -> bool:
    """判断工具结果是否带来了可供后续回答使用的新信息。"""
    if action == "get_poem_detail":
        return isinstance(observation, dict) and "error" not in observation
    return bool(observation)


def force_finish(
    user_query: str,
    llm,
    trajectory: list,
    session_poems: dict[int, str],
    *,
    candidate_pool: CandidatePool | None = None,
    session: AgentSession | None = None,
    cached_details: dict[str, dict] | None = None,
    request_phase_complete: bool = False,
    verbose: bool = False,
) -> dict:
    """步数耗尽时强制作答，并沿用正常 finish 的统一终检。"""
    if session is not None and not isinstance(session, AgentSession):
        raise TypeError("session 必须是 AgentSession 或 None")
    if session is None:
        history = []
        request_snapshot = None
        rendered_request = None
        cached_poems = []
    else:
        history = _history_for_prompt(session)
        request_snapshot = session.request_snapshot()
        rendered_request = (
            render_resolved_request(session.consolidated_request)
            if session.consolidated_request is not None
            else None
        )
        cached_poems = session.cached_poems_snapshot()
    target_detail_checklist = (
        build_target_detail_checklist(session, cached_details or {})
        if session is not None
        else []
    )
    prompt = build_force_finish_prompt(
        user_query,
        trajectory,
        session_poems,
        candidate_pool=(
            candidate_pool.model_snapshot()
            if candidate_pool is not None
            else None
        ),
        history=history,
        resolved_request=request_snapshot,
        rendered_request=rendered_request,
        cached_poems=cached_poems,
        target_detail_checklist=target_detail_checklist,
        request_phase_complete=request_phase_complete,
    )
    if verbose:
        print("          [兜底] 达到步数上限,基于已有信息作答")
        _print_final_separator()

    raw_answer = llm.generate(prompt)
    payload, payload_error = _extract_force_finish_payload(
        raw_answer, candidate_pool
    )
    regenerate = lambda feedback: _extract_force_finish_payload(
        llm.generate(_append_regeneration_feedback(prompt, feedback)),
        candidate_pool,
    )
    result = _unified_final_check(
        payload,
        payload_error,
        regenerate,
        trajectory,
        session_poems,
        candidate_pool,
        cached_details=cached_details,
        force=True,
        verbose=verbose,
    )
    return _with_candidate_pool(result, candidate_pool)


def _with_candidate_pool(
    result: dict, candidate_pool: CandidatePool | None
) -> dict:
    """统一追加公开精简池快照；无需检索的兼容路径返回 None。"""
    return {
        **result,
        "candidate_pool": (
            candidate_pool.public_snapshot()
            if candidate_pool is not None
            else None
        ),
    }


def _extract_force_finish_payload(
    raw: str, candidate_pool: CandidatePool | None
) -> tuple[dict | None, str | None]:
    """读取强制收尾紧凑结构，并兼容 action=finish 包装。"""
    decision = parse_decision(raw)
    if decision is not None and decision["action"] == "finish":
        action_input = decision["action_input"]
    else:
        obj = _parse_json_object(raw)
        action_input = obj if isinstance(obj, dict) else {}
    try:
        return validate_analysis_assessment(action_input, candidate_pool), None
    except AnalysisAssessmentProtocolError as exc:
        answer = action_input.get("answer", "") if isinstance(action_input, dict) else ""
        return (
            {
                "answer": answer if isinstance(answer, str) else "",
                "analysis_assessment": None,
            },
            f"强制收尾 assessment 非法: {exc}",
        )


def _extract_finish_payload(
    raw: str, candidate_pool: CandidatePool | None
) -> tuple[dict | None, str | None]:
    """统一重生成只接受完整、精确的正常 finish 协议。"""
    decision = parse_decision(raw)
    if decision is None or decision["action"] != "finish":
        return None, "重生成必须返回 action=finish 的 JSON"
    try:
        return (
            validate_analysis_assessment(
                decision["action_input"], candidate_pool
            ),
            None,
        )
    except AnalysisAssessmentProtocolError as exc:
        answer = decision["action_input"].get("answer", "")
        return (
            {
                "answer": answer if isinstance(answer, str) else "",
                "analysis_assessment": None,
            },
            f"重生成 finish 协议错误: {exc}",
        )


def _append_regeneration_feedback(prompt: str, feedback: str) -> str:
    """追加一次统一终检的全部修正反馈。"""
    return (
        f"{prompt}\n\n## 统一终检反馈\n{feedback}\n"
        "请一次性重新生成完整 answer 和仅含 level、target_ids 的 "
        "analysis_assessment。正常流程输出 action=finish 包装；步数耗尽流程"
        "可输出紧凑的 answer + analysis_assessment 对象。不要再调用工具。"
    )


def _unified_final_check(
    payload: dict | None,
    payload_error: str | None,
    regenerate,
    trajectory: list,
    session_poems: dict[int, str],
    candidate_pool: CandidatePool | None,
    *,
    cached_details: dict[str, dict] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """完整性、引用和支撑申报共享至多一次重生成预算。"""
    original_payload = payload
    issues, _ = _inspect_final_payload(
        payload,
        payload_error,
        trajectory,
        session_poems,
        candidate_pool,
        cached_details=cached_details,
    )
    if issues:
        if verbose:
            print("          [统一终检] 检测到问题，合并反馈后重试一次")
        feedback = "\n".join(f"- {issue}" for issue in issues)
        payload, payload_error = regenerate(feedback)

    # 正常 finish 初始 assessment 必然合法；重生成协议失败时仍可使用原申报。
    if (
        (payload is None or payload.get("analysis_assessment") is None)
        and not force
        and original_payload is not None
    ):
        payload = {
            "answer": (
                payload.get("answer", "")
                if isinstance(payload, dict)
                else ""
            ),
            "analysis_assessment": original_payload["analysis_assessment"],
        }
    answer = (
        payload.get("answer", "")
        if isinstance(payload, dict)
        and isinstance(payload.get("answer"), str)
        else ""
    )
    assessment = (
        payload.get("analysis_assessment")
        if isinstance(payload, dict)
        else None
    )

    degraded = False
    if is_answer_suspiciously_incomplete(answer):
        answer = answer_integrity_fallback(trajectory)
        degraded = True
        if verbose:
            print("          [统一终检] 重试后答案仍不完整，安全降级")

    answer = _strip_invalid_citation_markers(answer)
    evidence = collect_evidence(
        answer,
        trajectory,
        session_poems,
        cached_details=cached_details,
    )
    dangling = list_dangling_citations(evidence)
    if dangling:
        degraded = True
        answer = _append_degraded_notice(answer)
        if verbose:
            print("          [统一终检] 重试后仍有悬空引用，安全降级")

    if assessment is None:
        analysis_support = force_fallback_analysis_support()
    else:
        evaluation = evaluate_analysis_support(
            assessment,
            candidate_pool,
            evidence,
            degraded=degraded,
        )
        analysis_support = evaluation.analysis_support
    answer = append_required_support_notice(answer, analysis_support)
    return {
        "answer": answer,
        "evidence": evidence,
        "analysis_support": analysis_support,
        "degraded": degraded,
    }


def _inspect_final_payload(
    payload: dict | None,
    payload_error: str | None,
    trajectory: list,
    session_poems: dict[int, str],
    candidate_pool: CandidatePool | None,
    *,
    cached_details: dict[str, dict] | None = None,
) -> tuple[list[str], dict | None]:
    """收集一次终检的全部可修正问题，不产生状态修改。"""
    issues: list[str] = []
    if payload_error:
        issues.append(payload_error)
    if payload is None:
        issues.append("未形成可检查的最终回答结构")
        return issues, None

    answer = payload.get("answer", "")
    if is_answer_suspiciously_incomplete(answer):
        issues.append("answer 为空、过短或疑似截断，请重新生成完整回答")
    cleaned_answer = _strip_invalid_citation_markers(
        answer if isinstance(answer, str) else ""
    )
    evidence = collect_evidence(
        cleaned_answer,
        trajectory,
        session_poems,
        cached_details=cached_details,
    )
    dangling = list_dangling_citations(evidence)
    if dangling:
        issues.append(
            "存在悬空引用，请删除或改用真实依据："
            + "；".join(dangling)
        )

    assessment = payload.get("analysis_assessment")
    if not isinstance(assessment, dict):
        issues.append("analysis_assessment 缺失或非法")
        return issues, None
    evaluation = evaluate_analysis_support(
        assessment, candidate_pool, evidence, degraded=False
    )
    if evaluation.analysis_support["level"] != assessment["level"]:
        limited = [
            f"target {target_id}={state}"
            for target_id, state in evaluation.target_states.items()
            if state != "fully_supported"
        ]
        detail = "、".join(limited) or "当前没有合法分析 evidence"
        issues.append(
            f"申报 level={assessment['level']} 超过系统客观上限 "
            f"{evaluation.maximum}；{detail}"
        )
    return issues, evaluation.analysis_support


def _parse_json_object(raw: str) -> dict | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    obj = _try_json(text)
    if isinstance(obj, dict):
        return obj
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        obj = _try_json(text[start : end + 1])
    return obj if isinstance(obj, dict) else None


def _assign_session_poem(
    session_poems: dict[int, str], poem_id: str
) -> int:
    """为取到详情的诗分配稳定会话序号；同一 poem_id 始终复用原序号。"""
    for poem_number, known_poem_id in session_poems.items():
        if known_poem_id == poem_id:
            return poem_number
    poem_number = len(session_poems) + 1
    session_poems[poem_number] = poem_id
    return poem_number


def _env_flag_enabled(name: str) -> bool:
    """识别常见的环境变量布尔值。"""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def parse_decision(raw: str) -> dict | None:
    """解析 LLM 输出为决策字典 {thought, action, action_input}。
    解析失败或结构非法一律返回 None —— 由循环回填错误让模型自我修正,不抛异常。
    """
    if not raw or not isinstance(raw, str):
        return None

    # 1. LLM可能使用md包裹内容
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2. LLM可能在内容前有解释性文字
    obj = _try_json(text)
    if obj is None:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = _try_json(text[start : end + 1])
    if obj is None:
        return None

    # 3. 结构校验:必须是 dict,且 action 合法
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    if not isinstance(action, str) or not action:
        return None

    # action_input 缺失时补空 dict;不是 dict 则视为非法
    action_input = obj.get("action_input", {})
    if not isinstance(action_input, dict):
        return None

    # 4. 归一化返回(thought 可选)
    return {
        "thought": obj.get("thought", ""),
        "action": action,
        "action_input": action_input,
    }


def _try_json(s: str) -> dict | None:
    """安全地尝试 json.loads,失败返回 None。"""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
