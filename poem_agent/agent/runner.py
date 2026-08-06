"""Agent 单轮执行与状态推进。"""

from __future__ import annotations

import os

from ..analysis_support import (
    AnalysisAssessmentProtocolError,
    validate_analysis_assessment,
)
from ..request import (
    ConsolidatedRequestProtocolError,
    normalize_consolidated_request,
    render_resolved_request,
    request_semantics_error,
)
from ..session import AgentSession
from .decisions import normalize_detail_action_input, parse_decision
from .detail_policy import (
    build_target_detail_checklist,
    detail_coverage_priority_error,
    finish_readiness_error,
)
from .display import (
    _print_final_separator,
    _print_observation,
    _print_step_decision,
    _print_step_error,
)
from .finalization import (
    append_regeneration_feedback,
    extract_finish_payload,
    force_finish,
    history_for_prompt,
    unified_final_check,
    with_candidate_pool,
)
from .observation import _summarize_observation
from .prompts import build_prompt


DETAILS_PER_ACTIVE_TARGET = 2
MAX_FRAMEWORK_STEPS = 2
MAX_RECOVERY_STEPS = 2


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
            readiness_error = finish_readiness_error(
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
            regenerate = lambda feedback: extract_finish_payload(
                llm.generate(append_regeneration_feedback(prompt, feedback)),
                current_session.candidate_pool,
            )
            result = unified_final_check(
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
                observation = _closed_request_phase_observation(current_session)
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
            if action == "initialize_candidate_pool" and current_session.candidate_pool:
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
        elif error is None and poem_id in candidate_pool.failed_candidate_ids:
            error = "该作品详情此前连续失败，已被隔离"
        cached_detail = (
            current_session.cached_detail(poem_id)
            if error is None and isinstance(poem_id, str)
            else None
        )
        if error is None and cached_detail is None and poem_id not in legal_ids:
            error = "详情协议错误: poem_id 不属于当前可见未读窗口"
        if error is None:
            error = detail_coverage_priority_error(
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
            # 动态读取包入口，保留 poem_agent.agent.TOOLS 的注入契约。
            from . import TOOLS

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
                "visible_candidate_ids": candidate_pool.visible_candidate_ids(),
            }
            recovery_steps += 1
        elif (
            not isinstance(observation, dict)
            or "error" in observation
            or observation.get("poem_id") != poem_id
        ):
            raise RuntimeError("get_poem_detail 返回了非法结果")
        else:
            if cached_detail is None:
                current_session.add_detail(observation)
            activated_details[poem_id] = observation
            detail_steps += 1

        trajectory.append(
            {
                "thought": decision.get("thought", ""),
                "action": decision["action"],
                "input": decision["action_input"],
                "observation": observation,
            }
        )
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
    return build_prompt(
        user_query,
        trajectory,
        session.session_poems_snapshot(),
        candidate_pool=(
            session.candidate_pool.model_snapshot()
            if session.candidate_pool is not None
            else None
        ),
        history=history_for_prompt(session),
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


def _finalize_round(
    user_query: str, result: dict, session: AgentSession
) -> dict:
    public_result = with_candidate_pool(result, session.candidate_pool)
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


def _env_flag_enabled(name: str) -> bool:
    """识别常见的环境变量布尔值。"""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
