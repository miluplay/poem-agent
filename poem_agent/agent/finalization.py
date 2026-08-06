"""finish、强制收尾与答案统一终检。"""

from __future__ import annotations

from ..contracts import AnalysisSupport, FinalResult
from ..evidence.citations import (
    _append_degraded_notice,
    _strip_invalid_citation_markers,
    collect_evidence,
    list_dangling_citations,
)
from ..evidence.integrity import (
    answer_integrity_fallback,
    is_answer_suspiciously_incomplete,
)
from ..evidence.support import (
    AnalysisAssessmentProtocolError,
    append_required_support_notice,
    evaluate_analysis_support,
    force_fallback_analysis_support,
    validate_analysis_assessment,
)
from ..candidate_pool import CandidatePool
from ..request import render_resolved_request
from ..session import AgentSession
from .decisions import parse_decision, parse_json_object
from .detail_policy import build_target_detail_checklist
from .display import _print_final_separator
from .prompts import build_force_finish_prompt


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
) -> FinalResult:
    """步数耗尽时强制作答，并沿用正常 finish 的统一终检。"""
    if session is not None and not isinstance(session, AgentSession):
        raise TypeError("session 必须是 AgentSession 或 None")
    if session is None:
        history = []
        request_snapshot = None
        rendered_request = None
        cached_poems = []
    else:
        history = history_for_prompt(session)
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
    payload, payload_error = extract_force_finish_payload(
        raw_answer, candidate_pool
    )
    regenerate = lambda feedback: extract_force_finish_payload(
        llm.generate(append_regeneration_feedback(prompt, feedback)),
        candidate_pool,
    )
    result = unified_final_check(
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
    return with_candidate_pool(result, candidate_pool)


def with_candidate_pool(
    result: dict, candidate_pool: CandidatePool | None
) -> FinalResult:
    """统一追加公开精简池快照；无需检索的兼容路径返回 None。"""
    return {
        **result,
        "candidate_pool": (
            candidate_pool.public_snapshot()
            if candidate_pool is not None
            else None
        ),
    }


def extract_force_finish_payload(
    raw: str, candidate_pool: CandidatePool | None
) -> tuple[dict | None, str | None]:
    """读取强制收尾紧凑结构，并兼容 action=finish 包装。"""
    decision = parse_decision(raw)
    if decision is not None and decision["action"] == "finish":
        action_input = decision["action_input"]
    else:
        obj = parse_json_object(raw)
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


def extract_finish_payload(
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


def append_regeneration_feedback(prompt: str, feedback: str) -> str:
    """追加一次统一终检的全部修正反馈。"""
    return (
        f"{prompt}\n\n## 统一终检反馈\n{feedback}\n"
        "请一次性重新生成完整 answer 和仅含 level、target_ids 的 "
        "analysis_assessment。正常流程输出 action=finish 包装；步数耗尽流程"
        "可输出紧凑的 answer + analysis_assessment 对象。不要再调用工具。"
    )


def unified_final_check(
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
) -> FinalResult:
    """完整性、引用和支撑申报共享至多一次重生成预算。"""
    original_payload = payload
    issues, _ = inspect_final_payload(
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


def inspect_final_payload(
    payload: dict | None,
    payload_error: str | None,
    trajectory: list,
    session_poems: dict[int, str],
    candidate_pool: CandidatePool | None,
    *,
    cached_details: dict[str, dict] | None = None,
) -> tuple[list[str], AnalysisSupport | None]:
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


def history_for_prompt(session: AgentSession) -> list[dict]:
    """从 Session 快照取历史，并移除当前已 frozen 的 target 元数据。"""
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
