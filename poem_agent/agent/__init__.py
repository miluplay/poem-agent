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
    required_support_notice,
    validate_analysis_assessment,
)
from ..candidate_pool import CandidatePool, CandidatePoolProtocolError
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


MAX_PRODUCTIVE_STEPS = 6
MAX_RECOVERY_STEPS = 2
MAX_TOTAL_STEPS = 8
# 保留旧名字供既有调用方读取；循环不再依赖单一预算。
MAX_STEPS = MAX_PRODUCTIVE_STEPS

REPEATED_ACTION_OBSERVATION = (
    "你刚才已经用相同参数执行过 {action},结果没有变化。请不要重复,"
    "请基于 Candidate Pool 和已有详情继续或结束作答。"
)
STALLED_ACTION_OBSERVATION = (
    "你已多次调用工具但未获得有效结果,请停止调用,"
    "基于 Candidate Pool 和已有信息作答或告知用户找不到。"
)


def run_agent(user_query: str, llm, verbose: bool = False) -> dict:
    verbose = verbose or _env_flag_enabled("POEM_AGENT_VERBOSE")
    trajectory = []   # 观察轨迹:每步的 thought/action/observation
    session_poems: dict[int, str] = {}  # 会话诗序号 → poem_id
    candidate_pool: CandidatePool | None = None
    productive_steps = 0
    recovery_steps = 0
    total_steps = 0

    while (
        productive_steps < MAX_PRODUCTIVE_STEPS
        and recovery_steps < MAX_RECOVERY_STEPS
        and total_steps < MAX_TOTAL_STEPS
    ):
        # 1. 构建 prompt(系统指令 + 工具描述 + 已有观察)
        prompt = build_prompt(
            user_query,
            trajectory,
            session_poems,
            candidate_pool=(
                candidate_pool.model_snapshot()
                if candidate_pool is not None
                else None
            ),
        )

        # 2. LLM 决策,要求返回结构化 JSON
        decision = parse_decision(llm.generate(prompt))
        total_steps += 1

        # 3. 解析失败 → 回填错误,让模型自我修正(不崩)
        if decision is None:
            error = "格式非法,请返回 JSON:{thought, action, action_input}"
            trajectory.append({"error": error})
            recovery_steps += 1
            if verbose:
                _print_step_error(total_steps, error)
            continue

        if verbose:
            _print_step_decision(total_steps, decision)

        # 4. 终止：协议合法后才计正常步，并进入一次统一终检。
        if decision["action"] == "finish":
            try:
                finish_payload = validate_analysis_assessment(
                    decision["action_input"], candidate_pool
                )
            except AnalysisAssessmentProtocolError as exc:
                error = f"finish 协议错误: {exc}"
                trajectory.append({"error": error})
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{error}")
                continue
            productive_steps += 1
            if verbose:
                _print_final_separator()
            regenerate = lambda feedback: _extract_finish_payload(
                llm.generate(_append_regeneration_feedback(prompt, feedback)),
                candidate_pool,
            )
            result = _unified_final_check(
                finish_payload,
                None,
                regenerate,
                trajectory,
                session_poems,
                candidate_pool,
                verbose=verbose,
            )
            return _with_candidate_pool(result, candidate_pool)

        # 5. Candidate Pool 是主循环局部持有的一次性有状态动作，不进 TOOLS。
        if decision["action"] == "initialize_candidate_pool":
            if candidate_pool is not None:
                error = (
                    "Candidate Pool 已成功初始化；同一 run 只允许成功初始化一次，"
                    "原池保持不变"
                )
                trajectory.append({"error": error})
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{error}")
                continue
            action_input = decision["action_input"]
            if set(action_input) != {"targets"}:
                error = (
                    "Candidate Pool 协议错误: action_input 只能且必须包含 targets"
                )
                trajectory.append({"error": error})
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{error}")
                continue
            try:
                candidate_pool = CandidatePool.initialize(
                    action_input["targets"]
                )
            except CandidatePoolProtocolError as exc:
                error = f"Candidate Pool 协议错误: {exc}"
                trajectory.append({"error": error})
                recovery_steps += 1
                if verbose:
                    _print_observation(f"错误：{error}")
                continue

            observation = candidate_pool.model_snapshot()
            productive_steps += 1
            trajectory.append(
                {
                    "thought": decision.get("thought", ""),
                    "action": decision["action"],
                    "input": action_input,
                    "observation": observation,
                }
            )
            if verbose:
                _print_observation(
                    _summarize_observation(
                        observation,
                        session_poems=session_poems,
                        concise=True,
                    )
                )
            continue

        # 6. 详情是唯一普通动作，执行前由主循环强制校验池与滚动窗口。
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
        legal_ids = (
            candidate_pool.visible_candidate_ids()
            if candidate_pool is not None
            else []
        )
        if candidate_pool is None:
            error = "Candidate Pool 尚未初始化，不能读取详情"
        elif set(action_input) != {"poem_id"}:
            error = "详情协议错误: action_input 只能且必须包含 poem_id"
        elif not isinstance(poem_id, str) or not poem_id.strip():
            error = "详情协议错误: poem_id 必须是非空字符串"
        elif candidate_pool.is_loaded(poem_id):
            error = "该作品详情已加载，请使用现有详情摘要和 trajectory"
        elif poem_id in candidate_pool.failed_candidate_ids:
            error = "该作品详情此前连续失败，已被隔离"
        elif poem_id not in legal_ids:
            error = "详情协议错误: poem_id 不属于当前可见未读窗口"

        if error is not None:
            observation = {
                "error": error,
                "poem_id": poem_id,
                "visible_candidate_ids": legal_ids,
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
                    f"错误：{error}；当前合法 IDs: {legal_ids}"
                )
            continue

        tool = TOOLS["get_poem_detail"]
        automatically_retried = False
        try:
            observation = tool(poem_id=poem_id)
        except Exception:
            # 工具异常仅允许相同 ID 立即重试；第二次异常原样向上抛出。
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
            candidate_pool.add_detail(observation, session_poems)
            productive_steps += 1

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
                    observation, session_poems=session_poems, concise=True
                )
            )

    # 7. 任一预算耗尽后强制收尾；内部重试/重筛不占 LLM 决策轮次。
    return force_finish(
        user_query,
        llm,
        trajectory,
        session_poems,
        candidate_pool=candidate_pool,
        verbose=verbose,
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
    verbose: bool = False,
) -> dict:
    """步数耗尽时强制作答，并沿用正常 finish 的统一终检。"""
    prompt = build_force_finish_prompt(
        user_query,
        trajectory,
        session_poems,
        candidate_pool=(
            candidate_pool.model_snapshot()
            if candidate_pool is not None
            else None
        ),
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
    evidence = collect_evidence(answer, trajectory, session_poems)
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
    evidence = collect_evidence(cleaned_answer, trajectory, session_poems)
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
    notice = required_support_notice(evaluation.analysis_support)
    if notice is not None and notice not in cleaned_answer.splitlines():
        issues.append(f"回答必须披露固定说明：{notice}")
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
