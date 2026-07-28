"""手写 agent 循环。第一条纵线:一步 finish 版。
★ 这是你要亲手写透的核心,面试会被追问每个环节。"""

import json
import os
import re

from ..tools import TOOLS
from ..trust import (
    answer_integrity_gate,
    trustworthiness_check,
)
from .display import (
    _print_final_separator,
    _print_observation,
    _print_step_decision,
    _print_step_error,
)
from .observation import (
    EMPTY_SEARCH_OBSERVATION,
    _session_poem_number,
    _summarize_observation,
)
from .prompts import (
    SYSTEM_INSTRUCTION,
    build_force_finish_prompt,
    build_prompt,
)


MAX_STEPS = 6

REPEATED_ACTION_OBSERVATION = (
    "你刚才已经用相同参数执行过 {action},结果没有变化。请不要重复,"
    "换一种查询,或基于已有信息作答/告知用户。"
)
STALLED_ACTION_OBSERVATION = (
    "你已多次检索但未获得有效结果,请停止检索,"
    "基于已有信息作答或告知用户找不到。"
)


def run_agent(user_query: str, llm, verbose: bool = False) -> dict:
    verbose = verbose or _env_flag_enabled("POEM_AGENT_VERBOSE")
    trajectory = []   # 观察轨迹:每步的 thought/action/observation
    session_poems: dict[int, str] = {}  # 会话诗序号 → poem_id
    seen_actions: set[tuple[str, str]] = set()
    stalled_action_counts: dict[str, int] = {}

    for step in range(MAX_STEPS):
        # 1. 构建 prompt(系统指令 + 工具描述 + 已有观察)
        prompt = build_prompt(user_query, trajectory, session_poems)

        # 2. LLM 决策,要求返回结构化 JSON
        decision = parse_decision(llm.generate(prompt))

        # 3. 解析失败 → 回填错误,让模型自我修正(不崩)
        if decision is None:
            error = "格式非法,请返回 JSON:{thought, action, action_input}"
            trajectory.append({"error": error})
            if verbose:
                _print_step_error(step + 1, error)
            continue

        if verbose:
            _print_step_decision(step + 1, decision)

        # 4. 终止:模型认为够了 → 进可信度层
        if decision["action"] == "finish":
            if verbose:
                _print_final_separator()
            regenerate = lambda feedback="": _extract_finish_answer(
                llm.generate(_append_regeneration_feedback(prompt, feedback))
            )
            answer, degraded = answer_integrity_gate(
                decision["action_input"].get("answer", ""),
                regenerate,
                trajectory,
                verbose=verbose,
            )
            result = trustworthiness_check(
                answer,
                trajectory,
                session_poems,
                regenerate=regenerate,
                verbose=verbose,
            )
            if degraded:
                result["degraded"] = True
            return result

        # 5. 校验工具
        tool = TOOLS.get(decision["action"])
        if tool is None:
            error = f"未知工具 {decision['action']}"
            trajectory.append({"error": error})
            if verbose:
                _print_observation(f"错误：{error}")
            continue

        # 6. 重复或持续无进展时不再执行工具,把干预作为观察交给下一轮。
        intervention = detect_repeated_action(
            decision["action"],
            decision["action_input"],
            seen_actions,
            stalled_action_counts,
        )
        if intervention is not None:
            trajectory.append(
                {
                    "thought": decision.get("thought", ""),
                    "action": decision["action"],
                    "input": decision["action_input"],
                    "observation": intervention,
                }
            )
            if verbose:
                print("          [拦截] 检测到重复检索")
                _print_observation(intervention)
            continue

        # 7. 执行工具,观察回填轨迹
        signature = _action_signature(
            decision["action"], decision["action_input"]
        )
        seen_actions.add(signature)
        try:
            observation = tool(**decision["action_input"])
        except (TypeError, ValueError) as exc:
            stalled_action_counts[decision["action"]] = (
                stalled_action_counts.get(decision["action"], 0) + 1
            )
            error = f"工具参数错误: {exc}"
            trajectory.append({"error": error})
            if verbose:
                _print_observation(f"错误：{error}")
            continue

        if _observation_made_progress(decision["action"], observation):
            stalled_action_counts[decision["action"]] = 0
        else:
            stalled_action_counts[decision["action"]] = (
                stalled_action_counts.get(decision["action"], 0) + 1
            )

        if (
            decision["action"] == "get_poem_detail"
            and isinstance(observation, dict)
            and "error" not in observation
        ):
            _assign_session_poem(session_poems, observation["poem_id"])

        trajectory_step = {
            "thought": decision.get("thought", ""),
            "action": decision["action"],
            "input": decision["action_input"],
            "observation": observation,
        }
        trajectory.append(trajectory_step)
        if verbose:
            if decision["action"] == "search_poems" and observation == []:
                print("          [提示] 空结果反思")
            _print_observation(
                _summarize_observation(
                    observation, session_poems=session_poems, concise=True
                )
            )

    # 8. 步数耗尽兜底:再给模型一次机会,仅基于现有检索轨迹作答。
    return force_finish(
        user_query, llm, trajectory, session_poems, verbose=verbose
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
    if action == "search_poems":
        return isinstance(observation, list) and bool(observation)
    if action == "get_poem_detail":
        return isinstance(observation, dict) and "error" not in observation
    return bool(observation)


def force_finish(
    user_query: str,
    llm,
    trajectory: list,
    session_poems: dict[int, str],
    *,
    verbose: bool = False,
) -> dict:
    """步数耗尽时强制作答,并沿用正常 finish 的引用可信度检查。"""
    prompt = build_force_finish_prompt(user_query, trajectory, session_poems)
    if verbose:
        print("          [兜底] 达到步数上限,基于已有信息作答")
        _print_final_separator()

    raw_answer = llm.generate(prompt)
    answer = _extract_force_finish_answer(raw_answer)
    regenerate = lambda feedback="": _extract_force_finish_answer(
        llm.generate(_append_regeneration_feedback(prompt, feedback))
    )
    answer, degraded = answer_integrity_gate(
        answer,
        regenerate,
        trajectory,
        verbose=verbose,
    )
    result = trustworthiness_check(
        answer,
        trajectory,
        session_poems,
        regenerate=regenerate,
        verbose=verbose,
    )
    if degraded:
        result["degraded"] = True
    return result


def _extract_force_finish_answer(raw: str) -> str:
    """兼容模型仍按旧协议返回 finish JSON 的情况,否则保留回答原文。"""
    decision = parse_decision(raw)
    if decision is not None:
        if decision["action"] == "finish":
            answer = decision["action_input"].get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer
        # force_finish 阶段不再接受新的工具决策,避免把决策 JSON 当答案返回。
        return (
            "已达到步数上限。这是基于现有信息的初步回答；模型未能形成最终作答。"
            "请补充准确的诗名、作者或希望分析的角度后继续追问。"
        )

    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return ""


def _extract_finish_answer(raw: str) -> str:
    """从正常 finish 的重试响应中提取答案；非法响应交给完整性闸门降级。"""
    decision = parse_decision(raw)
    if decision is None or decision["action"] != "finish":
        return ""
    answer = decision["action_input"].get("answer")
    return answer if isinstance(answer, str) else ""


def _append_regeneration_feedback(prompt: str, feedback: str) -> str:
    """有修正反馈时追加到原 prompt；完整性重试则保持原 prompt 不变。"""
    if not feedback:
        return prompt
    return (
        f"{prompt}\n\n## 上一次答案的引用修正反馈\n{feedback}\n"
        "请立刻按反馈重新作答。正常 finish 流程仍只输出 action=finish 的"
        " JSON；步数耗尽流程仍只输出最终回答正文。"
    )


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
