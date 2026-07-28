"""手写 agent 循环。第一条纵线:一步 finish 版。
★ 这是你要亲手写透的核心,面试会被追问每个环节。"""
import json
import os
import re

from .tools import TOOLS
from .trust import answer_integrity_gate, trustworthiness_check
from .utils import short_id

MAX_STEPS = 6

REPEATED_ACTION_OBSERVATION = (
    "你刚才已经用相同参数执行过 {action},结果没有变化。请不要重复,"
    "换一种查询,或基于已有信息作答/告知用户。"
)
STALLED_ACTION_OBSERVATION = (
    "你已多次检索但未获得有效结果,请停止检索,"
    "基于已有信息作答或告知用户找不到。"
)
EMPTY_SEARCH_OBSERVATION = (
    "检索到 0 首。可能原因:①名称或字词有误(如作者名、诗名写错);"
    "②该作品不在语料库中。请考虑修正查询,或直接告知用户不在范围,"
    "不要用相同查询重复检索。"
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
            answer, degraded = answer_integrity_gate(
                decision["action_input"].get("answer", ""),
                lambda: _extract_finish_answer(llm.generate(prompt)),
                trajectory,
                verbose=verbose,
            )
            result = trustworthiness_check(answer, trajectory, session_poems)
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
    answer, degraded = answer_integrity_gate(
        answer,
        lambda: _extract_force_finish_answer(llm.generate(prompt)),
        trajectory,
        verbose=verbose,
    )
    result = trustworthiness_check(answer, trajectory, session_poems)
    if degraded:
        result["degraded"] = True
    return result


def build_force_finish_prompt(
    user_query: str,
    trajectory: list,
    session_poems: dict[int, str],
) -> str:
    """构造只允许输出最终答案、不再调用工具的步数耗尽提示。"""
    return (
        build_prompt(user_query, trajectory, session_poems)
        + "\n\n## 步数耗尽后的最终要求\n"
        "已达到步数上限。请基于【已检索到的信息】给出你目前能给出的最佳回答,"
        "诚实说明这是基于现有信息的初步回答,并具体提示用户可以如何追问以获得"
        "更完整的结果。仍需遵守:每处解读标注 [诗N-xxx] 引用,不得编造。"
        "如果轨迹里没有任何有效检索结果,请诚实说明没有找到。"
        "这一次不要再选择工具,也不要输出 JSON,只输出最终回答正文。"
    )


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


def _format_action(action: str, action_input: dict) -> str:
    if action == "finish" and "answer" in action_input:
        action_input = {**action_input, "answer": "（见最终作答）"}
    args = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in action_input.items()
    )
    return f"{action}({args})"


def _print_step_decision(step: int, decision: dict) -> None:
    thought = decision.get("thought") or "（无）"
    print(f"\n[步骤 {step}] 💭 {thought}")
    print(f"          🔧 {_format_action(decision['action'], decision['action_input'])}")


def _print_step_error(step: int, error: str) -> None:
    print(f"\n[步骤 {step}] 💭 （模型返回格式非法）")
    _print_observation(f"错误：{error}")


def _print_observation(summary: str) -> None:
    print(f"          👀 {summary}")


def _print_final_separator() -> None:
    print("\n───────── 最终作答 ─────────")


# prompt
SYSTEM_INSTRUCTION = """你是古诗文赏析与交流研究助手。你只能依据【工具返回的资料】作答,严禁凭记忆或常识补充任何原文、注释、译文或赏析内容。

## 输出格式(严格)
每一步只输出一个 JSON 对象,不要输出任何额外文字、不要用 markdown 代码块包裹:
{
  "thought": "你的推理:现在掌握了什么,还缺什么",
  "action": "工具名 或 finish",
  "action_input": { ... }
}

## 可用工具
- search_poems(query: str, top_k: int = 5):按用户意图检索诗词,返回候选列表,
  每项包含真实 poem_id、标题、作者和相似度。适用于用户提到某首诗的标题,
  或描述某类主题、意象、情感、作者的诗。top_k 可省略。
- get_poem_detail(poem_id: str):按真实 poem_id 取一首诗的正文、注释、译文、
  赏析。poem_id 必须原样复制自 search_poems 的候选,不要填写标题。

## 工具调用流程
回答任何关于具体诗词的问题前,必须先调用 search_poems 检索。拿到候选后,
选择与用户意图最匹配的一首,从候选中原样复制其 poem_id,再调用
get_poem_detail(poem_id) 取详情后作答。
若 search_poems 返回空列表,说明作品不在当前语料范围,按无据不答处理。

## finish 的用法
当你已获得足够资料回答用户时,输出:
{
  "thought": "...",
  "action": "finish",
  "action_input": { "answer": "你的回答,每处解读后标注引用编号,如 [诗1-appr-0]" }
}

## 必须遵守
1. 无据不答:若 search_poems 返回空列表,或 get_poem_detail 返回 not_found,
   直接 finish,并在 answer 中说明"该作品不在当前语料范围,无法给出有依据的解读",
   不得编造原文或赏析。
2. 引用标注:finish 的 answer 中,每一处解读都要标注它依据的 evidence_id
   (如 [诗1-appr-0]、[诗1-anno-2]),编号必须原样使用详情观察中展示的
   appreciation/annotations 块引用。
3. 只依据资料:不要用你自己的知识补充或"纠正"工具返回的内容。
"""

# 赏析进行摘要处理
def _summarize_observation(
    obs: dict | list[dict] | str,
    *,
    session_poems: dict[int, str] | None = None,
    concise: bool = False,
) -> str:
    """把工具结果整理成带引用编号、可直接作答的上下文。"""
    if isinstance(obs, str):
        return obs

    if isinstance(obs, list):
        if not obs:
            return EMPTY_SEARCH_OBSERVATION
        candidates = []
        for item in obs:
            candidates.append(
                f'《{item["title"]}》{item["author"]} '
                f'(poem_id={item["poem_id"]}, score={item["score"]:.2f})'
            )
        return f"检索到 {len(obs)} 首候选:\n" + "\n".join(candidates)

    # 错误情况:如 not_found,直接把错误透传给模型(触发无据不答)
    if "error" in obs:
        return f'错误:{obs["error"]}(poem_id={obs.get("poem_id", "?")})'

    appr = obs.get("appreciation", [])
    anno = obs.get("annotations", [])
    poem_number = _session_poem_number(session_poems or {}, obs.get("poem_id"))
    poem_label = f"【诗{poem_number}】" if poem_number is not None else ""

    if concise:
        return (
            f'{poem_label}取到《{obs["title"]}》'
            f'({obs["dynasty"]}·{obs["author"]})，'
            f"赏析 {len(appr)} 块，注释 {len(anno)} 条。"
        )

    lines = [
        f'{poem_label}取到《{obs["title"]}》'
        f'({obs["dynasty"]}·{obs["author"]})。',
        f'正文：\n{obs.get("content", "")}',
    ]

    if appr:
        lines.append(f"赏析共 {len(appr)} 块:")
        for item in appr:
            short = short_id(item["evidence_id"])
            cite = (
                f"诗{poem_number}-{short}"
                if poem_number is not None
                else short
            )
            lines.append(f"[{cite}] {item['text']}")
    else:
        lines.append("(无赏析)")

    if anno:
        lines.append(f"注释共 {len(anno)} 条:")
        for item in anno:
            short = short_id(item["evidence_id"])
            cite = (
                f"诗{poem_number}-{short}"
                if poem_number is not None
                else short
            )
            lines.append(f"[{cite}] {item['text']}")

    return "\n".join(lines)


def _session_poem_number(
    session_poems: dict[int, str], poem_id: str | None
) -> int | None:
    """按真实 poem_id 反查会话诗序号。"""
    for poem_number, known_poem_id in session_poems.items():
        if known_poem_id == poem_id:
            return poem_number
    return None


def build_prompt(
    user_query: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
) -> str:
    """系统指令 + 观察轨迹 + 用户问题。"""
    parts = [SYSTEM_INSTRUCTION]
    session_poems = session_poems or {}

    # 观察轨迹:把之前每步的 action + observation 回填,让模型看到进展
    if trajectory:
        parts.append("\n## 已执行的步骤")
        for i, step in enumerate(trajectory):
            if "error" in step:
                parts.append(f"[{i}] 错误:{step['error']}")
            else:
                obs = _summarize_observation(
                    step["observation"], session_poems=session_poems
                )
                parts.append(f'[{i}] 调用 {step["action"]}({step["input"]}) → {obs}')

    parts.append(f"\n## 用户问题\n{user_query}")
    parts.append("\n请输出你这一步的 JSON 决策:")
    return "\n".join(parts)

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
