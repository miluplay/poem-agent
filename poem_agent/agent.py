"""手写 agent 循环。第一条纵线:一步 finish 版。
★ 这是你要亲手写透的核心,面试会被追问每个环节。"""
import json
import os
import re

from .tools import TOOLS
from .trust import trustworthiness_check
from .utils import short_id

MAX_STEPS = 4   # 纵线阶段够用;后面 A 线再调大

def run_agent(user_query: str, llm, verbose: bool = False) -> dict:
    verbose = verbose or _env_flag_enabled("POEM_AGENT_VERBOSE")
    trajectory = []   # 观察轨迹:每步的 thought/action/observation

    for step in range(MAX_STEPS):
        # 1. 构建 prompt(系统指令 + 工具描述 + 已有观察)
        prompt = build_prompt(user_query, trajectory)

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
            return trustworthiness_check(decision["action_input"]["answer"], trajectory)

        # 5. 校验工具
        tool = TOOLS.get(decision["action"])
        if tool is None:
            error = f"未知工具 {decision['action']}"
            trajectory.append({"error": error})
            if verbose:
                _print_observation(f"错误：{error}")
            continue

        # 6. 执行工具,观察回填轨迹
        try:
            observation = tool(**decision["action_input"])
        except (TypeError, ValueError) as exc:
            error = f"工具参数错误: {exc}"
            trajectory.append({"error": error})
            if verbose:
                _print_observation(f"错误：{error}")
            continue
        trajectory.append({
            "thought": decision.get("thought", ""),
            "action": decision["action"],
            "input": decision["action_input"],
            "observation": observation,
        })
        if verbose:
            _print_observation(_summarize_observation(observation, concise=True))

    # 7. 步数耗尽兜底:基于已有轨迹收尾(先简单返回,后面接 force_finish)
    if verbose:
        _print_final_separator()
    return trustworthiness_check("(达到最大步数)", trajectory)


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
  包含 poem_id、标题、作者、朝代、相似度和匹配方式。适用于用户提到某首诗的
  标题,或描述某类主题、意象、情感、作者的诗。top_k 可省略。
- get_poem_detail(poem_id: str):取一首诗的正文、注释、译文、赏析。

## 工具调用流程
回答任何关于具体诗词的问题前,必须先调用 search_poems 检索。拿到候选后,
选择与用户意图最匹配的一首,再调用 get_poem_detail(poem_id) 取详情后作答。
若 search_poems 返回空列表,说明作品不在当前语料范围,按无据不答处理。

## finish 的用法
当你已获得足够资料回答用户时,输出:
{
  "thought": "...",
  "action": "finish",
  "action_input": { "answer": "你的回答,每处解读后标注引用编号,如 [appr-0]" }
}

## 必须遵守
1. 无据不答:若 search_poems 返回空列表,或 get_poem_detail 返回 not_found,
   直接 finish,并在 answer 中说明"该作品不在当前语料范围,无法给出有依据的解读",
   不得编造原文或赏析。
2. 引用标注:finish 的 answer 中,每一处解读都要标注它依据的 evidence_id
   (如 [appr-0]、[anno-2]),编号对应工具返回的 appreciation/annotations 块。
3. 只依据资料:不要用你自己的知识补充或"纠正"工具返回的内容。
"""

# 赏析进行摘要处理
def _summarize_observation(
    obs: dict | list[dict], *, concise: bool = False
) -> str:
    """把工具结果整理成带引用编号、可直接作答的上下文。"""
    if isinstance(obs, list):
        if not obs:
            return "未检索到候选诗词。"
        candidates = []
        for index, item in enumerate(obs, start=1):
            candidates.append(
                f'{index}.《{item["title"]}》{item["author"]} '
                f'(poem_id={item["poem_id"]}, score={item["score"]:.2f})'
            )
        return f"检索到 {len(obs)} 首候选:" + " ".join(candidates)

    # 错误情况:如 not_found,直接把错误透传给模型(触发无据不答)
    if "error" in obs:
        return f'错误:{obs["error"]}(poem_id={obs.get("poem_id", "?")})'

    appr = obs.get("appreciation", [])
    anno = obs.get("annotations", [])

    if concise:
        return (
            f'取到《{obs["title"]}》({obs["dynasty"]}·{obs["author"]})，'
            f"赏析 {len(appr)} 块，注释 {len(anno)} 条。"
        )

    lines = [
        f'取到《{obs["title"]}》({obs["dynasty"]}·{obs["author"]})。',
        f'正文：\n{obs.get("content", "")}',
    ]

    if appr:
        lines.append(f"赏析共 {len(appr)} 块:")
        for item in appr:
            short = short_id(item["evidence_id"])
            lines.append(f"[{short}] {item['text']}")
    else:
        lines.append("(无赏析)")

    if anno:
        lines.append(f"注释共 {len(anno)} 条:")
        for item in anno:
            short = short_id(item["evidence_id"])
            lines.append(f"[{short}] {item['text']}")

    return "\n".join(lines)


def build_prompt(user_query: str, trajectory: list) -> str:
    """系统指令 + 观察轨迹 + 用户问题。"""
    parts = [SYSTEM_INSTRUCTION]

    # 观察轨迹:把之前每步的 action + observation 回填,让模型看到进展
    if trajectory:
        parts.append("\n## 已执行的步骤")
        for i, step in enumerate(trajectory):
            if "error" in step:
                parts.append(f"[{i}] 错误:{step['error']}")
            else:
                obs = _summarize_observation(step["observation"])
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
