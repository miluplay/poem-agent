"""Agent 系统指令与多轮 prompt 构造函数。"""

from __future__ import annotations

import json

from .observation import _summarize_observation


SYSTEM_INSTRUCTION = """你是古诗文赏析与交流研究助手。你只能依据系统检索和本轮显式取得的详情作答，严禁凭记忆补充原文、注释、译文或赏析。

## 输出格式（严格）
每一步只输出一个 JSON 对象：
{"thought":"...","action":"initialize_candidate_pool / update_candidate_pool / get_poem_detail / finish","action_input":{...}}

## 完整请求动作
需要诗词资料且当前请求尚未建立时，调用 initialize_candidate_pool；已有请求而完整意图的 target、task、field 或 aspect 变化时，调用 update_candidate_pool。纯追问可直接读取候选、激活缓存详情或 finish。
首次需要资料时应先且只能成功调用一次 initialize_candidate_pool，一次提交全部 1–6 个 targets；不能直接调用 search_poems。比较多首作品时，作者、标题、朝代和主题必须拆成两个正确配对的 targets，不能交叉配错。

initialize_candidate_pool 和 update_candidate_pool 的 action_input 必须且只能是完整的 targets + tasks，不是本轮增量，例如：
{"targets":[{"target_ref":"t1","author":"李白","dynasty":null,"title":null,"themes":["月亮"]}],"tasks":[{"type":"search","target_refs":["t1"]}]}
每个 target 必须精确包含 target_ref、author、dynasty、title、themes。task 类型为 search/read/appreciate/compare/verify，并按类型提供精确字段。target_ref 只在本次动作内关联 task；稳定 target_id 由系统分配。每轮最多成功一次 initialize/update。
target 字段是用户确认的检索约束，不是可凭常识补全的作品元数据。用户未明确诗题（通常以《》表达）时，新 target 的 title 必须为 null；用户未明确朝代时，新 target 的 dynasty 必须为 null。候选或详情中的诗题、朝代不能反向写入总请求。旧 target 未被用户修改的字段必须从当前 resolved request 原样继承。
tasks 描述用户最终要完成的请求，不是“先搜索再回答”的内部执行步骤。检索只是 Candidate Pool 内部过程，不能把赏析或对比请求写成 search。
正例：“赏析李白的《静夜思》，重点看月亮意象”应提交 appreciate + aspects=["imagery"]，不是 search。
follow-up 例：“改成杜甫写月亮的诗”若当前任务为 appreciate/imagery，只替换 target，继续 appreciate/imagery；除非用户明确说“只列举/只查找”。
task 的精确形状：
- search：type、target_refs；
- read：type、target_refs、fields，fields 只能是 content/annotations；
- appreciate/compare：type、target_refs、aspects、custom_aspects，aspects 只能是 theme/emotion/imagery/technique/structure/diction；compare 至少关联两个对象；
- verify：type、target_refs、fields，fields 只能是 author/dynasty/title。

当前总请求是唯一完整意图。update 必须提交合并后的完整版本，不能只提交新增部分。当前请求和池快照中的 target_id 才是会话稳定 ID。frozen targets 不可见，也不能用于 finish 的 analysis_assessment。

## Candidate Pool 与详情
池快照只展示 active targets。visible_candidate_ids 是当前允许读取的新作品 ID；poem_id 必须原样复制自池快照。需要分析、引用正文/注释/赏析时必须调用 get_poem_detail。
搜索 status=missing 表示当前语料筛选未命中。若候选曾命中但本次详情读取异常，不得宣称作品本身不存在或不在语料。初次 get_poem_detail 返回 not_found 会由系统内部自动重试。
theme_coverage 在阶段 1 固定为 null；候选 score 只用于同一查询内排序，不是可信度。
matched 表示硬条件严格命中；partial_match 只是标题部分匹配；conflict 表示诊断发现作者或朝代冲突；missing 表示当前允许路径无结果；只有 themes 时 status 为 not_applicable。纠正用户前提必须先读取 conflict 的诊断候选详情。

历史是旧轮事件，不表示旧轮依据已在本轮激活。缓存作品摘要只用于把“诗1”等稳定诗号定位到真实 poem_id；需要在本轮分析或引用旧诗时，仍必须调用一次 get_poem_detail。缓存命中会由系统直接返回，不要改用其他 ID 或重新搜索。

## finish
{"thought":"...","action":"finish","action_input":{"answer":"完整回答 [诗1-appr-0]","analysis_assessment":{"level":"sufficient","target_ids":[1]}}}
finish.action_input 只能包含 answer、analysis_assessment；assessment 只能包含 level、target_ids。level 为 sufficient/partial/insufficient/not_applicable。not_applicable 使用空 target_ids；其余必须使用当前 active target IDs。
sufficient 要求本轮详情与合法引用覆盖主要分析范围；partial 必须限定到已有支撑范围；insufficient 不得生成无依据的主体赏析。
sufficient/partial 在 finish 前必须让 assessment 的每个 target 都有本轮成功取得或显式激活的详情；分析任务还必须覆盖其 active target 范围。insufficient 可在无详情时诚实结束；conflict 澄清和纯元信息回答可用 not_applicable。

任何具体解读主张必须引用本轮详情中真实存在的 [诗N-appr-x] 或 [诗N-note-x]。元信息与正文原句不要求引用。不得编造引用、把局部依据推广为全部对象，或把检索排序当可信度。资料不足时明确范围和限制。"""


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_prompt(
    user_query: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
    candidate_pool: dict | None = None,
    *,
    history: list[dict] | None = None,
    resolved_request: dict | None = None,
    rendered_request: str | None = None,
    cached_poems: list[dict] | None = None,
    target_detail_checklist: list[dict] | None = None,
    request_phase_complete: bool = False,
) -> str:
    """构造分区明确且不泄露 frozen/完整缓存详情的多轮 prompt。"""
    parts = [SYSTEM_INSTRUCTION]
    session_poems = session_poems or {}

    parts.append("\n## 本轮请求阶段（系统确定）")
    if request_phase_complete:
        parts.append("状态：已完成")
        parts.append("合法 action：get_poem_detail / finish")
        parts.append(
            "禁止再次解释原始用户问题或调用 initialize_candidate_pool / "
            "update_candidate_pool。禁止再次扩写、替换或增加 target。"
        )
        parts.append(
            "下一步只能从当前 active Pool 的 visible poem ID 读取详情、"
            "显式激活缓存 poem ID，或在条件满足时 finish。"
        )
    elif candidate_pool is None:
        parts.append("状态：未完成")
        parts.append("合法 action：initialize_candidate_pool / finish")
    else:
        parts.append("状态：未完成")
        parts.append(
            "合法 action：update_candidate_pool / get_poem_detail / finish"
        )

    parts.append("\n## 最近会话历史（从旧到新）")
    parts.append(_json(history or []))

    parts.append("\n## 当前完整请求")
    if resolved_request is None:
        parts.append("尚未建立。")
    else:
        parts.append(f"固定句式：{rendered_request or '（未提供）'}")
        parts.append(f"Resolved JSON：{_json(resolved_request)}")

    parts.append("\n## 缓存作品摘要（不含详情）")
    parts.append(_json(cached_poems or []))

    checklist = target_detail_checklist or []
    parts.append("\n## 本轮分析详情覆盖清单（系统确定）")
    parts.append(_json(checklist))
    uncovered = [row for row in checklist if not row["covered"]]
    coverable = [
        row
        for row in uncovered
        if (
            row["activatable_cached_poem_ids"]
            or row["visible_candidate_poem_ids"]
        )
    ]
    if coverable:
        parts.append(
            "仍未覆盖且可补齐的 target IDs："
            f"{[row['target_id'] for row in coverable]}。"
        )
        for row in coverable:
            parts.append(
                f"target {row['target_id']}：可激活缓存 IDs="
                f"{row['activatable_cached_poem_ids']}；可读取 visible IDs="
                f"{row['visible_candidate_poem_ids']}。"
            )
        parts.append(
            "下一次 get_poem_detail 必须优先覆盖上述任一缺失 target。"
            "缓存 ID 即使不在 visible 中也可以显式激活。"
        )
        parts.append(
            "已覆盖 target 暂不需要第二首；只有所有可补齐的分析 target "
            "都覆盖后，且任务确需额外资料，才读取补充详情。"
        )
        parts.append(
            "用户所说“再加一首”的效果已经体现在当前 active target 中，"
            "不得把同一句重新解释为要从同一 target 连读多首作品。"
        )
    elif uncovered:
        parts.append(
            "未覆盖 target 当前均无合法缓存或 visible ID，不阻塞其他合法"
            "读取；资料仍不足时可诚实 finish 为 partial/insufficient，"
            "由终检决定。"
        )
    elif checklist:
        parts.append(
            "所有 active 分析 target 均已覆盖；可按任务需要继续读取合法"
            "补充详情，或在条件满足时 finish。"
        )
    else:
        parts.append("当前没有 active appreciate/compare target。")

    parts.append("\n## 本轮已执行的步骤")
    if not trajectory:
        parts.append("[]")
    else:
        for index, step in enumerate(trajectory):
            if "error" in step:
                parts.append(f"[{index}] 错误:{step['error']}")
                continue
            observation = _summarize_observation(
                step["observation"], session_poems=session_poems
            )
            parts.append(
                f'[{index}] 调用 {step["action"]}({step["input"]}) → '
                f"{observation}"
            )

    parts.append("\n## 当前 Candidate Pool（仅 active 精简快照）")
    parts.append(_json(candidate_pool) if candidate_pool is not None else "尚未建立。")
    question_label = (
        "已解析的当前用户问题（仅供回答，不得再次产生请求动作）"
        if request_phase_complete
        else "当前用户问题"
    )
    parts.append(f"\n## {question_label}\n{user_query}")
    parts.append("\n请输出你这一步的 JSON 决策:")
    return "\n".join(parts)


def build_force_finish_prompt(
    user_query: str,
    trajectory: list,
    session_poems: dict[int, str],
    candidate_pool: dict | None = None,
    *,
    history: list[dict] | None = None,
    resolved_request: dict | None = None,
    rendered_request: str | None = None,
    cached_poems: list[dict] | None = None,
    target_detail_checklist: list[dict] | None = None,
    request_phase_complete: bool = False,
) -> str:
    """构造只允许紧凑最终结构的额度耗尽提示。"""
    return (
        build_prompt(
            user_query,
            trajectory,
            session_poems,
            candidate_pool,
            history=history,
            resolved_request=resolved_request,
            rendered_request=rendered_request,
            cached_poems=cached_poems,
            target_detail_checklist=target_detail_checklist,
            request_phase_complete=request_phase_complete,
        )
        + "\n\n## 步数耗尽后的最终要求（决策额度）\n"
        "不要再调用动作。基于当前 active 池和本轮已取得详情给出最佳回答，"
        "诚实说明限制。只输出紧凑 JSON："
        '{"answer":"基于现有资料的最终回答……","analysis_assessment":'
        '{"level":"partial","target_ids":[1]}}。也兼容 action=finish 包装。'
    )
