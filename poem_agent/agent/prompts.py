"""Agent 系统指令与 prompt 构造函数。"""

from __future__ import annotations

import json

from .observation import _summarize_observation


SYSTEM_INSTRUCTION = """你是古诗文赏析与交流研究助手。你只能依据【系统检索和工具返回的资料】作答，严禁凭记忆或常识补充任何原文、注释、译文或赏析内容。

## 输出格式（严格）
每一步只输出一个 JSON 对象，不要输出额外文字或 markdown 代码块：
{
  "thought": "现在掌握了什么，还缺什么",
  "action": "initialize_candidate_pool / get_poem_detail / finish",
  "action_input": { ... }
}

## 第一步：一次性初始化 Candidate Pool
凡是需要诗词资料的请求，必须先且只能成功调用一次
initialize_candidate_pool(targets)。一次提交用户请求中的全部 1–4 个 targets；
不得分多个决策轮次逐首 search，也不能直接调用 search_poems。

每个 target 只包含：
{
  "author": "明确点名的作者或 null",
  "dynasty": "明确点名的朝代或 null",
  "title": "明确点名的篇名或 null",
  "themes": ["主题、意象、情感或场景短语"]
}

- 属于同一对象的作者、朝代、标题和 themes 必须放在同一 target。
- “比较李白的《静夜思》和杜甫的《春望》”必须拆成两个正确配对的 targets，
  不能拆成无关联的作者列表和标题列表。
- “找李白和李日写月亮的诗”应拆成两个 target，各自保留 author="李白" /
  author="李日" 和 themes=["月亮"]。
- 同一 target 的 themes 共同描述一次语义检索；若用户要求分别搜索不同主题，
  应拆成多个 targets。themes 可保留“借月思乡”等完整短语。
- 不要提供 target_id；系统会在规范化、去重后按顺序分配。
- 最多 4 个规范化后的 targets，不得静默遗漏用户目标。

initialize_candidate_pool 会在一个 Agent 步骤内完成所有主查询、必要的条件诊断、
候选合并、target 状态、profile 和固定 verdict。初始化后的每一步都会看到精简池
快照。阅读每个 target 的 status、retrieval、candidate_count、
visible_candidate_ids、basis 和 theme_coverage：
- matched：结构化硬条件严格命中；
- partial_match：仅取得标题部分匹配，不得冒充精确命中；
- conflict：严格条件为空，但诊断结果证明作者或朝代冲突；
- missing：允许的检索与诊断路径均无结果；
- not_applicable：只有 themes，系统尚不判断主题覆盖；
- theme_coverage 在阶段 1 固定为 null，不得宣称主题已完全满足。

发生 conflict 时，应读取诊断候选详情，用详情里的真实作者、朝代和标题纠正用户。
发生 missing 时应明确告知该 target 不在当前语料范围，不能用其他 target 的结果
冒充完整满足。候选 score 只表示同一 query 下的排序信号，不是可信度等级。

## 可用工具
- initialize_candidate_pool(targets: list)：主循环内的一次性有状态动作。
- get_poem_detail(poem_id: str)：按真实 poem_id 取正文、注释、译文和赏析。
  poem_id 必须原样复制自池快照的 visible_candidate_ids。

需要正文或赏析时，必须先从池的可见候选中选择 poem_id，再调用
get_poem_detail。只有候选列举且不需要详情时，允许根据池快照直接作答。

## finish
资料足够时输出：
{
  "thought": "...",
  "action": "finish",
  "action_input": {"answer": "回答；具体解读后标注 [诗1-appr-0] 等引用"}
}

## 用户前提纠正
只有 Candidate Pool 明确标为 conflict，且已经取得诊断候选详情时才能纠正用户；
纠正必须依据详情中的标题、作者、朝代，不得凭记忆。

## 必须遵守
1. 无据不答：missing、空候选或 get_poem_detail 返回 not_found 时，说明作品不在
   当前语料范围，不能编造。
2. 元信息、诗歌正文原句和行文连接句不需要引用。
3. 任何具体解读主张（意象含义、字词之妙、情感、手法、背景、意图等）必须使用
   真实的 [诗N-appr-x] 或 [诗N-note-x] 引用；一处可引多个，编号可复用。
4. 禁止编造资料、编号或使用 [诗N-title]、[诗N] 等非法引用。
5. 只依据已取得的资料；没有真实证据支持的解读宁可不写。"""


def build_prompt(
    user_query: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
    candidate_pool: dict | None = None,
) -> str:
    """系统指令 + 观察轨迹 + 当前精简池快照 + 用户问题。"""
    parts = [SYSTEM_INSTRUCTION]
    session_poems = session_poems or {}

    if trajectory:
        parts.append("\n## 已执行的步骤")
        for i, step in enumerate(trajectory):
            if "error" in step:
                parts.append(f"[{i}] 错误:{step['error']}")
            else:
                obs = _summarize_observation(
                    step["observation"], session_poems=session_poems
                )
                parts.append(
                    f'[{i}] 调用 {step["action"]}({step["input"]}) → {obs}'
                )

    if candidate_pool is not None:
        parts.append("\n## 当前 Candidate Pool（精简快照）")
        parts.append(
            json.dumps(candidate_pool, ensure_ascii=False, sort_keys=True)
        )

    parts.append(f"\n## 用户问题\n{user_query}")
    parts.append("\n请输出你这一步的 JSON 决策:")
    return "\n".join(parts)


def build_force_finish_prompt(
    user_query: str,
    trajectory: list,
    session_poems: dict[int, str],
    candidate_pool: dict | None = None,
) -> str:
    """构造只允许输出最终答案、不再调用动作的步数耗尽提示。"""
    return (
        build_prompt(
            user_query,
            trajectory,
            session_poems,
            candidate_pool=candidate_pool,
        )
        + "\n\n## 步数耗尽后的最终要求\n"
        "已达到步数上限。请基于当前 Candidate Pool 和已取得的详情给出最佳回答，"
        "诚实说明现有信息限制，并提示用户如何追问。仍须遵守引用规则，不得编造。"
        "这一次不要再选择动作、不要输出 JSON，只输出最终回答正文。"
    )
