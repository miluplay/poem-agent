"""工具观察的摘要与会话诗编号辅助函数。"""

from __future__ import annotations

from ..utils import short_id


EMPTY_SEARCH_OBSERVATION = (
    "检索到 0 首。可能原因:①名称或字词有误(如作者名、诗名写错);"
    "②该作品不在语料库中。请考虑修正查询,或直接告知用户不在范围,"
    "不要用相同查询重复检索。"
)


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
            if short.startswith("anno-"):
                short = "note-" + short.removeprefix("anno-")
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
