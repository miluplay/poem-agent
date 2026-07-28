"""可信度层。★ 你亲手写。纵线阶段:先只做引用绑定。"""
import re
from collections.abc import Callable

# 抽出会话诗序号和诗内短 id，如 [诗1-appr-0]、[诗2-anno-12]。
_SESSION_CITE = re.compile(r"\[诗(\d+)-((?:appr|anno)-[\w-]+)\]")
_MIN_ANSWER_LENGTH = 10


def is_answer_suspiciously_incomplete(answer: str) -> bool:
    """判断模型答案是否疑似被截断；这里只检查形态，不改写正常答案。"""
    if not isinstance(answer, str):
        return True
    stripped = answer.strip()
    return (
        not stripped
        or stripped.startswith("[")
        or len(stripped) < _MIN_ANSWER_LENGTH
    )


def answer_integrity_gate(
    answer: str,
    regenerate: Callable[[], str],
    trajectory: list,
    *,
    verbose: bool = False,
) -> tuple[str, bool]:
    """在答案返回前做一次完整性闸门，异常时仅重试一次，再失败则诚实降级。

    返回 ``(最终答案, 是否降级)``；正常答案及重试成功的答案都保持原文。
    """
    if not is_answer_suspiciously_incomplete(answer):
        return answer, False

    if verbose:
        print("          [完整性检查] 检测到答案疑似截断,重试")
    retried_answer = regenerate()
    if not is_answer_suspiciously_incomplete(retried_answer):
        return retried_answer, False

    if verbose:
        print("          [完整性检查] 检测到答案疑似截断,降级")
    titles = _collected_poem_titles(trajectory)
    title_list = "、".join(f"《{title}》" for title in titles) or "（暂无）"
    fallback = (
        f"生成回答时出现异常,已获取的资料涉及:{title_list}。\n"
        "请重试,或追问具体某一首诗。"
    )
    return fallback, True


def _collected_poem_titles(trajectory: list) -> list[str]:
    """从成功取得详情的轨迹中按首次出现顺序收集诗名。"""
    titles: list[str] = []
    for step in trajectory:
        if step.get("action") != "get_poem_detail":
            continue
        observation = step.get("observation")
        if not isinstance(observation, dict) or "error" in observation:
            continue
        title = observation.get("title")
        if isinstance(title, str) and title.strip() and title.strip() not in titles:
            titles.append(title.strip())
    return titles


def trustworthiness_check(
    answer: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
) -> dict:
    """finish 后必经此处。
    纵线阶段:把 answer 里引用到的证据块,从轨迹里捞出来附上(引用绑定)。
    后续增量:在这里加 no_hit / low_conf / 前提纠正 三种降级分支。"""
    evidence = collect_evidence(answer, trajectory, session_poems or {})
    return {
        "answer": answer,
        "evidence": evidence,   # [{evidence_id, text, poem_id, title}]
        "degraded": False,      # 增量 3 起,降级时置 True 并带原因
    }


def collect_evidence(
    answer: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
) -> list:
    """★ 引用绑定核心:
    1. 从 answer 抽出所有 [诗N-appr-x]/[诗N-anno-x] 引用;
    2. 用 N 经 session_poems 找 poem_id;
    3. 用 poem_id + "#" + 诗内短 id 定位完整证据块。
    """
    session_poems = session_poems or {}
    citations = _extract_session_citations(answer)

    # 完整 evidence_id 全局唯一，因此索引不会再被跨诗短 id 覆盖。
    index = _build_evidence_index(trajectory)

    evidence: list[dict] = []
    for poem_number, short_evidence_id in citations:
        poem_id = session_poems.get(poem_number)
        if poem_id is None:
            evidence.append(
                _dangling_evidence(
                    poem_number,
                    short_evidence_id,
                    reason="引用了未取详情的诗",
                )
            )
            continue

        full_evidence_id = f"{poem_id}#{short_evidence_id}"
        block = index.get(full_evidence_id)
        if block is not None:
            evidence.append({**block, "poem_number": poem_number})
        else:
            evidence.append(
                _dangling_evidence(
                    poem_number,
                    short_evidence_id,
                    poem_id=poem_id,
                    reason="段编号不存在",
                )
            )
    return evidence


def _extract_session_citations(answer: str) -> list[tuple[int, str]]:
    """提取引用并去重，同时保留它们在答案中的首次出现顺序。"""
    citations: list[tuple[int, str]] = []
    for poem_number, short_evidence_id in _SESSION_CITE.findall(answer):
        citation = (int(poem_number), short_evidence_id)
        if citation not in citations:
            citations.append(citation)
    return citations


def _dangling_evidence(
    poem_number: int,
    short_evidence_id: str,
    *,
    reason: str,
    poem_id: str | None = None,
) -> dict:
    """统一构造两类悬空引用，保留模型原始的会话引用以便排查。"""
    return {
        "evidence_id": (
            f"{poem_id}#{short_evidence_id}"
            if poem_id is not None
            else short_evidence_id
        ),
        "text": None,
        "poem_id": poem_id,
        "poem_number": poem_number,
        "citation": f"诗{poem_number}-{short_evidence_id}",
        "dangling": True,
        "reason": reason,
    }


def _build_evidence_index(trajectory: list) -> dict:
    """遍历轨迹里所有 get_poem_detail 观察，建完整 evidence_id → 块索引。
    块里带上 poem_id/title,方便前端显示'出自哪首诗'。"""
    index: dict[str, dict] = {}
    for step in trajectory:
        if step.get("action") != "get_poem_detail":
            continue
        obs = step.get("observation")
        if not isinstance(obs, dict) or "error" in obs:
            continue
        poem_id = obs.get("poem_id")
        title = obs.get("title")
        for key in ("appreciation", "annotations"):
            for item in obs.get(key, []):
                full_id = item.get("evidence_id")
                if not full_id: continue
                index[full_id] = {
                    "evidence_id": full_id,
                    "text": item["text"],
                    "poem_id": poem_id,
                    "title": title,
                }
    return index
