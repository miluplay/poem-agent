"""可信度层。★ 你亲手写。纵线阶段:先只做引用绑定。"""
import re
from collections.abc import Callable

# 对外只允许赏析和注释两类引用；宽匹配用于剥离模型自造的类似引用。
_LEGAL_CITE = re.compile(r"\[诗\d+-(?:appr|note)-\d+\]")
_CITE_LIKE = re.compile(r"\[诗\d+[^\]]*\]")
_SESSION_CITE = re.compile(r"\[诗(\d+)-((?:appr|note)-\d+)\]")
_MIN_ANSWER_LENGTH = 10
_DANGLING_DEGRADED_NOTICE = (
    "（部分解读未能匹配到可靠出处，已尽力修正，请谨慎参考）"
)


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
    regenerate: Callable[[str], str] | None = None,
    *,
    verbose: bool = False,
) -> dict:
    """finish 后必经此处。
    先剥离非法 marker 并绑定引用；悬空时携带具体反馈重生成一次，
    重试仍失败才诚实降级。"""
    session_poems = session_poems or {}
    answer = _strip_invalid_citation_markers(answer)
    evidence = collect_evidence(answer, trajectory, session_poems)
    dangling_citations = list_dangling_citations(evidence)
    degraded = False
    regeneration_attempted = False

    if dangling_citations and regenerate is not None:
        regeneration_attempted = True
        if verbose:
            print(
                "          [悬空引用检查] 检测到悬空引用，"
                "携带具体出处反馈重试"
            )
        answer = _strip_invalid_citation_markers(
            regenerate(_dangling_regeneration_feedback(dangling_citations))
        )
        evidence = collect_evidence(answer, trajectory, session_poems)
        dangling_citations = list_dangling_citations(evidence)
        if verbose and not dangling_citations:
            print("          [悬空引用检查] 重试后已消除悬空引用")

    if dangling_citations:
        degraded = True
        answer = _append_degraded_notice(answer)
        if verbose:
            state = "重试后仍有" if regeneration_attempted else "存在"
            print(f"          [悬空引用检查] {state}悬空引用，降级")

    return {
        "answer": answer,
        "evidence": evidence,   # [{evidence_id, text, poem_id, title}]
        "degraded": degraded,
    }


def list_dangling_citations(evidence: list) -> list[str]:
    """返回按答案引用顺序排列的“诗号-段号 + 原因”简明清单。"""
    dangling: list[str] = []
    for item in evidence:
        if not isinstance(item, dict) or item.get("dangling") is not True:
            continue
        citation = item.get("citation") or item.get("evidence_id") or "未知引用"
        reason = item.get("reason") or "无法匹配出处"
        dangling.append(f"[{citation}]：{reason}")
    return dangling


def _dangling_regeneration_feedback(dangling_citations: list[str]) -> str:
    """把悬空清单转成一次性重生成所需的明确修正反馈。"""
    details = "\n".join(
        f"- 第 {index} 处引用 {citation}，指向不存在的出处。"
        for index, citation in enumerate(dangling_citations, start=1)
    )
    return (
        "你的回答存在悬空引用：\n"
        f"{details}\n"
        "请勿编造引用。请重新生成完整答案，只使用【已执行的步骤】中"
        "详情观察里真实存在的赏析/注释编号；若某处解读没有真实支撑，"
        "请删去该处解读。"
    )


def _append_degraded_notice(answer: str) -> str:
    """保留悬空正文和 marker，仅在尾部追加一次诚实提示。"""
    stripped = answer.rstrip()
    if stripped.endswith(_DANGLING_DEGRADED_NOTICE):
        return stripped
    separator = "\n" if stripped else ""
    return f"{stripped}{separator}{_DANGLING_DEGRADED_NOTICE}"


def _strip_invalid_citation_markers(answer: str) -> str:
    """静默剥离正文中形似引用但不符合合法格式的 marker。"""
    cleaned = _CITE_LIKE.sub(
        lambda match: (
            match.group(0)
            if _LEGAL_CITE.fullmatch(match.group(0))
            else ""
        ),
        answer,
    )
    # 清理剥离后形成的双空格、中文句中的孤立空格和行尾空格。
    cleaned = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", cleaned)
    cleaned = re.sub(
        r"(?<=[\u3400-\u9fff，。！？；：、])[ \t]+"
        r"(?=[\u3400-\u9fff，。！？；：、])",
        "",
        cleaned,
    )
    cleaned = re.sub(r"[ \t]+(?=[，。！？；：、,.!?;:])", "", cleaned)
    return re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)


def collect_evidence(
    answer: str,
    trajectory: list,
    session_poems: dict[int, str] | None = None,
) -> list:
    """★ 引用绑定核心:
    1. 从 answer 抽出所有 [诗N-appr-x]/[诗N-note-x] 引用;
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
                block = {
                    "evidence_id": full_id,
                    "text": item["text"],
                    "poem_id": poem_id,
                    "title": title,
                }
                index[full_id] = block
                # 语料中的注释 ID 沿用 anno-*，对外合法 marker 使用 note-*。
                if "#anno-" in full_id:
                    index[full_id.replace("#anno-", "#note-", 1)] = block
    return index
