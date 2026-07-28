"""可信度层。★ 你亲手写。纵线阶段:先只做引用绑定。"""
import re
from collections.abc import Callable

# 分诗采信度阈值，集中定义以便后续 eval 校准。
CONF_NORMAL = 0.6
CONF_LOWCONF = 0.35

# 对外只允许赏析和注释两类引用；宽匹配用于剥离模型自造的类似引用。
_LEGAL_CITE = re.compile(r"\[诗\d+-(?:appr|note)-\d+\]")
_CITE_LIKE = re.compile(r"\[诗\d+[^\]]*\]")
_SESSION_CITE = re.compile(r"\[诗(\d+)-((?:appr|note)-\d+)\]")
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
    *,
    verbose: bool = False,
) -> dict:
    """finish 后必经此处。
    纵线阶段:把 answer 里引用到的证据块,从轨迹里捞出来附上(引用绑定)。
    后续增量:在这里加 no_hit / low_conf / 前提纠正 三种降级分支。"""
    session_poems = session_poems or {}
    answer = _strip_invalid_citation_markers(answer)
    evidence = collect_evidence(answer, trajectory, session_poems)
    confidence = compute_confidence(trajectory, session_poems)
    if verbose:
        _print_confidence(confidence)
    return {
        "answer": answer,
        "evidence": evidence,   # [{evidence_id, text, poem_id, title}]
        "confidence": confidence,
        "degraded": False,      # 增量 3 起,降级时置 True 并带原因
    }


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


def compute_confidence(
    trajectory: list,
    session_poems: dict[int, str],
) -> dict:
    """按每首进入会话的诗计算检索采信度，不改变答案或降级状态。

    每个 poem_id 取所有 ``search_poems`` 观察中的最高 score；从未在检索
    结果中出现的诗按 0 分处理。诗序号和 poem_id 以 session_poems 为准，
    标题则从成功的 ``get_poem_detail`` 观察中补齐。
    """
    highest_scores: dict[str, float] = {}
    titles: dict[str, str] = {}

    for step in trajectory:
        action = step.get("action")
        observation = step.get("observation")

        if action == "search_poems" and isinstance(observation, list):
            for candidate in observation:
                if not isinstance(candidate, dict):
                    continue
                poem_id = candidate.get("poem_id")
                score = candidate.get("score")
                if (
                    not isinstance(poem_id, str)
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                ):
                    continue
                numeric_score = float(score)
                highest_scores[poem_id] = max(
                    highest_scores.get(poem_id, numeric_score),
                    numeric_score,
                )

        elif (
            action == "get_poem_detail"
            and isinstance(observation, dict)
            and "error" not in observation
        ):
            poem_id = observation.get("poem_id")
            title = observation.get("title")
            if isinstance(poem_id, str) and isinstance(title, str):
                titles[poem_id] = title

    confidence_table: dict[int, dict] = {}
    for poem_number, poem_id in session_poems.items():
        score = highest_scores.get(poem_id, 0.0)
        confidence_table[poem_number] = {
            "poem_id": poem_id,
            "title": titles.get(poem_id),
            "score": score,
            "level": _confidence_level(score),
        }

    levels = [item["level"] for item in confidence_table.values()]
    overall_level = (
        min(levels, key={"normal": 2, "low_conf": 1, "no_hit": 0}.get)
        if levels
        else "no_hit"
    )
    return {
        "confidence_table": confidence_table,
        "overall_level": overall_level,
    }


def _confidence_level(score: float) -> str:
    """把单首诗的最高检索分数映射为具名采信度等级。"""
    if score >= CONF_NORMAL:
        return "normal"
    if score >= CONF_LOWCONF:
        return "low_conf"
    return "no_hit"


def _print_confidence(confidence: dict) -> None:
    """verbose 模式下打印便于人工核对的分诗采信度表。"""
    table = confidence["confidence_table"]
    if not table:
        print("[采信度] 无诗 → overall=no_hit")
        return
    summaries = [
        f'诗{poem_number}《{item["title"] or "未知标题"}》 '
        f'score={item["score"]:.2f} → {item["level"]}'
        for poem_number, item in table.items()
    ]
    print("[采信度] " + ";".join(summaries))


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
