"""最终回答的完整性保护。"""

from collections.abc import Callable

_MIN_ANSWER_LENGTH = 10


def is_answer_suspiciously_incomplete(answer: str) -> bool:
    """判断模型答案是否疑似被截断；这里只检查形态，不改写正常答案。"""
    if not isinstance(answer, str):
        return True
    stripped = answer.strip()
    return not stripped or stripped.startswith("[") or len(stripped) < _MIN_ANSWER_LENGTH


def answer_integrity_gate(answer: str, regenerate: Callable[[], str], trajectory: list, *, verbose: bool = False) -> tuple[str, bool]:
    """在答案返回前做一次完整性闸门，异常时仅重试一次，再失败则诚实降级。"""
    if not is_answer_suspiciously_incomplete(answer):
        return answer, False
    if verbose:
        print("          [完整性检查] 检测到答案疑似截断,重试")
    retried_answer = regenerate()
    if not is_answer_suspiciously_incomplete(retried_answer):
        return retried_answer, False
    if verbose:
        print("          [完整性检查] 检测到答案疑似截断,降级")
    return answer_integrity_fallback(trajectory), True


def answer_integrity_fallback(trajectory: list) -> str:
    """构造完整性重试仍失败时的稳定安全回答。"""
    titles = _collected_poem_titles(trajectory)
    title_list = "、".join(f"《{title}》" for title in titles) or "（暂无）"
    return f"生成回答时出现异常,已获取的资料涉及:{title_list}。\n请重试,或追问具体某一首诗。"


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

