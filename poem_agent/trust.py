"""可信度层。★ 你亲手写。纵线阶段:先只做引用绑定。"""
import re
from .utils import short_id

# 匹配 answer 里的引用标记,如 [appr-0]、[anno-12]、[appr-1-2]
_CITE_PATTERN = re.compile(r"\[(appr|anno)[-\w]*\]")
# 更精确地抽出方括号里的完整 id 主体(去掉方括号)
_CITE_ID = re.compile(r"\[((?:appr|anno)[-\w]*)\]")


def trustworthiness_check(answer: str, trajectory: list) -> dict:
    """finish 后必经此处。
    纵线阶段:把 answer 里引用到的证据块,从轨迹里捞出来附上(引用绑定)。
    后续增量:在这里加 no_hit / low_conf / 前提纠正 三种降级分支。"""
    evidence = collect_evidence(answer, trajectory)
    return {
        "answer": answer,
        "evidence": evidence,   # [{evidence_id, text, poem_id, title}]
        "degraded": False,      # 增量 3 起,降级时置 True 并带原因
    }


def collect_evidence(answer: str, trajectory: list) -> list:
    """★ 引用绑定核心:
    1. 从 answer 里抽出所有被引用的 evidence_id(如 appr-0、anno-2);
    2. 建一张 evidence_id → 完整块 的索引(遍历轨迹里 get_poem_detail 的观察);
    3. 按 answer 里出现的引用,返回对应的完整证据块。
    """
    # 1. answer 里模型标注的引用 id(去重,保留出现顺序)
    cited_ids: list[str] = []
    for m in _CITE_ID.findall(answer):
        if m not in cited_ids:
            cited_ids.append(m)

    # 2. 从轨迹里建 evidence_id → 块 的全量索引
    index = _build_evidence_index(trajectory)

    # 3. 按引用捞块;引用了但索引里没有的,标记为"悬空引用"(模型幻觉的信号)
    evidence: list[dict] = []
    for cid in cited_ids:
        block = index.get(cid)
        if block is not None:
            evidence.append(block)
        else:
            evidence.append({
                "evidence_id": cid,
                "text": None,
                "dangling": True,   # ← 模型引用了不存在的 id,可信度告警
            })
    return evidence


def _build_evidence_index(trajectory: list) -> dict:
    """遍历轨迹里所有 get_poem_detail 的观察,建 evidence_id → 完整块 索引。
    块里带上 poem_id/title,方便前端显示'出自哪首诗'。"""
    index: dict[str, dict] = {}
    for step in trajectory:
        obs = step.get("observation")
        if not isinstance(obs, dict) or "error" in obs:
            continue
        poem_id = obs.get("poem_id")
        title = obs.get("title")
        for key in ("appreciation", "annotations"):
            for item in obs.get(key, []):
                full_id = item.get("evidence_id")
                if not full_id: continue
                short = short_id(full_id)
                index[short] = {
                    "evidence_id": full_id,
                    "text": item["text"],
                    "poem_id": poem_id,
                    "title": title,
                }
    return index
