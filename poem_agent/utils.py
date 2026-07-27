# utils.py(新建,因为 agent.py 和 trust.py 都要用它)
def short_id(evidence_id: str) -> str:
    """把完整 evidence_id 截成短 id:'5b9a...#appr-0' → 'appr-0'。
    没有 '#' 时原样返回。"""
    return evidence_id.split("#")[-1] if evidence_id else ""
