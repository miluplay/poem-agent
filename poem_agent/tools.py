"""供 agent 调用的工具函数。"""

from . import store

def get_poem_detail(poem_id: str) -> dict:
    """取单首的正文/注释/译文/赏析。
    返回结构(契约,不要改):
    {
        "poem_id": str, "title": str, "author": str, "dynasty": str,
        "content": str,
        "appreciation": [{"evidence_id": str, "text": str}, ...],
        "annotations": [{"evidence_id": str, "text": str}, ...],
        "source_url": str | None,   # 本地数据用 source 字段代替
    }
    找不到时返回 {"error": "not_found", "poem_id": poem_id}。"""
    poem = store.get_by_id(poem_id)
    if poem is None:
        return {"error": "not_found", "poem_id": poem_id}

    # 只暴露工具契约中的字段，并复制证据块，避免调用方改写缓存数据。
    return {
        "poem_id": poem["poem_id"],
        "title": poem["title"],
        "author": poem["author"],
        "dynasty": poem["dynasty"],
        "content": poem["content"],
        "appreciation": [
            {"evidence_id": item["evidence_id"], "text": item["text"]}
            for item in poem.get("appreciation", [])
        ],
        "annotations": [
            {"evidence_id": item["evidence_id"], "text": item["text"]}
            for item in poem.get("annotations", [])
        ],
        "source_url": poem.get("source_url") or poem.get("source"),
    }


# 工具注册表:循环靠它查名字、调用
TOOLS = {
    "get_poem_detail": get_poem_detail,
}
