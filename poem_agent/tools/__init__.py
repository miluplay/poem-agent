"""
工具契约层。每个工具 = 薄封装：校验入参 → 调下层原语 → 返回契约结构。
新增工具规则：
1. 检索/算法逻辑写到 retrieval/，不在本包实现；
2. 数据详情读取可直接调用 store；工具需在 __init__.py 注册进 TOOLS；
3. prompt 里的工具说明更新到 agent/prompts.py。
"""

from .detail import get_poem_detail
from .search import search_poems


# 工具注册表:循环靠它查名字、调用
TOOLS = {
    "get_poem_detail": get_poem_detail,
}

__all__ = ["TOOLS", "get_poem_detail", "search_poems"]
