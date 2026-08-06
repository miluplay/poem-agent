"""完整用户请求的结构化协议、规范化和确定性渲染。"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Callable



TARGET_FIELDS = frozenset({"author", "dynasty", "title", "themes"})
_REQUEST_FIELDS = frozenset({"targets", "tasks"})
_TARGET_REF_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}\Z")
_TASK_TYPES = frozenset({"search", "read", "appreciate", "compare", "verify"})
_READ_FIELDS = frozenset({"content", "annotations"})
_VERIFY_FIELDS = frozenset({"author", "dynasty", "title"})
_ASPECTS = frozenset(
    {"theme", "emotion", "imagery", "technique", "structure", "diction"}
)
_ASPECT_LABELS = {
    "theme": "主题",
    "emotion": "情感",
    "imagery": "意象",
    "technique": "手法",
    "structure": "结构",
    "diction": "字词",
}
_FIELD_LABELS = {
    "content": "原文",
    "annotations": "注释",
    "author": "作者",
    "dynasty": "朝代",
    "title": "标题",
}
_EXPLICIT_TASK_SIGNALS = (
    ("compare", ("对比", "比较", "异同", "区别")),
    (
        "appreciate",
        ("赏析", "分析", "解读", "意象", "情感", "手法", "结构", "字词", "炼字"),
    ),
    ("read", ("原文", "全文", "注释", "读一下")),
    ("verify", ("核对", "验证", "是不是", "是否", "谁写", "作者", "朝代")),
    ("search", ("查找", "找", "列举", "有哪些", "推荐")),
)
_EXPLICIT_ASPECT_SIGNALS = (
    ("theme", ("主题",)),
    ("emotion", ("情感",)),
    ("imagery", ("意象",)),
    ("technique", ("手法",)),
    ("structure", ("结构",)),
    ("diction", ("字词", "炼字")),
)
_ANALYSIS_TASK_TYPES = frozenset({"appreciate", "compare"})
_BOOK_TITLE_PATTERN = re.compile(r"《([^《》]+)》")
_DYNASTY_ALIASES = {
    "先秦": ("先秦",),
    "两汉": ("两汉", "汉代", "汉朝"),
    "魏晋": ("魏晋",),
    "南北朝": ("南北朝",),
    "隋代": ("隋代", "隋朝"),
    "唐代": ("唐代", "唐朝"),
    "五代": ("五代",),
    "宋代": ("宋代", "宋朝"),
    "金朝": ("金朝", "金代"),
    "元代": ("元代", "元朝"),
    "明代": ("明代", "明朝"),
    "清代": ("清代", "清朝"),
    "近代": ("近代",),
    "现代": ("现代",),
}


class ConsolidatedRequestProtocolError(ValueError):
    """模型提交的内容不符合完整总请求协议。"""


@dataclass(frozen=True)
class NormalizedTargetFields:
    """Candidate Pool 与总请求共用的 target 字段规范化结果。"""

    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def identity(self) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
        return self.author, self.dynasty, self.title, self.themes


@dataclass(frozen=True)
class RequestTarget:
    target_ref: str
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "themes", tuple(self.themes))

    def snapshot(self) -> dict:
        return {
            "target_ref": self.target_ref,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass(frozen=True)
class RequestTask:
    type: str
    target_refs: tuple[str, ...]
    fields: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()
    custom_aspects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_refs", tuple(self.target_refs))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "aspects", tuple(self.aspects))
        object.__setattr__(self, "custom_aspects", tuple(self.custom_aspects))

    def snapshot(self) -> dict:
        value: dict[str, object] = {
            "type": self.type,
            "target_refs": list(self.target_refs),
        }
        if self.type in {"read", "verify"}:
            value["fields"] = list(self.fields)
        elif self.type in {"appreciate", "compare"}:
            value["aspects"] = list(self.aspects)
            value["custom_aspects"] = list(self.custom_aspects)
        return value


@dataclass(frozen=True)
class ConsolidatedRequest:
    targets: tuple[RequestTarget, ...]
    tasks: tuple[RequestTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def snapshot(self) -> dict:
        return {
            "targets": [target.snapshot() for target in self.targets],
            "tasks": [task.snapshot() for task in self.tasks],
        }


@dataclass(frozen=True)
class ResolvedTarget:
    """用会话级稳定 ID 表达的 canonical target。"""

    target_id: int
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "themes", tuple(self.themes))

    def snapshot(self) -> dict:
        return {
            "target_id": self.target_id,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass(frozen=True)
class ResolvedTask:
    """临时 target_ref 已被替换为持久 target_id 的任务。"""

    type: str
    target_ids: tuple[int, ...]
    fields: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()
    custom_aspects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ids", tuple(self.target_ids))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "aspects", tuple(self.aspects))
        object.__setattr__(self, "custom_aspects", tuple(self.custom_aspects))

    def snapshot(self) -> dict:
        value: dict[str, object] = {
            "type": self.type,
            "target_ids": list(self.target_ids),
        }
        if self.type in {"read", "verify"}:
            value["fields"] = list(self.fields)
        elif self.type in {"appreciate", "compare"}:
            value["aspects"] = list(self.aspects)
            value["custom_aspects"] = list(self.custom_aspects)
        return value


@dataclass(frozen=True)
class ResolvedRequest:
    """可由 AgentSession 长期持有的完整请求快照。"""

    targets: tuple[ResolvedTarget, ...]
    tasks: tuple[ResolvedTask, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "tasks", tuple(self.tasks))

    def snapshot(self) -> dict:
        return {
            "targets": [target.snapshot() for target in self.targets],
            "tasks": [task.snapshot() for task in self.tasks],
        }


def resolve_consolidated_request(
    request: ConsolidatedRequest,
    target_ids_by_ref: dict[str, int],
) -> ResolvedRequest:
    """把一次提交内的 refs 确定性映射为会话级 target IDs。"""
    if not isinstance(request, ConsolidatedRequest):
        raise TypeError("request 必须是 ConsolidatedRequest")
    expected_refs = {target.target_ref for target in request.targets}
    if set(target_ids_by_ref) != expected_refs:
        raise ValueError("target_ids_by_ref 必须精确覆盖请求中的 canonical refs")
    targets = tuple(
        ResolvedTarget(
            target_ids_by_ref[target.target_ref],
            target.author,
            target.dynasty,
            target.title,
            target.themes,
        )
        for target in request.targets
    )
    tasks = tuple(
        ResolvedTask(
            task.type,
            tuple(target_ids_by_ref[target_ref] for target_ref in task.target_refs),
            fields=task.fields,
            aspects=task.aspects,
            custom_aspects=task.custom_aspects,
        )
        for task in request.tasks
    )
    return ResolvedRequest(targets, tasks)


ProtocolErrorFactory = Callable[[str], ValueError]

