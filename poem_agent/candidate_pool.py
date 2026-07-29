"""一次 Agent 运行内持有的 Candidate Pool 阶段 1 状态。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

from . import store
from .retrieval import retrieve_all_poems


THEME_SEPARATOR = "；"
VISIBLE_CANDIDATE_LIMIT = 5
_TARGET_FIELDS = frozenset({"author", "dynasty", "title", "themes"})


class CandidatePoolProtocolError(ValueError):
    """模型提交的 targets 不符合 Candidate Pool 协议。"""


@dataclass(frozen=True)
class Target:
    target_id: int
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]

    def snapshot(self) -> dict:
        return {
            "target_id": self.target_id,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
        }


@dataclass
class QueryTask:
    target_id: int
    kind: str
    query: dict
    status: str = "pending"
    candidate_ids: list[str] = field(default_factory=list)


SearchFunction = Callable[..., list[dict]]


class CandidatePool:
    """保存规范化 targets、查询任务、完整候选、画像与固定 verdict。"""

    def __init__(
        self,
        targets: list[Target],
        *,
        search_fn: SearchFunction | None = None,
        visible_limit: int = VISIBLE_CANDIDATE_LIMIT,
    ) -> None:
        self.targets = tuple(targets)
        self.visible_limit = visible_limit
        self.main_tasks: dict[int, QueryTask] = {}
        self.diagnostic_tasks: dict[int, QueryTask] = {}
        self.candidates: dict[str, dict] = {}
        self.target_candidate_ids: dict[int, list[str]] = {
            target.target_id: [] for target in targets
        }
        self.target_results: list[dict] = []
        self.profile: dict = {}
        self.verdict = ""
        self._compile_tasks()
        self._execute(search_fn or retrieve_all_poems)
        self._build_profile()

    @classmethod
    def initialize(
        cls,
        raw_targets,
        *,
        search_fn: SearchFunction | None = None,
        visible_limit: int = VISIBLE_CANDIDATE_LIMIT,
    ) -> "CandidatePool":
        """先完整校验并规范化，再创建并执行池，避免协议错误部分初始化。"""
        targets = normalize_targets(raw_targets)
        return cls(
            targets,
            search_fn=search_fn or retrieve_all_poems,
            visible_limit=visible_limit,
        )

    def _compile_tasks(self) -> None:
        for target in self.targets:
            query = _main_query(target)
            self.main_tasks[target.target_id] = QueryTask(
                target_id=target.target_id,
                kind="main",
                query=query,
            )
            diagnostic = _diagnostic_query(target)
            if diagnostic is not None:
                self.diagnostic_tasks[target.target_id] = QueryTask(
                    target_id=target.target_id,
                    kind="diagnostic",
                    query=diagnostic,
                )

    def _execute(self, search_fn: SearchFunction) -> None:
        for target in self.targets:
            main_task = self.main_tasks[target.target_id]
            main_results = self._run_task(main_task, search_fn)
            diagnostic_results: list[dict] = []
            diagnostic_task = self.diagnostic_tasks.get(target.target_id)
            if diagnostic_task is not None:
                if main_results:
                    diagnostic_task.status = "skipped"
                else:
                    diagnostic_results = self._run_task(
                        diagnostic_task, search_fn
                    )
            self._associate(
                target.target_id, [*main_results, *diagnostic_results]
            )

    def _run_task(
        self, task: QueryTask, search_fn: SearchFunction
    ) -> list[dict]:
        try:
            results = search_fn(**task.query)
        except Exception:
            task.status = "failed"
            raise
        task.status = "completed"
        task.candidate_ids = [item["poem_id"] for item in results]
        for item in results:
            self.candidates.setdefault(item["poem_id"], dict(item))
        return results

    def _associate(self, target_id: int, results: list[dict]) -> None:
        associated = self.target_candidate_ids[target_id]
        for item in results:
            poem_id = item["poem_id"]
            if poem_id not in associated:
                associated.append(poem_id)

    def _build_profile(self) -> None:
        self.target_results = [
            self._target_result(target) for target in self.targets
        ]
        author_dist: dict[str, int] = {}
        for candidate in self.candidates.values():
            author = candidate.get("author")
            if isinstance(author, str):
                author_dist[author] = author_dist.get(author, 0) + 1
        self.profile = {
            "size": len(self.candidates),
            "author_dist": author_dist,
            "target_results": self.target_results,
            "theme_coverage": None,
        }
        self.verdict = _build_verdict(self.target_results)

    def _target_result(self, target: Target) -> dict:
        main_ids = self.main_tasks[target.target_id].candidate_ids
        diagnostic_task = self.diagnostic_tasks.get(target.target_id)
        diagnostic_ids = (
            diagnostic_task.candidate_ids if diagnostic_task is not None else []
        )
        status, basis = _target_status(
            target,
            main_ids,
            diagnostic_ids,
            self.candidates,
        )
        candidate_ids = self.target_candidate_ids[target.target_id]
        return {
            "target_id": target.target_id,
            "target": target.snapshot(),
            "status": status,
            "retrieval": "found" if candidate_ids else "empty",
            "candidate_count": len(candidate_ids),
            "visible_candidate_ids": candidate_ids[: self.visible_limit],
            "basis": basis,
            "theme_coverage": None,
        }

    def model_snapshot(self) -> dict:
        """返回适合写入 Prompt 的固定大小精简快照。"""
        snapshot = self.public_snapshot()
        for target_result in snapshot["profile"]["target_results"]:
            target_result["visible_candidates"] = [
                deepcopy(self.candidates[poem_id])
                for poem_id in target_result["visible_candidate_ids"]
            ]
        return snapshot

    def public_snapshot(self) -> dict:
        """返回最终公开结果；不复制完整候选正文或无限候选列表。"""
        return {
            "targets": [target.snapshot() for target in self.targets],
            "profile": {
                "size": self.profile["size"],
                "author_dist": dict(self.profile["author_dist"]),
                "target_results": deepcopy(self.target_results),
                "theme_coverage": None,
            },
            "verdict": self.verdict,
        }


def normalize_targets(raw_targets) -> list[Target]:
    if not isinstance(raw_targets, list):
        raise CandidatePoolProtocolError("targets 必须是列表")

    normalized: list[tuple[str | None, str | None, str | None, tuple[str, ...]]] = []
    seen: set[tuple] = set()
    for index, raw_target in enumerate(raw_targets, start=1):
        if not isinstance(raw_target, dict):
            raise CandidatePoolProtocolError(
                f"targets[{index}] 必须是对象"
            )
        unknown = set(raw_target) - _TARGET_FIELDS
        if unknown:
            names = "、".join(sorted(unknown))
            raise CandidatePoolProtocolError(
                f"targets[{index}] 包含未知字段: {names}"
            )
        author = _optional_string(
            f"targets[{index}].author", raw_target.get("author")
        )
        dynasty = _optional_string(
            f"targets[{index}].dynasty", raw_target.get("dynasty")
        )
        raw_title = _optional_string(
            f"targets[{index}].title", raw_target.get("title")
        )
        title = (store._normalize_title(raw_title) or None) if raw_title else None
        themes = _normalize_themes(
            raw_target.get("themes", []), target_index=index
        )
        if author is None and dynasty is None and title is None and not themes:
            raise CandidatePoolProtocolError(
                f"targets[{index}] 至少需要 author、dynasty、title 或一个 theme"
            )
        value = (author, dynasty, title, themes)
        if value not in seen:
            seen.add(value)
            normalized.append(value)

    if not 1 <= len(normalized) <= 4:
        raise CandidatePoolProtocolError(
            "targets 规范化去重后必须有 1–4 个，不能静默截断"
        )
    return [
        Target(index, author, dynasty, title, themes)
        for index, (author, dynasty, title, themes) in enumerate(
            normalized, start=1
        )
    ]


def _optional_string(name: str, value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise CandidatePoolProtocolError(f"{name} 必须是字符串或 None")
    return value.strip() or None


def _normalize_themes(raw_themes, *, target_index: int) -> tuple[str, ...]:
    if not isinstance(raw_themes, list):
        raise CandidatePoolProtocolError(
            f"targets[{target_index}].themes 必须是字符串列表"
        )
    themes: list[str] = []
    for theme_index, theme in enumerate(raw_themes, start=1):
        if not isinstance(theme, str) or not theme.strip():
            raise CandidatePoolProtocolError(
                f"targets[{target_index}].themes[{theme_index}] "
                "必须是非空字符串"
            )
        normalized = theme.strip()
        if normalized not in themes:
            themes.append(normalized)
    return tuple(themes)


def _main_query(target: Target) -> dict:
    return {
        "query": THEME_SEPARATOR.join(target.themes) or None,
        "author": target.author,
        "dynasty": target.dynasty,
        "title": target.title,
    }


def _diagnostic_query(target: Target) -> dict | None:
    query = THEME_SEPARATOR.join(target.themes) or None
    if target.title is not None and (
        target.author is not None or target.dynasty is not None
    ):
        return {
            "query": query,
            "author": None,
            "dynasty": None,
            "title": target.title,
        }
    if (
        target.title is None
        and target.author is not None
        and target.dynasty is not None
    ):
        return {
            "query": query,
            "author": target.author,
            "dynasty": None,
            "title": None,
        }
    return None


def _target_status(
    target: Target,
    main_ids: list[str],
    diagnostic_ids: list[str],
    candidates: dict[str, dict],
) -> tuple[str, dict | None]:
    hard_condition = any((target.author, target.dynasty, target.title))
    if not hard_condition:
        return "not_applicable", None

    if main_ids:
        strict_ids = [
            poem_id
            for poem_id in main_ids
            if _strictly_matches(target, candidates[poem_id])
        ]
        if strict_ids:
            return "matched", None
        if target.title is not None:
            return (
                "partial_match",
                {
                    "requested_title": target.title,
                    "partial_title_candidate_ids": list(main_ids),
                },
            )
        return "missing", None

    if diagnostic_ids:
        conflicts = []
        for poem_id in diagnostic_ids:
            item = candidates[poem_id]
            differences = {}
            for field_name in ("author", "dynasty"):
                expected = getattr(target, field_name)
                actual = item.get(field_name)
                if expected is not None and actual != expected:
                    differences[field_name] = {
                        "expected": expected,
                        "actual": actual,
                    }
            conflicts.append(
                {"poem_id": poem_id, "differences": differences}
            )
        return "conflict", {"diagnostic_conflicts": conflicts}

    return "missing", None


def _strictly_matches(target: Target, candidate: dict) -> bool:
    return (
        (target.author is None or candidate.get("author") == target.author)
        and (
            target.dynasty is None
            or candidate.get("dynasty") == target.dynasty
        )
        and (
            target.title is None
            or (
                isinstance(candidate.get("title"), str)
                and store._normalize_title(candidate["title"]) == target.title
            )
        )
    )


def _build_verdict(target_results: list[dict]) -> str:
    statuses = [item["status"] for item in target_results]
    if "conflict" in statuses:
        return (
            "请求不符：条件诊断发现一个或多个 target 的作者或朝代"
            "与语料不一致。"
        )
    if "partial_match" in statuses and "missing" in statuses:
        return (
            "部分满足：存在标题部分匹配 target，另有 target 未命中。"
        )
    if "partial_match" in statuses:
        return "标题部分匹配：至少一个 target 仅找到标题部分匹配候选。"
    if all(status == "missing" for status in statuses):
        return "未命中：所有 target 在允许的检索与诊断路径中均无结果。"
    if "missing" in statuses:
        return "部分满足：部分 target 已命中，另有 target 未命中。"
    if all(status == "not_applicable" for status in statuses):
        if any(item["retrieval"] == "found" for item in target_results):
            return "已取得主题排序候选，但主题覆盖待评估。"
        return "未命中：主题检索未取得候选，主题覆盖仍待评估。"
    if "not_applicable" in statuses:
        return "部分满足：结构化 target 已命中，主题覆盖仍待评估。"
    return "全部命中：所有 target 的结构化条件均得到严格匹配。"
