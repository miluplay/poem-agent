"""单进程内的 Agent 会话状态、稳定编号和有限历史。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

from .analysis_support import LEVELS
from .candidate_pool import (
    VISIBLE_CANDIDATE_LIMIT,
    CandidatePool,
    SearchFunction,
)
from .request import ConsolidatedRequest, ResolvedRequest


MAX_HISTORY_ROUNDS = 8
MAX_HISTORY_CHARS = 16_000


class AgentSessionStateError(RuntimeError):
    """AgentSession 的生命周期或内部状态不一致。"""


@dataclass(frozen=True)
class HistoryTarget:
    target_id: int
    author: str | None
    dynasty: str | None
    title: str | None
    themes: tuple[str, ...]
    status: str
    detail_access_status: str
    loaded_candidate_ids: tuple[str, ...]

    def snapshot(self) -> dict:
        return {
            "target_id": self.target_id,
            "author": self.author,
            "dynasty": self.dynasty,
            "title": self.title,
            "themes": list(self.themes),
            "status": self.status,
            "detail_access_status": self.detail_access_status,
            "loaded_candidate_ids": list(self.loaded_candidate_ids),
        }


@dataclass(frozen=True)
class HistoryRound:
    user_query: str
    answer: str
    analysis_support_level: str
    degraded: bool
    targets: tuple[HistoryTarget, ...]

    def snapshot(self) -> dict:
        return {
            "user_query": self.user_query,
            "answer": self.answer,
            "analysis_support": {"level": self.analysis_support_level},
            "degraded": self.degraded,
            "targets": [target.snapshot() for target in self.targets],
        }


class AgentSession:
    """由调用方持有的纯内存会话状态。"""

    def __init__(
        self,
        *,
        max_history_rounds: int = MAX_HISTORY_ROUNDS,
        max_history_chars: int = MAX_HISTORY_CHARS,
    ) -> None:
        self._validate_history_limit(
            "max_history_rounds", max_history_rounds
        )
        self._validate_history_limit("max_history_chars", max_history_chars)
        self.max_history_rounds = max_history_rounds
        self.max_history_chars = max_history_chars
        self.consolidated_request: ResolvedRequest | None = None
        self.candidate_pool: CandidatePool | None = None
        self._session_poems: dict[int, str] = {}
        self._history: tuple[HistoryRound, ...] = ()

    @staticmethod
    def _validate_history_limit(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} 必须是非布尔正整数")

    @property
    def session_poems(self) -> dict[int, str]:
        """返回编号映射的防御性副本。"""
        return dict(self._session_poems)

    @property
    def history(self) -> tuple[HistoryRound, ...]:
        """返回不可变历史对象序列。"""
        return self._history

    @property
    def history_char_count(self) -> int:
        return sum(self._history_round_cost(item) for item in self._history)

    def initialize_request(
        self,
        request: ConsolidatedRequest,
        *,
        search_fn: SearchFunction | None = None,
        visible_limit: int = VISIBLE_CANDIDATE_LIMIT,
        baseline: dict | None = None,
    ) -> ResolvedRequest:
        """创建 Pool；仅在全部初始化成功后同时提交 Pool 和总请求。"""
        self._require_consistent_request_state()
        if self.candidate_pool is not None:
            raise AgentSessionStateError("会话请求已经初始化，不能重复初始化")
        pool, resolved = CandidatePool.initialize_request(
            request,
            search_fn=search_fn,
            visible_limit=visible_limit,
            baseline=baseline,
        )
        self.candidate_pool = pool
        self.consolidated_request = resolved
        return resolved

    def update_request(
        self, request: ConsolidatedRequest
    ) -> ResolvedRequest:
        """更新活 Pool；仅在 Pool 成功返回后替换当前总请求。"""
        self._require_consistent_request_state()
        if self.candidate_pool is None:
            raise AgentSessionStateError("会话请求尚未初始化，不能 update")
        resolved = self.candidate_pool.update_request(request)
        self.consolidated_request = resolved
        return resolved

    def request_snapshot(self) -> dict | None:
        if self.consolidated_request is None:
            return None
        return self.consolidated_request.snapshot()

    def assign_poem(self, poem_id: str) -> int:
        normalized = self._normalize_poem_id(poem_id)
        self._validate_session_poems()
        for poem_number, known_id in self._session_poems.items():
            if known_id == normalized:
                return poem_number
        poem_number = max(self._session_poems, default=0) + 1
        self._session_poems[poem_number] = normalized
        return poem_number

    def poem_number(self, poem_id: str) -> int | None:
        normalized = self._normalize_poem_id(poem_id)
        self._validate_session_poems()
        return next(
            (
                poem_number
                for poem_number, known_id in self._session_poems.items()
                if known_id == normalized
            ),
            None,
        )

    def session_poems_snapshot(self) -> dict[int, str]:
        self._validate_session_poems()
        return dict(self._session_poems)

    def add_detail(self, detail: dict) -> int:
        """把真实编号映射交给当前 Pool，供后续 Agent 接入使用。"""
        self._require_consistent_request_state()
        self._validate_session_poems()
        if self.candidate_pool is None:
            raise AgentSessionStateError("Candidate Pool 尚未初始化")
        maximum = max(self._session_poems, default=0)
        if set(self._session_poems) == set(range(1, maximum + 1)):
            return self.candidate_pool.add_detail(
                detail, self._session_poems
            )
        staged_poems = dict(self._session_poems)
        # CandidatePool 的 legacy 分配按 len + 1；仅在内部暂存映射中填补编号缺口，
        # 让 Session 即使面对合法缺口也始终使用历史最大编号 + 1。
        for poem_number in range(1, maximum + 1):
            staged_poems.setdefault(poem_number, object())
        poem_number = self.candidate_pool.add_detail(detail, staged_poems)
        poem_id = self.candidate_pool.loaded_details[
            detail["poem_id"]
        ]["poem_id"]
        known_number = self.poem_number(poem_id)
        if known_number is not None:
            return known_number
        self._session_poems[poem_number] = poem_id
        return poem_number

    def cached_poems_snapshot(self) -> list[dict]:
        self._validate_session_poems()
        if self.candidate_pool is None:
            return []
        values = []
        for poem_number, poem_id in sorted(self._session_poems.items()):
            detail = self.candidate_pool.loaded_details.get(poem_id)
            if detail is None:
                continue
            values.append(
                {
                    "poem_number": poem_number,
                    "poem_id": poem_id,
                    "title": detail.get("title"),
                    "author": detail.get("author"),
                    "dynasty": detail.get("dynasty"),
                }
            )
        return values

    def cached_detail(self, poem_id: str) -> dict | None:
        normalized = self._normalize_poem_id(poem_id)
        if self.candidate_pool is None:
            return None
        detail = self.candidate_pool.loaded_details.get(normalized)
        return deepcopy(detail) if detail is not None else None

    def append_history(self, user_query: str, result: dict) -> None:
        """原子追加一个完整问答轮次，再按轮数和字符预算裁剪。"""
        history_round = self._build_history_round(user_query, result)
        retained = [*self._history, history_round]
        while len(retained) > self.max_history_rounds:
            retained.pop(0)
        while (
            len(retained) > 1
            and sum(self._history_round_cost(item) for item in retained)
            > self.max_history_chars
        ):
            retained.pop(0)
        self._history = tuple(retained)

    def history_snapshot(self) -> list[dict]:
        return [item.snapshot() for item in self._history]

    @staticmethod
    def _history_round_cost(history_round: HistoryRound) -> int:
        return len(
            json.dumps(
                history_round.snapshot(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def _build_history_round(
        self, user_query: str, result: dict
    ) -> HistoryRound:
        if not isinstance(user_query, str):
            raise TypeError("user_query 必须是字符串")
        if not isinstance(result, dict):
            raise TypeError("result 必须是对象")
        answer = result.get("answer")
        if not isinstance(answer, str):
            raise ValueError("result.answer 必须是字符串")
        analysis_support = result.get("analysis_support")
        if not isinstance(analysis_support, dict):
            raise ValueError("result.analysis_support 必须是对象")
        level = analysis_support.get("level")
        if level not in LEVELS:
            raise ValueError("analysis_support.level 不是合法值")
        degraded = result.get("degraded")
        if not isinstance(degraded, bool):
            raise ValueError("result.degraded 必须是布尔值")
        target_results = (
            self.candidate_pool.target_results
            if self.candidate_pool is not None
            else []
        )
        targets = tuple(
            self._build_history_target(item) for item in target_results
        )
        return HistoryRound(user_query, answer, level, degraded, targets)

    @staticmethod
    def _build_history_target(item: dict) -> HistoryTarget:
        if not isinstance(item, dict):
            raise ValueError("Candidate Pool target result 必须是对象")
        required = {
            "target_id",
            "target",
            "status",
            "detail_access_status",
            "loaded_candidate_ids",
        }
        missing = required - set(item)
        if missing:
            raise ValueError(
                "Candidate Pool target result 缺少字段: "
                + "、".join(sorted(missing))
            )
        target_id = item["target_id"]
        if (
            isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or target_id <= 0
        ):
            raise ValueError("target_id 必须是非布尔正整数")
        target = item["target"]
        if not isinstance(target, dict):
            raise ValueError("target 必须是对象")
        target_fields = {
            "target_id",
            "author",
            "dynasty",
            "title",
            "themes",
        }
        missing_target_fields = target_fields - set(target)
        if missing_target_fields:
            raise ValueError(
                "target 缺少字段: "
                + "、".join(sorted(missing_target_fields))
            )
        if target["target_id"] != target_id:
            raise ValueError("target_id 与 target.target_id 不一致")
        for name in ("author", "dynasty", "title"):
            value = target[name]
            if value is not None and not isinstance(value, str):
                raise ValueError(f"target.{name} 必须是字符串或 None")
        themes = target["themes"]
        if not isinstance(themes, (list, tuple)) or any(
            not isinstance(theme, str) for theme in themes
        ):
            raise ValueError("target.themes 必须是字符串列表")
        status = item["status"]
        detail_access_status = item["detail_access_status"]
        if not isinstance(status, str) or not status:
            raise ValueError("status 必须是非空字符串")
        if (
            not isinstance(detail_access_status, str)
            or not detail_access_status
        ):
            raise ValueError("detail_access_status 必须是非空字符串")
        loaded_ids = item["loaded_candidate_ids"]
        if not isinstance(loaded_ids, list) or any(
            not isinstance(poem_id, str) for poem_id in loaded_ids
        ):
            raise ValueError("loaded_candidate_ids 必须是字符串列表")
        return HistoryTarget(
            target_id,
            target["author"],
            target["dynasty"],
            target["title"],
            tuple(themes),
            status,
            detail_access_status,
            tuple(loaded_ids),
        )

    def _require_consistent_request_state(self) -> None:
        if (self.candidate_pool is None) != (
            self.consolidated_request is None
        ):
            raise AgentSessionStateError(
                "Candidate Pool 与 consolidated request 状态不一致"
            )

    @staticmethod
    def _normalize_poem_id(poem_id: str) -> str:
        if not isinstance(poem_id, str):
            raise TypeError("poem_id 必须是字符串")
        normalized = poem_id.strip()
        if not normalized:
            raise ValueError("poem_id 必须是非空字符串")
        return normalized

    def _validate_session_poems(self) -> None:
        seen: set[str] = set()
        for poem_number, poem_id in self._session_poems.items():
            if (
                isinstance(poem_number, bool)
                or not isinstance(poem_number, int)
                or poem_number <= 0
            ):
                raise AgentSessionStateError("诗编号必须是非布尔正整数")
            if (
                not isinstance(poem_id, str)
                or not poem_id.strip()
                or poem_id != poem_id.strip()
            ):
                raise AgentSessionStateError(
                    "会话编号中的 poem_id 必须是规范化非空字符串"
                )
            if poem_id in seen:
                raise AgentSessionStateError(
                    "同一 poem_id 不能对应多个会话诗编号"
                )
            seen.add(poem_id)
