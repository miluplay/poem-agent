import json
import unittest
from unittest.mock import patch

from poem_agent.agent import (
    SYSTEM_INSTRUCTION,
    _assign_session_poem,
    _summarize_observation,
    run_agent,
)
from poem_agent.trust import collect_evidence


def detail(poem_id: str, title: str, text: str) -> dict:
    return {
        "poem_id": poem_id,
        "title": title,
        "author": "作者",
        "dynasty": "唐",
        "content": "正文",
        "appreciation": [
            {"evidence_id": f"{poem_id}#appr-0", "text": text}
        ],
        "annotations": [
            {"evidence_id": f"{poem_id}#anno-0", "text": f"{text}注"}
        ],
    }


def detail_step(observation: dict) -> dict:
    return {
        "action": "get_poem_detail",
        "input": {"poem_id": observation["poem_id"]},
        "observation": observation,
    }


def request_input(targets: list[dict], *, task_type="search") -> dict:
    normalized = [
        {
            "target_ref": f"t{index}",
            "author": target.get("author"),
            "dynasty": target.get("dynasty"),
            "title": target.get("title"),
            "themes": target.get("themes", []),
        }
        for index, target in enumerate(targets, start=1)
    ]
    task = {
        "type": task_type,
        "target_refs": [item["target_ref"] for item in normalized],
    }
    if task_type in {"appreciate", "compare"}:
        task.update({"aspects": [], "custom_aspects": []})
    return {"targets": normalized, "tasks": [task]}


class FakeLLM:
    def __init__(self, decisions: list[dict]):
        self.decisions = iter(decisions)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(next(self.decisions), ensure_ascii=False)


class SessionCitationTests(unittest.TestCase):
    def test_dangling_citation_regeneration_still_uses_real_evidence(self):
        poem = detail("poem-a", "静夜思", "真实赏析")
        llm = FakeLLM(
            [
                {
                    "thought": "初始化",
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(
                        [{"title": "静夜思"}], task_type="appreciate"
                    ),
                },
                {
                    "thought": "取详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "poem-a"},
                },
                {
                    "thought": "首次作答",
                    "action": "finish",
                    "action_input": {
                        "answer": "这是一段带悬空引用的完整解读 [诗1-appr-99]",
                        "analysis_assessment": {
                            "level": "sufficient",
                            "target_ids": [1],
                        },
                    },
                },
                {
                    "thought": "修正引用",
                    "action": "finish",
                    "action_input": {
                        "answer": "这是一段已修正引用的完整解读 [诗1-appr-0]",
                        "analysis_assessment": {
                            "level": "sufficient",
                            "target_ids": [1],
                        },
                    },
                },
            ]
        )
        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=[
                    {
                        "poem_id": "poem-a",
                        "title": "静夜思",
                        "author": "李白",
                        "dynasty": "唐",
                        "score": None,
                    }
                ],
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: poem},
                clear=True,
            ),
        ):
            result = run_agent("赏析静夜思", llm)

        self.assertFalse(result["degraded"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "poem-a#appr-0")
        self.assertIn("悬空引用", llm.prompts[-1])

    def test_single_poem_run_keeps_evidence_binding(self):
        poem = detail("poem-a", "静夜思", "单诗赏析")
        candidates = [
            {
                "poem_id": "poem-a",
                "title": "静夜思",
                "author": "李白",
                "dynasty": "唐",
                "score": 1.0,
            }
        ]
        llm = FakeLLM(
            [
                {
                    "thought": "先初始化候选池",
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(
                        [{"title": "静夜思"}], task_type="appreciate"
                    ),
                },
                {
                    "thought": "取详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "poem-a"},
                },
                {
                    "thought": "作答",
                    "action": "finish",
                    "action_input": {
                        "answer": "这是完整解读 [诗1-appr-0]",
                        "analysis_assessment": {
                            "level": "sufficient",
                            "target_ids": [1],
                        },
                    },
                },
            ]
        )

        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                return_value=candidates,
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": lambda poem_id: poem},
                clear=True,
            ),
        ):
            result = run_agent("赏析静夜思", llm)

        self.assertEqual(result["evidence"][0]["evidence_id"], "poem-a#appr-0")
        self.assertEqual(result["evidence"][0]["text"], "单诗赏析")
        self.assertEqual(result["evidence"][0]["poem_number"], 1)
        self.assertIn('"visible_candidate_ids": ["poem-a"]', llm.prompts[1])
        self.assertIn("【诗1】", llm.prompts[2])
        self.assertIn("[诗1-appr-0]", llm.prompts[2])
        self.assertIn("'poem_id': 'poem-a'", llm.prompts[2])

    def test_real_poem_id_survives_multiple_searches_without_remapping(self):
        first_candidates = [
            {
                "poem_id": "poem-a",
                "title": "甲诗",
                "author": "甲",
                "dynasty": "唐",
                "score": 1.0,
            }
        ]
        second_candidates = [
            {
                "poem_id": "poem-b",
                "title": "乙诗",
                "author": "乙",
                "dynasty": "唐",
                "score": 1.0,
            }
        ]
        received_poem_ids: list[str] = []

        def get_detail(poem_id: str) -> dict:
            received_poem_ids.append(poem_id)
            return detail(poem_id, "甲诗", "甲赏析")

        def search(query: str) -> list[dict]:
            return first_candidates if query == "甲诗" else second_candidates

        llm = FakeLLM(
            [
                {
                    "thought": "一次初始化两个目标",
                    "action": "initialize_candidate_pool",
                    "action_input": request_input(
                        [{"title": "甲诗"}, {"title": "乙诗"}],
                        task_type="compare",
                    ),
                },
                {
                    "thought": "取第一次结果的详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "poem-a"},
                },
                {
                    "thought": "作答",
                    "action": "finish",
                    "action_input": {
                        "answer": "这是甲诗解读 [诗1-appr-0]",
                        "analysis_assessment": {
                            "level": "insufficient",
                            "target_ids": [1],
                        },
                    },
                },
            ]
        )

        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                side_effect=lambda **query: search(query["title"]),
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": get_detail},
                clear=True,
            ),
        ):
            result = run_agent("比较甲诗和乙诗", llm)

        self.assertEqual(received_poem_ids, ["poem-a"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "poem-a#appr-0")

    def test_system_instruction_requires_real_poem_id_and_session_citation(self):
        self.assertIn("poem_id 必须原样复制自池快照", SYSTEM_INSTRUCTION)
        self.assertIn("[诗1-appr-0]", SYSTEM_INSTRUCTION)
        self.assertNotIn("候选N", SYSTEM_INSTRUCTION)
        self.assertNotIn("(如 [appr-0]", SYSTEM_INSTRUCTION)

    def test_two_poems_with_same_short_id_bind_independently(self):
        first = detail("poem-a", "甲诗", "甲证据")
        second = detail("poem-b", "乙诗", "乙证据")
        evidence = collect_evidence(
            "甲 [诗1-appr-0]，乙 [诗2-appr-0]",
            [detail_step(first), detail_step(second)],
            {1: "poem-a", 2: "poem-b"},
        )

        self.assertEqual(
            [(item["evidence_id"], item["text"]) for item in evidence],
            [
                ("poem-a#appr-0", "甲证据"),
                ("poem-b#appr-0", "乙证据"),
            ],
        )

    def test_both_dangling_reasons_are_distinguished(self):
        poem = detail("poem-a", "甲诗", "甲证据")
        evidence = collect_evidence(
            "未取详情 [诗2-appr-0]；错段 [诗1-appr-99]",
            [detail_step(poem)],
            {1: "poem-a"},
        )

        self.assertEqual(evidence[0]["reason"], "引用了未取详情的诗")
        self.assertEqual(evidence[1]["reason"], "段编号不存在")
        self.assertTrue(evidence[0]["dangling"])
        self.assertTrue(evidence[1]["dangling"])

    def test_cached_details_bind_without_current_detail_step(self):
        poem = detail("poem-a", "甲诗", "缓存证据")
        evidence = collect_evidence(
            "赏析 [诗1-appr-0]，注释 [诗1-note-0]",
            [],
            {1: "poem-a"},
            cached_details={"poem-a": poem},
        )
        self.assertEqual(
            [(item["evidence_id"], item["text"]) for item in evidence],
            [
                ("poem-a#appr-0", "缓存证据"),
                ("poem-a#anno-0", "缓存证据注"),
            ],
        )
        self.assertTrue(all("dangling" not in item for item in evidence))

    def test_unnumbered_cache_and_missing_cached_segment_stay_dangling(self):
        poem = detail("poem-a", "甲诗", "缓存证据")
        evidence = collect_evidence(
            "未编号 [诗2-appr-0]；错段 [诗1-appr-9]",
            [],
            {1: "poem-a"},
            cached_details={"poem-a": poem},
        )
        self.assertEqual(
            [item["reason"] for item in evidence],
            ["引用了未取详情的诗", "段编号不存在"],
        )

    def test_trajectory_overrides_cache_and_citations_are_not_duplicated(self):
        cached = detail("poem-a", "甲诗", "旧缓存")
        current = detail("poem-a", "甲诗", "本轮证据")
        evidence = collect_evidence(
            "先 [诗1-appr-0] 再重复 [诗1-appr-0]，后 [诗1-note-0]",
            [detail_step(current)],
            {1: "poem-a"},
            cached_details={"poem-a": cached},
        )
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["text"], "本轮证据")
        self.assertEqual(evidence[1]["text"], "本轮证据注")

    def test_invalid_cached_entries_are_ignored_deterministically(self):
        valid = detail("poem-a", "甲诗", "有效缓存")
        invalid_cases = (
            {"wrong-key": valid},
            {"poem-a": {**valid, "error": "not_found"}},
            {"poem-a": {**valid, "appreciation": "bad"}},
            {
                "poem-a": {
                    **valid,
                    "appreciation": [
                        {"evidence_id": "other#appr-0", "text": "坏"}
                    ],
                }
            },
        )
        for cached_details in invalid_cases:
            with self.subTest(cached_details=cached_details):
                evidence = collect_evidence(
                    "引用 [诗1-appr-0]",
                    [],
                    {1: "poem-a"},
                    cached_details=cached_details,
                )
                self.assertTrue(evidence[0]["dangling"])
                self.assertEqual(evidence[0]["reason"], "段编号不存在")
        with self.assertRaises(TypeError):
            collect_evidence(
                "引用 [诗1-appr-0]",
                [],
                {1: "poem-a"},
                cached_details=[],
            )

    def test_session_number_is_reused_for_same_poem(self):
        session_poems: dict[int, str] = {}
        self.assertEqual(_assign_session_poem(session_poems, "poem-a"), 1)
        self.assertEqual(_assign_session_poem(session_poems, "poem-a"), 1)
        self.assertEqual(_assign_session_poem(session_poems, "poem-b"), 2)
        self.assertEqual(session_poems, {1: "poem-a", 2: "poem-b"})

    def test_search_summary_exposes_real_poem_ids(self):
        summary = _summarize_observation(
            [
                {
                    "poem_id": "long-poem-id",
                    "title": "静夜思",
                    "author": "李白",
                    "score": 0.74,
                }
            ]
        )
        self.assertEqual(
            summary,
            "检索到 1 首候选:\n"
            "《静夜思》李白 (poem_id=long-poem-id, score=0.74)",
        )
        self.assertNotIn("候选1", summary)


if __name__ == "__main__":
    unittest.main()
