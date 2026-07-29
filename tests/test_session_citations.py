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


class FakeLLM:
    def __init__(self, decisions: list[dict]):
        self.decisions = iter(decisions)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(next(self.decisions), ensure_ascii=False)


class SessionCitationTests(unittest.TestCase):
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
                    "thought": "先检索",
                    "action": "search_poems",
                    "action_input": {"query": "静夜思"},
                },
                {
                    "thought": "取详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "poem-a"},
                },
                {
                    "thought": "作答",
                    "action": "finish",
                    "action_input": {"answer": "解读 [诗1-appr-0]"},
                },
            ]
        )

        with patch.dict(
            "poem_agent.agent.TOOLS",
            {
                "search_poems": lambda query: candidates,
                "get_poem_detail": lambda poem_id: poem,
            },
            clear=True,
        ):
            result = run_agent("赏析静夜思", llm)

        self.assertEqual(result["evidence"][0]["evidence_id"], "poem-a#appr-0")
        self.assertEqual(result["evidence"][0]["text"], "单诗赏析")
        self.assertEqual(result["evidence"][0]["poem_number"], 1)
        self.assertIn("《静夜思》李白 (poem_id=poem-a", llm.prompts[1])
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
                    "thought": "第一次检索",
                    "action": "search_poems",
                    "action_input": {"query": "甲诗"},
                },
                {
                    "thought": "第二次检索",
                    "action": "search_poems",
                    "action_input": {"query": "乙诗"},
                },
                {
                    "thought": "取第一次结果的详情",
                    "action": "get_poem_detail",
                    "action_input": {"poem_id": "poem-a"},
                },
                {
                    "thought": "作答",
                    "action": "finish",
                    "action_input": {"answer": "甲诗解读 [诗1-appr-0]"},
                },
            ]
        )

        with patch.dict(
            "poem_agent.agent.TOOLS",
            {
                "search_poems": search,
                "get_poem_detail": get_detail,
            },
            clear=True,
        ):
            result = run_agent("比较甲诗和乙诗", llm)

        self.assertEqual(received_poem_ids, ["poem-a"])
        self.assertEqual(result["evidence"][0]["evidence_id"], "poem-a#appr-0")

    def test_system_instruction_requires_real_poem_id_and_session_citation(self):
        self.assertIn("原样复制其 poem_id", SYSTEM_INSTRUCTION)
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
