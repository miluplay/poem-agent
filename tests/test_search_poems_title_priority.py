import json
import unittest
from unittest.mock import patch

from poem_agent.agent import run_agent
from poem_agent.agent.prompts import SYSTEM_INSTRUCTION
from poem_agent.tools import search_poems


def poem(
    poem_id: str,
    title: str,
    *,
    author: str = "作者",
    dynasty: str = "唐",
    tags: list[str] | None = None,
) -> dict:
    return {
        "poem_id": poem_id,
        "title": title,
        "author": author,
        "dynasty": dynasty,
        "tags": tags or [],
    }


class SearchPoemsContractTests(unittest.TestCase):
    def search_with_poems(self, poems: list[dict], **conditions) -> list[dict]:
        with patch(
            "poem_agent.retrieval.engine.store.load_poems",
            return_value=poems,
        ):
            return search_poems(**conditions)

    def test_single_structured_conditions_filter_exactly(self):
        poems = [
            poem("dufu-tang", "春望", author="杜甫", dynasty="唐"),
            poem("libai-tang", "静夜思", author="李白", dynasty="唐"),
            poem("dufu-song", "春望", author="杜甫", dynasty="宋"),
        ]

        cases = [
            ({"author": " 杜甫 "}, ["dufu-tang", "dufu-song"]),
            ({"dynasty": " 唐 "}, ["dufu-tang", "libai-tang"]),
            ({"title": " 春望 "}, ["dufu-tang", "dufu-song"]),
        ]
        for conditions, expected_ids in cases:
            with self.subTest(conditions=conditions):
                results = self.search_with_poems(poems, **conditions)
                self.assertEqual(
                    [item["poem_id"] for item in results],
                    expected_ids,
                )

    def test_hard_conditions_intersect_without_silent_relaxation(self):
        poems = [
            poem("spring-view", "春望", author="杜甫", dynasty="唐"),
            poem("partial-by-wrong-author", "春望怀古", author="李白", dynasty="唐"),
            poem("quiet-night", "静夜思", author="李白", dynasty="唐"),
        ]

        matched = self.search_with_poems(
            poems,
            author="杜甫",
            dynasty="唐",
            title="春望",
        )
        conflicted = self.search_with_poems(
            poems,
            author="李白",
            title="春望",
        )

        self.assertEqual([item["poem_id"] for item in matched], ["spring-view"])
        self.assertEqual(conflicted, [])

    def test_exact_titles_exclude_partial_matches_and_keep_all_duplicates(self):
        poems = [
            poem("exact-a", "月夜", author="甲"),
            poem("partial", "月夜忆舍弟", author="杜甫"),
            poem("exact-b", "《 月 夜 》", author="乙"),
        ]

        results = self.search_with_poems(poems, title="〈月夜〉")

        self.assertEqual(
            [item["poem_id"] for item in results],
            ["exact-a", "exact-b"],
        )
        self.assertTrue(all(item["score"] is None for item in results))

    def test_partial_title_matching_is_only_fallback(self):
        poems = [
            poem("first", "月夜忆舍弟"),
            poem("unrelated", "春望"),
            poem("second", "江楼月夜闻笛"),
        ]

        results = self.search_with_poems(poems, title="月夜")

        self.assertEqual(
            [item["poem_id"] for item in results],
            ["first", "second"],
        )

    def test_no_query_keeps_corpus_order_none_scores_and_skips_semantics(self):
        poems = [
            poem("first", "甲", author="杜甫"),
            poem("other", "乙", author="李白"),
            poem("second", "丙", author="杜甫"),
        ]

        with (
            patch(
                "poem_agent.retrieval.engine.store.load_poems",
                return_value=poems,
            ),
            patch(
                "poem_agent.retrieval.engine._semantic_scores"
            ) as semantic_scores,
        ):
            results = search_poems(author="杜甫")

        semantic_scores.assert_not_called()
        self.assertEqual(
            [item["poem_id"] for item in results],
            ["first", "second"],
        )
        self.assertEqual([item["score"] for item in results], [None, None])

    def test_query_scores_and_sorts_only_inside_hard_filtered_pool(self):
        poems = [
            poem("dufu-low", "甲", author="杜甫"),
            poem("libai-high", "乙", author="李白"),
            poem("dufu-high", "丙", author="杜甫"),
        ]
        semantic = {"dufu-low": 0.2, "dufu-high": 0.9}

        with (
            patch(
                "poem_agent.retrieval.engine.store.load_poems",
                return_value=poems,
            ),
            patch(
                "poem_agent.retrieval.engine._semantic_scores",
                return_value=semantic,
            ) as semantic_scores,
            patch(
                "poem_agent.retrieval.engine._extract_query_tags",
                return_value=set(),
            ),
        ):
            results = search_poems(query="月亮", author="杜甫")

        semantic_scores.assert_called_once_with(
            "月亮",
            frozenset({"dufu-low", "dufu-high"}),
        )
        self.assertEqual(
            [item["poem_id"] for item in results],
            ["dufu-high", "dufu-low"],
        )
        self.assertAlmostEqual(results[0]["score"], 0.54)
        self.assertAlmostEqual(results[1]["score"], 0.12)

    def test_query_is_not_parsed_as_author_title_or_dynasty(self):
        poems = [
            poem("spring-view", "春望", author="杜甫", dynasty="唐"),
            poem("quiet-night", "静夜思", author="李白", dynasty="唐"),
        ]

        with (
            patch(
                "poem_agent.retrieval.engine.store.load_poems",
                return_value=poems,
            ),
            patch(
                "poem_agent.retrieval.engine._semantic_scores",
                return_value={"quiet-night": 0.9, "spring-view": 0.1},
            ),
            patch(
                "poem_agent.retrieval.engine._extract_query_tags",
                return_value=set(),
            ),
        ):
            results = search_poems(query="李白的春望", top_k=2)

        self.assertEqual(
            [item["poem_id"] for item in results],
            ["quiet-night", "spring-view"],
        )

    def test_optional_strings_are_trimmed_and_blank_is_absent(self):
        poems = [poem("spring-view", "春望", author="杜甫")]

        results = self.search_with_poems(
            poems,
            query=" ",
            author=" 杜甫 ",
            dynasty=" ",
            title=None,
        )

        self.assertEqual([item["poem_id"] for item in results], ["spring-view"])
        self.assertIsNone(results[0]["score"])

    def test_empty_conditions_and_invalid_optional_types_raise(self):
        for conditions in (
            {},
            {"query": " ", "author": "\t", "title": "\n"},
            {"author": 123},
            {"query": ["月亮"]},
        ):
            with self.subTest(conditions=conditions):
                with self.assertRaisesRegex(ValueError, "检索条件|字符串"):
                    search_poems(**conditions)

    def test_top_k_accepts_only_non_boolean_integers_from_one_to_twenty(self):
        poems = [poem("one", "甲")]
        for top_k in (True, False, 0, -1, 21, 1.5, "5", None):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "1–20"):
                    search_poems(author="作者", top_k=top_k)

        for top_k in (1, 20):
            with self.subTest(top_k=top_k):
                results = self.search_with_poems(
                    poems,
                    author="作者",
                    top_k=top_k,
                )
                self.assertEqual(len(results), 1)

    def test_candidate_field_set_is_exact(self):
        results = self.search_with_poems(
            [poem("one", "甲")],
            author="作者",
        )

        self.assertEqual(
            set(results[0]),
            {"poem_id", "title", "author", "dynasty", "score"},
        )
        self.assertNotIn("matched_by", results[0])


class PromptAndAgentRegressionTests(unittest.TestCase):
    def test_prompt_defines_structured_responsibility_and_relaxation_order(self):
        self.assertIn(
            "不会从 query 识别、猜取或补全作者、朝代、\n  标题",
            SYSTEM_INSTRUCTION,
        )
        self.assertIn("有 title 时优先保留 title", SYSTEM_INSTRUCTION)
        self.assertIn("移除 author 和 dynasty", SYSTEM_INSTRUCTION)
        self.assertIn("优先保留 author,移除 dynasty", SYSTEM_INSTRUCTION)
        self.assertIn("放宽结果只用于识别冲突", SYSTEM_INSTRUCTION)
        self.assertIn("仍须调用\n   get_poem_detail", SYSTEM_INSTRUCTION)

    def test_agent_diagnoses_wrong_author_then_corrects_from_detail(self):
        candidate = {
            "poem_id": "spring-view",
            "title": "春望",
            "author": "杜甫",
            "dynasty": "唐",
            "score": None,
        }
        detail = {
            **candidate,
            "content": "国破山河在",
            "appreciation": [],
            "annotations": [],
        }
        search_calls: list[dict] = []
        detail_calls: list[str] = []

        def fake_search(**conditions) -> list[dict]:
            search_calls.append(conditions)
            return [] if conditions.get("author") == "李白" else [candidate]

        def fake_detail(poem_id: str) -> dict:
            detail_calls.append(poem_id)
            return detail

        llm = FakeLLM(
            [
                decision(
                    "先严格核对作者与标题",
                    "search_poems",
                    {"author": "李白", "title": "春望"},
                ),
                decision(
                    "组合条件为空，保留标题诊断冲突",
                    "search_poems",
                    {"title": "春望"},
                ),
                decision(
                    "读取详情后再纠正",
                    "get_poem_detail",
                    {"poem_id": "spring-view"},
                ),
                decision(
                    "依据详情纠正作者",
                    "finish",
                    {"answer": "《春望》的作者是杜甫，并非李白。"},
                ),
            ]
        )

        with patch.dict(
            "poem_agent.agent.TOOLS",
            {
                "search_poems": fake_search,
                "get_poem_detail": fake_detail,
            },
            clear=True,
        ):
            result = run_agent("请赏析李白的《春望》", llm)

        self.assertEqual(
            search_calls,
            [
                {"author": "李白", "title": "春望"},
                {"title": "春望"},
            ],
        )
        self.assertEqual(detail_calls, ["spring-view"])
        self.assertEqual(result["answer"], "《春望》的作者是杜甫，并非李白。")
        self.assertIn(
            "调用 search_poems({'author': '李白', 'title': '春望'})",
            llm.prompts[2],
        )
        self.assertIn(
            "调用 search_poems({'title': '春望'})",
            llm.prompts[2],
        )


def decision(thought: str, action: str, action_input: dict) -> dict:
    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
    }


class FakeLLM:
    def __init__(self, decisions: list[dict]):
        self.decisions = iter(decisions)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(next(self.decisions), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
