import json
import unittest
from unittest.mock import patch

from poem_agent.agent import run_agent
from poem_agent.agent.prompts import SYSTEM_INSTRUCTION
from poem_agent.retrieval import retrieve_all_poems
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

    def test_internal_full_retrieval_is_not_limited_by_public_top_k(self):
        poems = [poem(f"p{i}", f"诗{i}", author="甲") for i in range(7)]
        with patch(
            "poem_agent.retrieval.engine.store.load_poems",
            return_value=poems,
        ):
            public = search_poems(author="甲")
            full = retrieve_all_poems(author="甲")

        self.assertEqual(len(public), 5)
        self.assertEqual(len(full), 7)


class PromptAndAgentRegressionTests(unittest.TestCase):
    def test_prompt_requires_one_pool_initialization_and_target_pairing(self):
        self.assertIn("先且只能成功调用一次", SYSTEM_INSTRUCTION)
        self.assertIn("全部 1–4 个 targets", SYSTEM_INSTRUCTION)
        self.assertIn("不能直接调用 search_poems", SYSTEM_INSTRUCTION)
        self.assertIn("必须拆成两个正确配对的 targets", SYSTEM_INSTRUCTION)
        self.assertIn("theme_coverage 在阶段 1 固定为 null", SYSTEM_INSTRUCTION)
        self.assertIn("候选 score", SYSTEM_INSTRUCTION)
        self.assertNotIn("normal", SYSTEM_INSTRUCTION)
        self.assertNotIn("low_conf", SYSTEM_INSTRUCTION)
        self.assertNotIn("no_hit", SYSTEM_INSTRUCTION)

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
        detail_calls: list[str] = []

        def fake_search(**conditions) -> list[dict]:
            return [] if conditions.get("author") == "李白" else [candidate]

        def fake_detail(poem_id: str) -> dict:
            detail_calls.append(poem_id)
            return detail

        llm = FakeLLM(
            [
                decision(
                    "一次初始化并由系统诊断",
                    "initialize_candidate_pool",
                    {
                        "targets": [
                            {
                                "author": "李白",
                                "title": "春望",
                                "dynasty": None,
                                "themes": [],
                            }
                        ]
                    },
                ),
                decision(
                    "读取详情后再纠正",
                    "get_poem_detail",
                    {"poem_id": "spring-view"},
                ),
                decision(
                    "依据详情纠正作者",
                    "finish",
                    {
                        "answer": "《春望》的作者是杜甫，并非李白。",
                        "analysis_assessment": {
                            "level": "not_applicable",
                            "target_ids": [],
                        },
                    },
                ),
            ]
        )

        with (
            patch(
                "poem_agent.candidate_pool.retrieve_all_poems",
                side_effect=fake_search,
            ),
            patch.dict(
                "poem_agent.agent.TOOLS",
                {"get_poem_detail": fake_detail},
                clear=True,
            ),
        ):
            result = run_agent("请赏析李白的《春望》", llm)

        self.assertEqual(detail_calls, ["spring-view"])
        self.assertEqual(result["answer"], "《春望》的作者是杜甫，并非李白。")
        self.assertEqual(
            result["candidate_pool"]["profile"]["target_results"][0]["status"],
            "conflict",
        )
        self.assertIn('"status": "conflict"', llm.prompts[1])


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
