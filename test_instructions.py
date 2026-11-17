import sys
import tempfile
import unittest
from pathlib import Path

from language_model_evaluation import definition, instructions


def prompt(key, text):
    return {"key": key, "prompt": text, "instruction_id_list": ["length_constraints:number_sentences"],
            "kwargs": [{"relation": "at least", "num_sentences": 3}], "lang": "ja"}


# The authors' checks as a module of their form: a reply follows its prompt at the strict level when the first number
# drawn after the seed is above 0.8, as it is only when the seed is set before each reply, and at the loose level when
# it was asked in Japanese.
CHECKS = """
import dataclasses, random

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list
    prompt: str
    kwargs: list
    lang: str

@dataclasses.dataclass
class OutputExample:
    follow_all_instructions: bool
    follow_instruction_list: list

def test_instruction_following_strict(inp, response):
    found = random.random() > 0.8
    return OutputExample(found, [found])

def test_instruction_following_loose(inp, response):
    return OutputExample(inp.lang == "ja", [inp.lang == "ja"])
"""


class PromptsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        rows = [prompt(1, "3文以上の文章で、雨について書いてください。"), prompt(2, "雪について書いてください。")]
        instructions.write_lines(self.home / "data" / "ifeval_ja.jsonl", rows)

    def test_gives_a_languages_prompts_as_the_authors_give_them(self):
        self.assertEqual([row["key"] for row in instructions.prompts("ja", self.home)], [1, 2])
        with self.assertRaises(ValueError):
            instructions.prompts("de", self.home)

    def test_scores_only_replies_that_answer_each_prompt_once(self):
        asked = instructions.prompts("ja", self.home)
        with self.assertRaisesRegex(ValueError, "each of the prompts once"):
            instructions.evaluated(asked, {"雪について書いてください。": "雪。"}, self.home / "out", self.home, None)

    def test_runs_the_authors_checks_on_each_reply_at_both_levels_with_the_seeds_set_before_each(self):
        checks = self.home / instructions.CHECKS
        checks.mkdir(parents=True)
        (checks / "utils.py").write_text(CHECKS)
        (checks / "langdetect.py").write_text("class DetectorFactory:\n    seed = None\n")
        asked = instructions.prompts("ja", self.home)
        replies = {row["prompt"]: "雨。" for row in asked}
        results = instructions.evaluated(asked, replies, self.home / "scored", self.home, sys.executable)
        self.assertEqual([row["prompt"] for row in results["strict"]], list(replies))
        self.assertEqual([row["follow_all_instructions"] for row in results["strict"]], [True, True])
        self.assertEqual([row["follow_instruction_list"] for row in results["loose"]], [[True], [True]])


class FollowedTest(unittest.TestCase):
    def test_a_higher_share_of_prompts_followed_is_the_better_and_each_language_counts_equally(self):
        chosen = definition.load("instructions-ja-en-1")
        self.assertEqual((chosen["instructions"]["commit"], chosen["instructions"]["revision"]),
                         (instructions.COMMIT, instructions.REVISION))
        standing = definition.followed(chosen, {"en": 0.30, "ja": 0.15})
        self.assertEqual(standing, {"ifeval": 0.30, "ja_ifeval": 0.15})
        better = dict(standing, ja_ifeval=0.18)
        self.assertAlmostEqual(definition.improvement(chosen, better, standing), (0.18 / 0.15 - 1) / 2)
        self.assertLess(definition.improvement(chosen, standing, better), 0)


if __name__ == "__main__":
    unittest.main()
