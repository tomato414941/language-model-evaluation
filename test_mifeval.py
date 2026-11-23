import sys
import tempfile
import unittest
from pathlib import Path

from language_model_evaluation import definition, instructions, mifeval

# The authors' evaluation_main as a module of its form: a reply follows its prompt at the strict level when the first
# number drawn after the seed is above 0.8, as it is only when the seed is set before each reply, and at the loose level
# when the reply is in hiragana only.
EVALUATION_MAIN = """
import dataclasses, random, re

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list
    prompt: str
    kwargs: list

@dataclasses.dataclass
class OutputExample:
    follow_all_instructions: bool
    follow_instruction_list: list

def test_instruction_following_strict(inp, prompt_to_response):
    found = random.random() > 0.8
    return OutputExample(found, [found])

def test_instruction_following_loose(inp, prompt_to_response):
    found = re.fullmatch("[ぁ-ゖ。]+", prompt_to_response[inp.prompt]) is not None
    return OutputExample(found, [found])
"""


def prompt(key, text):
    return {"key": key, "prompt": text, "instruction_id_list": ["ja:letters:hiragana_only"], "kwargs": [{}]}


class PromptsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        rows = [prompt(1, "ひらがなだけで、雨について書いてください。"), prompt(2, "ひらがなだけで、雪について書いてください。")]
        instructions.write_lines(self.home / "data" / "ja_input_data.jsonl", rows)

    def test_gives_the_japanese_prompts_as_the_authors_give_them(self):
        self.assertEqual([row["key"] for row in mifeval.prompts("ja", self.home)], [1, 2])
        with self.assertRaises(ValueError):
            mifeval.prompts("en", self.home)

    def test_scores_only_replies_that_answer_each_prompt_once(self):
        asked = mifeval.prompts("ja", self.home)
        with self.assertRaisesRegex(ValueError, "each of the prompts once"):
            mifeval.evaluated(asked, {asked[1]["prompt"]: "ゆき。"}, self.home / "out", self.home, None)

    def test_runs_the_authors_checks_on_each_reply_at_both_levels_with_the_seeds_set_before_each(self):
        (self.home / "evaluation_main.py").write_text(EVALUATION_MAIN)
        (self.home / "langdetect.py").write_text("class DetectorFactory:\n    seed = None\n")
        asked = mifeval.prompts("ja", self.home)
        replies = {asked[0]["prompt"]: "あめがふる。", asked[1]["prompt"]: "雪が降る。"}
        results = mifeval.evaluated(asked, replies, self.home / "scored", self.home, sys.executable)
        self.assertEqual([row["prompt"] for row in results["strict"]], list(replies))
        self.assertEqual([row["follow_all_instructions"] for row in results["strict"]], [True, True])
        self.assertEqual([row["follow_instruction_list"] for row in results["loose"]], [[True], [False]])


class FollowedTest(unittest.TestCase):
    def test_each_language_counts_equally_and_its_two_sources_equally_within_it(self):
        chosen = definition.load("instructions-ja-en-v2")
        self.assertEqual(chosen["mifeval"]["commit"], mifeval.COMMIT)
        english = definition.load("instructions-en-v1")
        self.assertEqual((chosen["instructions"], chosen["ifbench"]), (english["instructions"], english["ifbench"]))
        shares = {"en": 0.30, "ja": 0.15, definition.prompts_name("ifbench", "en"): 0.10,
                  definition.prompts_name("mifeval", "ja"): 0.20}
        standing = definition.followed(chosen, shares)
        self.assertEqual(standing, {"ifeval": 0.30, "ifbench": 0.10, "ja_ifeval": 0.15, "ja_mifeval": 0.20})
        better = dict(standing, ja_mifeval=0.25)
        self.assertAlmostEqual(definition.improvement(chosen, better, standing), (0.25 / 0.20 - 1) / 4)


if __name__ == "__main__":
    unittest.main()
