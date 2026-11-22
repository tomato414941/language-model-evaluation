import sys
import tempfile
import unittest
from pathlib import Path

from language_model_evaluation import definition, ifbench, instructions


def prompt(key, text):
    return {"key": key, "prompt": text, "instruction_id_list": ["count:numbers"], "kwargs": [{"N": 3}]}


# The authors' evaluation_lib as a module of its form: a reply follows its prompt at the strict level when the first
# number drawn after the seed is above 0.8, as it is only when the seed is set before each reply, and at the loose level
# when the reply is the prompt's own.
EVALUATION_LIB = """
import dataclasses, random

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
    found = prompt_to_response[inp.prompt] == inp.prompt.upper()
    return OutputExample(found, [found])
"""


class PromptsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.home = Path(directory.name)
        rows = [prompt("0", "Write about rain with 3 numbers."), prompt("1", "Write about snow with 3 numbers.")]
        instructions.write_lines(self.home / ifbench.PROMPTS, rows)

    def test_gives_the_english_prompts_as_the_authors_give_them(self):
        self.assertEqual([row["key"] for row in ifbench.prompts("en", self.home)], ["0", "1"])
        with self.assertRaises(ValueError):
            ifbench.prompts("ja", self.home)

    def test_scores_only_replies_that_answer_each_prompt_once(self):
        asked = ifbench.prompts("en", self.home)
        with self.assertRaisesRegex(ValueError, "each of the prompts once"):
            ifbench.evaluated(asked, {"Write about snow with 3 numbers.": "Snow."}, self.home / "out", self.home, None)

    def test_runs_the_authors_checks_on_each_reply_at_both_levels_with_the_seeds_set_before_each(self):
        (self.home / "evaluation_lib.py").write_text(EVALUATION_LIB)
        (self.home / "langdetect.py").write_text("class DetectorFactory:\n    seed = None\n")
        asked = ifbench.prompts("en", self.home)
        replies = {asked[0]["prompt"]: asked[0]["prompt"].upper(), asked[1]["prompt"]: "Snow."}
        results = ifbench.evaluated(asked, replies, self.home / "scored", self.home, sys.executable)
        self.assertEqual([row["prompt"] for row in results["strict"]], list(replies))
        self.assertEqual([row["follow_all_instructions"] for row in results["strict"]], [True, True])
        self.assertEqual([row["follow_instruction_list"] for row in results["loose"]], [[True], [False]])


class FollowedTest(unittest.TestCase):
    def test_english_counts_ifevals_kinds_and_ifbenchs_new_kinds_equally(self):
        chosen = definition.load("instructions-en-v1")
        self.assertEqual(chosen["ifbench"]["commit"], ifbench.COMMIT)
        self.assertEqual(chosen["instructions"], definition.load("instructions-ja-en-v1")["instructions"])
        standing = definition.followed(chosen, {"en": 0.30, definition.prompts_name("ifbench", "en"): 0.10})
        self.assertEqual(standing, {"ifeval": 0.30, "ifbench": 0.10})
        better = dict(standing, ifbench=0.12)
        self.assertAlmostEqual(definition.improvement(chosen, better, standing), (0.12 / 0.10 - 1) / 2)


if __name__ == "__main__":
    unittest.main()
