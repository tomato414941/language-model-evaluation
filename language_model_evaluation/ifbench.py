"""Following instructions of kinds that IFEval does not ask, as IFBench asks and scores them in English.

IFBench (allenai/IFBench) asks 300 prompts, each a request of a held-out part of WildChat with one or more of 58
instructions that IFEval does not have, so that a model trained on IFEval's kinds of instructions is measured on kinds
that it was not. The authors' code at COMMIT holds the prompts (PROMPTS) and the checks, and tells whether a reply
follows every instruction that the prompt gives (strict) or would after a light cleaning (loose).

- `code(work)`: the authors' code and prompts at COMMIT;
- `prompts(language, home)`: the prompts, in English only, as the authors give them: key, prompt, instruction ids and
  their arguments;
- `score(language, replies, work)`: each prompt's results at both levels, from the authors' checks run unchanged in an
  environment of PACKAGES with the NLTK data that they read, and the share of prompts whose every instruction is
  followed.

The checks draw a random value where a prompt leaves one open, and langdetect draws its own: both are seeded before
each reply, so that a reply scores the same each time and beside any other. A project writes the replies itself, as it
writes BenchMAX's (instructions.py).
"""
from pathlib import Path

from .instructions import LEVELS, read_lines, run, write_lines

REPOSITORY = "https://github.com/allenai/IFBench.git"
COMMIT = "488918665d53f867fa908a293769ec6e71e74aec"
# The prompts as the authors' README names them.
PROMPTS = Path("data") / "IFBench_test.jsonl"
LANGUAGES = ("en",)
# What the authors' checks import, at the versions of the authors' uv.lock at COMMIT.
PACKAGES = ("absl-py==2.3.1", "emoji==2.15.0", "immutabledict==4.2.2", "langdetect==1.0.9", "nltk==3.9.2",
            "syllapy==0.8.0")
# The NLTK data that the checks fetch on import otherwise.
NLTK_DATA = ("punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger_eng")


def code(work):
    """The authors' code and prompts at COMMIT in work/ifbench/code."""
    home = Path(work) / "ifbench" / "code"
    if not (home / "evaluation_lib.py").exists():
        home.mkdir(parents=True, exist_ok=True)
        run("git", "init", "--quiet", cwd=home)
        run("git", "fetch", "--quiet", "--depth", "1", REPOSITORY, COMMIT, cwd=home)
        run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=home)
    return home


def environment(work):
    """The Python of an environment with PACKAGES and NLTK_DATA, in work/ifbench/environment."""
    python = Path(work) / "ifbench" / "environment" / "bin" / "python"
    if not python.exists():
        run("uv", "venv", "--quiet", "--python", "3.12", str(python.parent.parent))
        run("uv", "pip", "install", "--quiet", "--python", str(python), *PACKAGES)
        # Kept in the environment, where NLTK looks before the authors' code would fetch them.
        run(str(python), "-c", "import nltk, sys\n"
            f"for name in {NLTK_DATA!r}: nltk.download(name, sys.prefix + '/nltk_data', quiet=True)")
    return python


def prompts(language, home):
    """The authors' prompts of a language, in their order: English only."""
    if language not in LANGUAGES:
        raise ValueError(f"no prompts of {language}")
    return read_lines(Path(home) / PROMPTS)


# Run in the authors' environment: each reply checked by the strict and loose tests of their module of checks, each
# with the same seeds.
CHECK = """
import importlib, json, random, sys
home, directory, name = sys.argv[1:]
sys.path.insert(0, home)
import langdetect
checks = importlib.import_module(name)
rows = [json.loads(line) for line in open(directory + "/responses.jsonl", encoding="utf-8")]
for level in ("strict", "loose"):
    test = getattr(checks, "test_instruction_following_" + level)
    with open(directory + "/eval_results_" + level + ".jsonl", "w", encoding="utf-8") as out:
        for row in rows:
            random.seed(0)
            langdetect.DetectorFactory.seed = 0
            asked = checks.InputExample(key=row["key"], instruction_id_list=row["instruction_id_list"],
                                        prompt=row["prompt"], kwargs=row["kwargs"])
            found = test(asked, {row["prompt"]: row["response"]})
            result = {"prompt": row["prompt"], "follow_all_instructions": found.follow_all_instructions,
                      "follow_instruction_list": found.follow_instruction_list}
            out.write(json.dumps(result, ensure_ascii=False) + "\\n")
"""


def evaluated(asked, replies, directory, home, python, checks="evaluation_lib"):
    """The authors' results of each level for replies ({prompt: reply}) to prompts (as `prompts` gives them), from the
    tests of their module of checks, which M-IFEval's code holds as IFBench's does (mifeval.py)."""
    if set(replies) != {row["prompt"] for row in asked}:
        raise ValueError("the replies do not answer each of the prompts once")
    directory = Path(directory).absolute()
    write_lines(directory / "responses.jsonl", [{**row, "response": replies[row["prompt"]]} for row in asked])
    # The environment's Python stays the link that it is, by which it finds the environment's packages.
    run(str(Path(python).absolute()), "-c", CHECK, str(Path(home).absolute()), str(directory), checks)
    return {level: read_lines(directory / f"eval_results_{level}.jsonl") for level in LEVELS}


def score(language, replies, work):
    """A language's results for replies ({prompt: reply}): each level's share of prompts whose every instruction is
    followed, and each prompt's results."""
    home, python = code(work), environment(work)
    results = evaluated(prompts(language, home), replies, Path(work) / "ifbench" / "scored" / language, home, python)
    shares = {level: sum(row["follow_all_instructions"] for row in rows) / len(rows) for level, rows in results.items()}
    return {"language": language, "commit": COMMIT, "prompts": len(results["strict"]), **shares, "results": results}
