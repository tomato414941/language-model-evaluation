"""Following instructions, as BenchMAX asks and scores it in English and in Japanese.

BenchMAX (CONE-MT/BenchMAX, its prompts LLaMAX/BenchMAX_Rule-based) asks 429 of IFEval's prompts in English and the
same prompts in Japanese, translated by machine and corrected by three native speakers, without the instructions that
only English can follow. A model writes a reply to each prompt, and the authors' code tells whether the reply follows
every instruction that the prompt gives (strict) or would after a light cleaning (loose); it counts a Japanese reply's
words by MeCab and its sentences by their ends.

- `code(work)`: the authors' code at COMMIT and their prompts at REVISION;
- `prompts(language, home)`: the prompts of a language, as the authors give them: key, prompt, instruction ids and
  their arguments;
- `score(language, replies, work)`: each prompt's results at both levels, from the authors' checks run unchanged in an
  environment of PACKAGES, and the share of prompts whose every instruction is followed.

The authors' checks draw a random value where a prompt leaves one open, and langdetect draws its own: both are seeded
before each reply, so that a reply scores the same each time and beside any other.

A project writes the replies itself, each prompt the user's message of a conversation in its model's format, greedily
up to MAX_NEW_TOKENS, as the authors' task asks (temperature 0, 2048 tokens).
"""
import json
import subprocess
import urllib.request
from pathlib import Path

REPOSITORY = "https://github.com/CONE-MT/BenchMAX.git"
COMMIT = "03cfd656c1b0b93647b5dabf5a11ce9be362493c"
DATA = "LLaMAX/BenchMAX_Rule-based"
REVISION = "36a659fc9dddc2b344d98971ea0643527a3658d1"
# Where the authors' checks are in their code.
CHECKS = Path("tasks") / "ifeval"
LANGUAGES = ("en", "ja")
LEVELS = ("strict", "loose")
MAX_NEW_TOKENS = 2048
# What the authors' checks import, at the versions current when this was written; their requirements pin none.
PACKAGES = ("immutabledict==4.2.1", "langdetect==1.0.9", "nltk==3.9.1", "packaging==24.2", "sacrebleu[ja]==2.5.1",
            "setuptools==75.8.0")


def run(*command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def code(work):
    """The authors' code at COMMIT in work/code, with their prompts at REVISION in its data/."""
    home = Path(work) / "code"
    if not (home / CHECKS / "utils.py").exists():
        home.mkdir(parents=True, exist_ok=True)
        run("git", "init", "--quiet", cwd=home)
        run("git", "fetch", "--quiet", "--depth", "1", REPOSITORY, COMMIT, cwd=home)
        run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=home)
    for language in LANGUAGES:
        path = home / "data" / f"ifeval_{language}.jsonl"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            partial = path.with_suffix(".part")
            urllib.request.urlretrieve(
                f"https://huggingface.co/datasets/{DATA}/resolve/{REVISION}/{path.name}", partial)
            partial.rename(path)
    return home


def environment(work):
    """The Python of an environment with PACKAGES and the NLTK data that the checks read, in work/environment."""
    python = Path(work) / "environment" / "bin" / "python"
    if not python.exists():
        run("uv", "venv", "--quiet", "--python", "3.12", str(python.parent.parent))
        run("uv", "pip", "install", "--quiet", "--python", str(python), *PACKAGES)
        # The checks fetch it on import otherwise; kept in the environment, where NLTK looks.
        run(str(python), "-c", "import nltk, sys; nltk.download('punkt_tab', sys.prefix + '/nltk_data', quiet=True)")
    return python


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def prompts(language, home):
    """The authors' prompts of a language, in their order."""
    if language not in LANGUAGES:
        raise ValueError(f"no prompts of {language}")
    return read_lines(Path(home) / "data" / f"ifeval_{language}.jsonl")


# Run in the authors' environment: each reply checked by their strict and loose tests, each with the same seeds.
CHECK = """
import json, random, sys
checks, directory = sys.argv[1:]
sys.path.insert(0, checks)
import langdetect, utils
rows = [json.loads(line) for line in open(directory + "/responses.jsonl", encoding="utf-8")]
for level in ("strict", "loose"):
    test = getattr(utils, "test_instruction_following_" + level)
    with open(directory + "/eval_results_" + level + ".jsonl", "w", encoding="utf-8") as out:
        for row in rows:
            random.seed(0)
            langdetect.DetectorFactory.seed = 0
            asked = utils.InputExample(key=row["key"], instruction_id_list=row["instruction_id_list"],
                                       prompt=row["prompt"], kwargs=row["kwargs"], lang=row.get("lang", "en"))
            found = test(asked, row["response"])
            result = {"prompt": row["prompt"], "follow_all_instructions": found.follow_all_instructions,
                      "follow_instruction_list": found.follow_instruction_list}
            out.write(json.dumps(result, ensure_ascii=False) + "\\n")
"""


def evaluated(asked, replies, directory, home, python):
    """The authors' results of each level for replies ({prompt: reply}) to prompts (as `prompts` gives them)."""
    if set(replies) != {row["prompt"] for row in asked}:
        raise ValueError("the replies do not answer each of the prompts once")
    directory = Path(directory).absolute()
    write_lines(directory / "responses.jsonl", [{**row, "response": replies[row["prompt"]]} for row in asked])
    # The environment's Python stays the link that it is, by which it finds the environment's packages.
    run(str(Path(python).absolute()), "-c", CHECK, str((Path(home) / CHECKS).absolute()), str(directory))
    return {level: read_lines(directory / f"eval_results_{level}.jsonl") for level in LEVELS}


def score(language, replies, work):
    """A language's results for replies ({prompt: reply}): each level's share of prompts whose every instruction is
    followed, and each prompt's results."""
    home, python = code(work), environment(work)
    results = evaluated(prompts(language, home), replies, Path(work) / "scored" / language, home, python)
    shares = {level: sum(row["follow_all_instructions"] for row in rows) / len(rows) for level, rows in results.items()}
    return {"language": language, "commit": COMMIT, "prompts": len(results["strict"]), **shares, "results": results}
