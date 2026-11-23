"""Following instructions in Japanese, of IFEval's kinds and of kinds of Japanese writing, as M-IFEval asks and scores
them.

M-IFEval (lightblue-tech/M-IFEval) asks 172 prompts in Japanese, with 21 of IFEval's kinds of instructions and 12 that
IFEval does not have, such as writing in hiragana only, giving the readings of kanji, or ending sentences in nouns
(体言止め); its check of no 読点 counts 、. The authors' code at COMMIT holds the prompts and the checks, and tells
whether a reply follows every instruction that the prompt gives (strict) or would after a light cleaning (loose).

- `code(work)`: the authors' code and prompts at COMMIT;
- `prompts(language, home)`: the prompts, in Japanese only, as the authors give them: key, prompt, instruction ids and
  their arguments;
- `score(language, replies, work)`: each prompt's results at both levels, from the authors' checks run unchanged in an
  environment of their pins, and the share of prompts whose every instruction is followed.

The checks are run as IFBench's are (ifbench.py), seeded before each reply. Run on the replies that the authors publish
of three models, they repeat the authors' published results of every prompt at both levels, but one reply's at the
strict level: Claude 3.5 Sonnet's, written in Thai and Japanese, whose language langdetect draws. Seven prompts of the
input file say N文 (sentences) where the published replies were asked N行 (lines), with the same instructions to check;
they were compared as their replies were asked. A project writes the replies itself, as it writes BenchMAX's
(instructions.py).
"""
from pathlib import Path

from . import ifbench
from .instructions import LEVELS, read_lines, run

REPOSITORY = "https://github.com/lightblue-tech/M-IFEval.git"
COMMIT = "10b874dbf8c4ca25ed5d876e1d3226e7f3f9d0dc"
LANGUAGES = ("ja",)
# The authors' pins of what their checks import; the rest of their requirements serve to write replies.
PACKAGES = ("absl-py==1.4.0", "immutabledict==4.2.1", "janome==0.5.0", "ja_sentence_segmenter==0.0.2",
            "langdetect==1.0.9", "nltk==3.9.1", "spacy==3.7.5")
# The models that their README downloads, which the checks of other languages load when the checks are imported.
MODELS = tuple(f"https://github.com/explosion/spacy-models/releases/download/{name}-3.7.0/{name}-3.7.0-py3-none-any.whl"
               for name in ("es_core_news_sm", "xx_sent_ud_sm"))


def code(work):
    """The authors' code and prompts at COMMIT in work/mifeval/code."""
    home = Path(work) / "mifeval" / "code"
    if not (home / "evaluation_main.py").exists():
        home.mkdir(parents=True, exist_ok=True)
        run("git", "init", "--quiet", cwd=home)
        run("git", "fetch", "--quiet", "--depth", "1", REPOSITORY, COMMIT, cwd=home)
        run("git", "checkout", "--quiet", "FETCH_HEAD", cwd=home)
    return home


def environment(work):
    """The Python of an environment with the authors' pins and MODELS, in work/mifeval/environment."""
    python = Path(work) / "mifeval" / "environment" / "bin" / "python"
    if not python.exists():
        run("uv", "venv", "--quiet", "--python", "3.12", str(python.parent.parent))
        run("uv", "pip", "install", "--quiet", "--python", str(python), *PACKAGES, *MODELS)
        # The checks of French fetch it on import otherwise; kept in the environment, where NLTK looks.
        run(str(python), "-c", "import nltk, sys; nltk.download('punkt_tab', sys.prefix + '/nltk_data', quiet=True)")
    return python


def prompts(language, home):
    """The authors' prompts of a language, in their order: Japanese only."""
    if language not in LANGUAGES:
        raise ValueError(f"no prompts of {language}")
    return read_lines(Path(home) / "data" / f"{language}_input_data.jsonl")


def evaluated(asked, replies, directory, home, python):
    """The authors' results of each level for replies ({prompt: reply}) to prompts (as `prompts` gives them)."""
    return ifbench.evaluated(asked, replies, directory, home, python, checks="evaluation_main")


def score(language, replies, work):
    """A language's results for replies ({prompt: reply}): each level's share of prompts whose every instruction is
    followed, and each prompt's results."""
    home, python = code(work), environment(work)
    results = evaluated(prompts(language, home), replies, Path(work) / "mifeval" / "scored" / language, home, python)
    shares = {level: sum(row["follow_all_instructions"] for row in rows) / len(rows) for level, rows in results.items()}
    return {"language": language, "commit": COMMIT, "prompts": len(results["strict"]), **shares, "results": results}
