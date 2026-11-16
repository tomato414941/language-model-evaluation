"""FLORES+: the same sentences in English and in Japanese, by which a model's reading of the two languages compares.

FLORES+ (openlanguagedata/flores_plus, CC-BY-SA-4.0), which the Open Language Data Initiative keeps after Meta's
FLORES-200, holds sentences of English Wikinews, Wikijunior and Wikivoyage translated by people into some 200
languages: 997 in its dev split and 1,012 in its devtest split, the same sentences in every language. Its terms ask
that it is not hosted again where crawlers can collect it, and that a model that it evaluates has not learned it: its
artifacts stay on the machines that make and measure them, and `make_overlap` gives what tells the text that such a
model may not learn.

- `fetch`: a split's sentences in English and Japanese, as an artifact of the form of paloma.fetch's
  (texts/flores/SPLIT_LANGUAGE.jsonl.gz, a sentence {"id", "text"} a line, in the order of their ids in both
  languages), so that paloma.document_rows reads each sentence by itself as a document of the domain
  flores/LANGUAGE;
- `make_overlap`: the artifact of what tells the documents that hold a sentence of some splits, as overlap.load reads
  it. A sentence is too short for the stretches by which overlap.py finds Paloma's text, and is found as a question
  is, by the middle of its letters, as wide as its language's.

FLORES+ is gated: for `fetch`, HF_TOKEN must hold a token of a Hugging Face account that has accepted its terms.
"""
import gzip
import io
import json
import os
import tempfile
from pathlib import Path

import olmes_measure

from . import overlap, paloma

REPOSITORY = "openlanguagedata/flores_plus"
REVISION = "1147e2dc59f37fbb5c22fd8478770de02cd399f9"  # its version 5.0
# The languages that are kept, by FLORES+'s names, and overlap.py's names of them.
LANGUAGES = {"eng_Latn": "en", "jpn_Jpan": "ja"}


def download(path, directory):
    """One file of FLORES+ at REVISION, fetched into the directory."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPOSITORY, path, repo_type="dataset", revision=REVISION, local_dir=directory,
                           token=os.environ["HF_TOKEN"])


def fetch(out, split, download=download):
    """The artifact of a split's sentences (dev or devtest) in each of LANGUAGES, in the order of their ids. The
    languages must hold the same sentences, each with text."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=out))
    (work / "texts" / "flores").mkdir(parents=True)
    ids = None
    with tempfile.TemporaryDirectory() as staging:
        for language in LANGUAGES:
            lines = Path(download(f"{split}/{language}.jsonl", staging)).read_text(encoding="utf-8").splitlines()
            rows = sorted((json.loads(line) for line in lines if line.strip()), key=lambda row: int(row["id"]))
            if ids is None:
                ids = [row["id"] for row in rows]
            if [row["id"] for row in rows] != ids or not all(row["text"].strip() for row in rows):
                raise ValueError(f"{split} holds other sentences in {language}, or one without text")
            data = io.BytesIO()
            with gzip.GzipFile(fileobj=data, mode="wb", mtime=0) as file:  # no time in it: the same text, the same file
                for row in rows:
                    file.write((json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")
                               .encode("utf-8"))
            (work / "texts" / "flores" / f"{split}_{language}.jsonl.gz").write_bytes(data.getvalue())
    return olmes_measure.seal(work, f"flores-{split}", repository=REPOSITORY, revision=REVISION,
                              languages=list(LANGUAGES))


def make_overlap(out, artifacts):
    """The artifact of what tells the documents that hold a sentence of some of `fetch`'s artifacts: each sentence's
    key, as a question's (overlap.keys_of), as wide as its language's, in questions.txt as overlap.make keeps them."""
    keys = set()
    for artifact in artifacts:
        manifest = olmes_measure.check(artifact)
        if (manifest.get("repository"), manifest.get("revision")) != (REPOSITORY, REVISION):
            raise ValueError(f"{artifact} was not fetched from FLORES+ {REVISION}")
        for (_, language), paths in paloma.files(artifact).items():
            keys.update(overlap.keys_of(paloma.documents(paths), LANGUAGES[language]))
    return overlap.write(out, sorted(keys), b"", questions=sorted(Path(artifact).name for artifact in artifacts),
                         widths={name: overlap.WIDTHS[name] for name in sorted(LANGUAGES.values())})
