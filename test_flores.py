import json
import tempfile
import unittest
from pathlib import Path

import olmes_measure as om
from language_model_evaluation import definition, flores, overlap, paloma

# Two splits of the same sentences in both languages, by their ids, which do not sort as text does.
SENTENCES = {
    "dev": {"2": ("Farmers in the valley began the harvest a week earlier than usual because of the dry summer.",
                  "乾いた夏のため、谷の農家はいつもより一週間早く収穫を始めた。")},
    "devtest": {
        "12": ("The museum will stay open until midnight on Fridays during the festival, the city announced.",
               "市の発表によると、祭りの間、博物館は金曜日に限り真夜中まで開館する。"),
        "3": ("Travellers should carry enough water, since the villages on the northern route are far apart.",
              "北の道沿いの村は互いに遠く離れているので、旅行者は十分な水を持って行くべきだ。"),
        "7": ("The committee said that the bridge would reopen to traffic after the repairs finish next spring.",
              "委員会によると、橋は来年の春に修理が終わったあと、再び車が通れるようになるという。")}}


class FloresTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.sentences = {split: dict(found) for split, found in SENTENCES.items()}

    def download(self, path, directory):
        """A file of FLORES+, a sentence a line as it gives them, with fields beside the text."""
        split, name = path.split("/")
        language = name.removesuffix(".jsonl")
        column = list(flores.LANGUAGES).index(language)
        rows = [{"id": int(key), "iso_639_3": language[:3], "text": texts[column], "domain": "wikinews",
                 "split": split} for key, texts in self.sentences[split].items()]
        local = Path(directory) / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return local

    def test_keeps_a_splits_sentences_in_both_languages_in_the_order_of_their_ids(self):
        artifact = flores.fetch(self.directory / "out", "devtest", self.download)
        manifest = om.check(artifact)
        self.assertEqual((manifest["kind"], manifest["revision"]), ("flores-devtest", flores.REVISION))
        found = paloma.files(artifact)
        self.assertEqual(list(found), [("flores", "eng_Latn"), ("flores", "jpn_Jpan")])  # a domain for each language
        for column, language in enumerate(flores.LANGUAGES):
            self.assertEqual(list(paloma.documents(found["flores", language])),
                             [SENTENCES["devtest"][key][column] for key in ("3", "7", "12")])
        # the same sentences, the same artifact
        self.assertEqual(flores.fetch(self.directory / "again", "devtest", self.download).name, artifact.name)

    def test_refuses_languages_that_do_not_hold_the_same_sentences_each_with_text(self):
        self.sentences["devtest"]["7"] = (SENTENCES["devtest"]["7"][0], " ")
        with self.assertRaisesRegex(ValueError, "other sentences"):
            flores.fetch(self.directory / "out", "devtest", self.download)
        original = self.download

        def fewer(path, directory):  # a Japanese file without one of the sentences
            local = original(path, directory)
            if "jpn_Jpan" in path:
                local.write_text("".join(local.read_text(encoding="utf-8").splitlines(keepends=True)[1:]))
            return local

        self.sentences["devtest"]["7"] = SENTENCES["devtest"]["7"]
        with self.assertRaisesRegex(ValueError, "other sentences"):
            flores.fetch(self.directory / "out", "devtest", fewer)

    def test_finds_the_documents_that_hold_a_sentence_of_either_split_in_either_language(self):
        splits = [flores.fetch(self.directory / "out", split, self.download) for split in ("dev", "devtest")]
        artifact = flores.make_overlap(self.directory / "out", splits)
        manifest = om.check(artifact)
        self.assertEqual(manifest["widths"], {"en": [64, 24], "ja": [32, 12]})
        self.assertEqual(manifest["questions"], sorted(split.name for split in splits))
        found = overlap.load(artifact)
        english, japanese = SENTENCES["devtest"]["7"]
        self.assertTrue(found.holds("NEWS\n" + english.upper().replace(" ", "  ") + "\nMore later."))
        self.assertTrue(found.holds("記事：" + japanese.replace("、", "，") + "以上。"))
        self.assertTrue(found.holds(SENTENCES["dev"]["2"][1]))
        self.assertFalse(found.holds("The bridge reopened to traffic in the spring after the repairs."))
        self.assertFalse(found.holds("橋は春に修理を終え、再び車が通れるようになった。"))

    def test_refuses_to_make_the_overlap_of_another_revisions_sentences(self):
        artifact = flores.fetch(self.directory / "out", "dev", self.download)
        work = Path(tempfile.mkdtemp(dir=self.directory))
        (work / "texts").mkdir()
        manifest = om.check(artifact)
        other = om.seal(work, "flores-dev", repository=flores.REPOSITORY, revision="0" * 40,
                        languages=manifest["languages"])
        with self.assertRaisesRegex(ValueError, "was not fetched"):
            flores.make_overlap(self.directory / "out", [other])

    def test_makes_every_artifact_that_a_definition_of_a_balance_names_under_its_name(self):
        first = self.directory / "first"
        dev, devtest = (flores.fetch(first, split, self.download) for split in ("dev", "devtest"))
        toy = {"name": "toy", "balance": ["jpn_Jpan", "eng_Latn"],
               "flores": {"split": "devtest", "text": devtest.name},
               "overlap": {"flores": {"dev": dev.name, "devtest": devtest.name},
                           "artifact": flores.make_overlap(first, [dev, devtest]).name}}
        made = definition.make_flores(toy, self.directory / "second", self.download)
        self.assertEqual(set(made), {dev.name, devtest.name, toy["overlap"]["artifact"]})
        for name, artifact in made.items():
            self.assertEqual(om.check(artifact)["kind"], name.rsplit("-", 1)[0])
        # what is there already is checked and kept: nothing is fetched again
        refuse = lambda *arguments: self.fail("fetched again")
        self.assertEqual(definition.make_flores(toy, self.directory / "second", refuse), made)
        other = dict(toy, flores=dict(toy["flores"], text="flores-devtest-0000000000000000"))
        with self.assertRaisesRegex(ValueError, "flores-devtest-0000000000000000"):
            definition.make_flores(other, self.directory / "third", self.download)


if __name__ == "__main__":
    unittest.main()
