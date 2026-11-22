import tempfile
import unittest
from pathlib import Path

import olmes_measure as om
from language_model_evaluation import definition, flores, overlap, paloma
from test_olmes_measure import make_requests, request
from test_paloma import CATEGORIES, fetched, members_files

TOY = {"name": "toy", "compare": "mean_rate", "olmes": {"suites": ["first", "second"]},
       "categories": {"science": {"olmes": ["t1", {"hard": ["t2", "t3"]}]}, "sums": {"olmes": ["second"]},
                      "web": {"paloma": CATEGORIES["web"]}, "books": {"paloma": CATEGORIES["books"]}}}
GROUPS = {"first": ["t1", "pair"], "pair": ["t2", "t3"], "second": ["t4", "inner"], "inner": ["t5", "t6"]}
# Two languages: English as TOY's, Japanese questions and text beside them.
BOTH = {**TOY, "compare": "mean_rate_by_language", "japanese": {"suites": ["ja"]},
        "categories": {**TOY["categories"], "ja_common": {"japanese": ["j1", "j2"]},
                       "ja_web": {"japanese_text": ["mc4_ja"]}},
        "languages": {"en": ["science", "sums", "web", "books"], "ja": ["ja_common", "ja_web"]}}


class LoadTest(unittest.TestCase):
    def test_a_definition_within_another_says_what_differs_and_takes_the_rest(self):
        whole, part = definition.load("knowledge-en-v1"), definition.load("knowledge-en-part-v1")
        self.assertNotIn("within", whole)
        self.assertEqual((part["name"], part["within"]), ("knowledge-en-part-v1", "knowledge-en-v1"))
        for key in ("categories", "compare", "overlap"):
            self.assertEqual(part[key], whole[key])
        self.assertEqual(part["olmes"]["requests"], whole["olmes"]["requests"])
        self.assertEqual(part["olmes"]["suites"], whole["olmes"]["suites"])  # what the categories count
        self.assertEqual(part["olmes"]["per_task"], 100)
        self.assertNotEqual(part["olmes"]["selection"], whole["olmes"]["selection"])
        self.assertEqual(part["paloma"]["whole"], whole["paloma"]["text"])  # the text that the part is cut from
        self.assertNotEqual(part["paloma"]["text"], whole["paloma"]["text"])
        self.assertEqual(len(definition.categories(part)), 11)
        self.assertEqual(list(definition.members(whole)), ["web", "encyclopedia", "papers", "posts", "books", "news"])


class TwoLanguagesTest(unittest.TestCase):
    def test_knowledge_ja_en_measures_knowledge_en_beside_japanese_and_counts_the_two_languages_equally(self):
        english, both = definition.load("knowledge-en-v1"), definition.load("knowledge-ja-en-v1")
        for section in ("olmes", "paloma"):
            self.assertEqual(both[section], english[section])
        self.assertEqual({k: v for k, v in both["overlap"].items() if k != "japanese"}, english["overlap"])
        self.assertEqual(list(both["categories"])[:11], list(english["categories"]))
        self.assertEqual({language: len(names) for language, names in both["languages"].items()}, {"en": 11, "ja": 6})
        self.assertEqual(both["compare"], "mean_rate_by_language")
        self.assertEqual(list(definition.members(both, "japanese_text")), ["ja_web", "ja_encyclopedia"])

    def test_knowledge_ja_en_part_cuts_the_questions_and_the_text_of_both_languages_alike(self):
        whole, part = definition.load("knowledge-ja-en-v1"), definition.load("knowledge-ja-en-part-v1")
        self.assertEqual(part["within"], "knowledge-ja-en-v1")
        for key in ("categories", "languages", "compare", "overlap"):
            self.assertEqual(part[key], whole[key])
        self.assertEqual((part["olmes"]["per_task"], part["japanese"]["per_task"]), (3000, 3000))
        self.assertEqual(part["paloma"]["part"], part["japanese_text"]["part"])
        for section in ("paloma", "japanese_text"):
            self.assertEqual(part[section]["whole"], whole[section]["text"])  # the text that the part is cut from
            self.assertNotEqual(part[section]["text"], whole[section]["text"])


class ScoresTest(unittest.TestCase):
    def test_a_category_that_is_answered_is_the_mean_of_its_tasks_groups_and_named_means(self):
        scores = {"t1": 1.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0}
        found = definition.answered(TOY, GROUPS, scores)
        self.assertEqual(found, {"science": (1.0 + 3.0) / 2, "sums": (6.0 + 2.0) / 2})  # a group counts as one member
        self.assertEqual(found["sums"], om.suite_scores(GROUPS, scores)["second"])

    def test_the_categories_count_every_task_of_the_suites_once_unless_the_selection_is_partial(self):
        scores = {"t1": 1.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0}
        twice = dict(TOY, categories={**TOY["categories"], "again": {"olmes": ["t1"]}})
        with self.assertRaisesRegex(ValueError, "once"):
            definition.answered(twice, GROUPS, scores)
        fewer = dict(TOY, categories={"science": {"olmes": ["t1"]}, "sums": {"olmes": ["second"]}})
        with self.assertRaisesRegex(ValueError, "once"):
            definition.answered(fewer, GROUPS, scores)  # t2 and t3 are asked and not counted
        # a reserve that lacks t3 and all of the inner group: its groups hold what is left
        left = {"t1": 1.0, "t2": 2.0, "t4": 6.0}
        pruned = {"first": ["t1", "pair"], "pair": ["t2"], "second": ["t4"]}
        with self.assertRaisesRegex(ValueError, "does not hold"):
            definition.answered(TOY, pruned, left)
        self.assertEqual(definition.answered(TOY, pruned, left, partial=True), {"science": 1.5, "sums": 6.0})

    def test_a_category_that_is_read_is_the_mean_of_its_members_and_a_source_the_mean_of_its_domains(self):
        domains = {"c4_en/c4_en": 1.0, "c4_100_domains/1_a": 2.0, "c4_100_domains/2_b": 4.0,
                   "dolma-v1_5/common-crawl": 5.0, "dolma-v1_5/books": 7.0, "dolma-v1_5/wiki": 100.0}
        found = definition.read(TOY, domains)
        self.assertEqual(found, {"web": (1.0 + 3.0 + 5.0) / 3, "books": 7.0})  # the wiki is no member of these
        del domains["dolma-v1_5/common-crawl"]
        with self.assertRaisesRegex(ValueError, "common-crawl"):
            definition.read(TOY, domains)
        self.assertEqual(definition.read(TOY, domains, partial=True)["web"], 2.0)
        del domains["dolma-v1_5/books"]
        with self.assertRaisesRegex(ValueError, "books"):
            definition.read(TOY, domains, partial=True)  # a category with no member at all has no score

    def test_answers_and_reads_the_japanese_categories_beside_the_english_ones(self):
        scores = {"t1": 1.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0, "j1": 1.0, "j2": 2.0}
        groups = {**GROUPS, "ja": ["j1", "j2"]}
        found = definition.answered(BOTH, groups, scores)
        self.assertEqual(found, {"science": 2.0, "sums": 4.0, "ja_common": 1.5})
        fewer = dict(BOTH, categories={**BOTH["categories"], "ja_common": {"japanese": ["j1"]}})
        with self.assertRaisesRegex(ValueError, "once"):
            definition.answered(fewer, groups, scores)  # j2 is asked and not counted
        domains = {"c4_en/c4_en": 1.0, "c4_100_domains/1_a": 3.0, "dolma-v1_5/common-crawl": 5.0,
                   "dolma-v1_5/books": 7.0, "mc4_ja/mc4_ja": 0.9}
        self.assertEqual(definition.read(BOTH, domains), {"web": 3.0, "books": 7.0, "ja_web": 0.9})

    def test_each_language_counts_equally_however_many_categories_it_has(self):
        standing = {category: 2.0 for category in definition.categories(BOTH)}
        better_english = dict(standing, science=1.0)  # one of four English categories half the bits
        better_japanese = dict(standing, ja_web=1.0)  # one of two Japanese categories half the bits
        self.assertAlmostEqual(definition.improvement(BOTH, better_english, standing), (0.5 / 4) / 2)
        self.assertAlmostEqual(definition.improvement(BOTH, better_japanese, standing), (0.5 / 2) / 2)
        missing = dict(BOTH, languages={"en": ["science", "sums", "web", "books"], "ja": ["ja_common"]})
        with self.assertRaisesRegex(ValueError, "every category"):
            definition.improvement(missing, better_english, standing)

    def test_one_result_improves_on_another_by_the_mean_of_the_categories_rates(self):
        standing = {category: 2.0 for category in definition.categories(TOY)}
        candidate = dict(standing, web=1.0, sums=2.2)  # one category half the bits, one a tenth more
        self.assertEqual(definition.categories(TOY), ("science", "sums", "web", "books"))
        self.assertAlmostEqual(definition.improvement(TOY, candidate, standing), (0.5 - 0.1) / 4)
        self.assertEqual(definition.improvement(TOY, standing, standing), 0)
        with self.assertRaises(ValueError):
            definition.improvement(dict(TOY, compare="other"), candidate, standing)


# TOY with Paloma's aggregate beside its kinds of text, weighed by the bytes of the domains of two splits, and its
# result compared by its categories' rates weighted.
AGGREGATED = {**TOY, "categories": {**TOY["categories"], "text": {"paloma_aggregate": "bytes"}},
              "paloma": {"whole": "paloma-val-0", "reserve": {"whole": "paloma-test-0"}},
              "domain_bytes": {"paloma-val-0": {"c4_en/c4_en": 100, "c4_100_domains/1_a": 300, "c4_100_domains/2_b": 100,
                                                "dolma-v1_5/common-crawl": 500, "dolma-v1_5/books": 1000},
                               "paloma-test-0": {"c4_en/c4_en": 200, "c4_100_domains/1_a": 100, "c4_100_domains/2_b": 100,
                                                 "dolma-v1_5/common-crawl": 100, "dolma-v1_5/books": 600}},
              "compare": "weighted_mean_rate", "weights": {"science": 1, "sums": 1, "web": 0, "books": 0, "text": 2}}


class AggregateTest(unittest.TestCase):
    def test_knowledge_en_2_reads_what_knowledge_en_1_reads_and_weighs_paloma_aggregate_as_all_the_answers(self):
        old, new = definition.load("knowledge-en-v1"), definition.load("knowledge-en-v2")
        for section in ("olmes", "paloma", "overlap"):
            self.assertEqual(new[section], old[section])
        self.assertEqual(definition.members(new), definition.members(old))  # the same artifacts serve both
        self.assertEqual(definition.categories(new), definition.categories(old) + ("text",))
        weights, answered = new["weights"], [c for c, parts in new["categories"].items() if "olmes" in parts]
        self.assertEqual({weights[category] for category in answered}, {1})
        self.assertEqual(weights["text"], len(answered))
        self.assertEqual({weights[category] for category in definition.members(new)}, {0})  # shown alone
        old_part, new_part = definition.load("knowledge-en-part-v1"), definition.load("knowledge-en-part-v2")
        self.assertEqual(new_part["within"], "knowledge-en-v2")
        for section in ("olmes", "paloma"):
            self.assertEqual(new_part[section], old_part[section])

    def test_the_bytes_of_each_split_whole_hold_every_member_of_the_text_and_nothing_else(self):
        found = definition.load("knowledge-en-part-v2")
        text = found["paloma"]
        self.assertEqual(set(found["domain_bytes"]), {text["whole"], text["reserve"]["whole"]})
        names = [name for names in definition.members(found).values() for name in names]
        held = lambda key, name: (key.partition("/")[0] == name.partition("/")[0]
                                  and name.partition("/")[2] in ("", key.partition("/")[2]))
        for artifact, sizes in found["domain_bytes"].items():
            for name in names:
                self.assertTrue(any(held(key, name) and size for key, size in sizes.items()) or
                                name == "dolma-v1_5/books" and artifact == text["reserve"]["whole"], (artifact, name))
            self.assertTrue(all(any(held(key, name) for name in names) for key in sizes), artifact)

    def test_paloma_aggregate_is_the_bits_of_all_of_the_text_over_its_bytes(self):
        domains = {"c4_en/c4_en": 1.0, "c4_100_domains/1_a": 2.0, "c4_100_domains/2_b": 4.0,
                   "dolma-v1_5/common-crawl": 5.0, "dolma-v1_5/books": 7.0}
        sizes = AGGREGATED["domain_bytes"]["paloma-val-0"]
        found = definition.read(AGGREGATED, domains)
        self.assertAlmostEqual(found["text"], sum(sizes[key] * bits for key, bits in domains.items()) / 2000)
        self.assertEqual(found["web"], (1.0 + 3.0 + 5.0) / 3)  # the kinds of text are the means of their members still
        with self.assertRaisesRegex(ValueError, "c4_100_domains/3_c is no domain"):
            definition.read(AGGREGATED, {**domains, "c4_100_domains/3_c": 1.0})

    def test_a_reserves_aggregate_weighs_the_domains_that_it_holds_by_the_bytes_of_its_own_split(self):
        held = {"c4_en/c4_en": 1.0, "c4_100_domains/1_a": 2.0, "dolma-v1_5/common-crawl": 5.0, "dolma-v1_5/books": 7.0}
        sizes = AGGREGATED["domain_bytes"]["paloma-test-0"]
        found = definition.read(AGGREGATED, held, partial=True)
        self.assertAlmostEqual(found["text"], (200 * 1.0 + 100 * 2.0 + 100 * 5.0 + 600 * 7.0) / 1000)
        self.assertEqual(sum(sizes.values()) - sizes["c4_100_domains/2_b"], 1000)

    def test_one_result_improves_on_another_by_the_categories_rates_weighted_as_the_definition_weighs_them(self):
        standing = {category: 2.0 for category in definition.categories(AGGREGATED)}
        self.assertAlmostEqual(definition.improvement(AGGREGATED, dict(standing, text=1.0), standing), 2 * 0.5 / 4)
        self.assertAlmostEqual(definition.improvement(AGGREGATED, dict(standing, sums=1.0), standing), 0.5 / 4)
        self.assertEqual(definition.improvement(AGGREGATED, dict(standing, web=1.0), standing), 0)  # shown alone
        fewer = dict(AGGREGATED, weights={"science": 1, "sums": 1, "web": 0, "text": 2})
        with self.assertRaisesRegex(ValueError, "every category"):
            definition.improvement(fewer, standing, standing)


# A measurement of TOY item by item: each task's questions' values and each domain's documents' bits and bytes.
ITEMS = {"questions": {"t1": [1.0, 3.0], "t2": [2.0], "t3": [4.0, 4.0, 4.0], "t4": [5.0, 7.0], "t5": [1.0],
                       "t6": [2.0, 4.0]},
         "documents": {"c4_en/c4_en": [[10.0, 10], [30.0, 10]], "c4_100_domains/1_a": [[6.0, 2]],
                       "dolma-v1_5/common-crawl": [[5.0, 1], [10.0, 2]], "dolma-v1_5/books": [[70.0, 10]]}}


def scaled(items, factor):
    """The same items, each question's value and each document's bits `factor` times as large."""
    return {"questions": {task: [value * factor for value in values] for task, values in items["questions"].items()},
            "documents": {name: [[bits * factor, size] for bits, size in found]
                          for name, found in items["documents"].items()}}


class ComparedTest(unittest.TestCase):
    def test_scores_a_category_from_its_items_as_from_its_tasks_and_domains(self):
        tasks = {"t1": 2.0, "t2": 2.0, "t3": 4.0, "t4": 6.0, "t5": 1.0, "t6": 3.0}
        domains = {"c4_en/c4_en": 2.0, "c4_100_domains/1_a": 3.0, "dolma-v1_5/common-crawl": 5.0,
                   "dolma-v1_5/books": 7.0}
        self.assertEqual(definition.scored(TOY, ITEMS, GROUPS),
                         {**definition.answered(TOY, GROUPS, tasks), **definition.read(TOY, domains)})

    def test_what_the_two_models_share_of_each_items_difficulty_does_not_vary_their_comparison(self):
        found = definition.compared(TOY, scaled(ITEMS, 0.9), ITEMS, GROUPS)
        self.assertAlmostEqual(found["improvement"], 0.1)
        self.assertLess(found["standard_error"], 1e-12)
        for category in ("science", "sums", "web"):
            self.assertAlmostEqual(found["categories"][category]["rate"], 0.1)
            self.assertLess(found["categories"][category]["standard_error"], 1e-12)

    def test_an_error_is_of_how_the_difference_varies_over_the_items_and_the_same_each_time(self):
        one = {"name": "one", "compare": "mean_rate", "better": "higher",
               "categories": {"en": {"instructions": ["en"]}}}
        model = {"prompts": {"en": [1, 0] * 200}}  # 400 prompts, half followed
        standing = {"prompts": {"en": [0.5] * 400}}  # each followed half the times by chance
        found = definition.compared(one, model, standing)
        self.assertEqual(found["improvement"], 0)
        # the share's standard error is 0.5 / sqrt(400), and the rate's that over the standing's 0.5
        self.assertAlmostEqual(found["standard_error"], 0.05, delta=0.005)
        self.assertEqual(found["categories"]["en"]["standard_error"], found["standard_error"])
        self.assertEqual(definition.compared(one, model, standing), found)

    def test_a_category_whose_score_does_not_vary_with_the_items_has_no_standard_error(self):
        found = definition.compared(TOY, scaled(ITEMS, 0.5), ITEMS, GROUPS)
        self.assertAlmostEqual(found["categories"]["books"]["rate"], 0.5)  # a single document
        self.assertIsNone(found["categories"]["books"]["standard_error"])
        self.assertIsNotNone(found["categories"]["web"]["standard_error"])

    def test_measurements_of_other_items_do_not_compare(self):
        fewer = {**ITEMS, "questions": {**ITEMS["questions"], "t1": [1.0]}}
        larger = {**ITEMS, "documents": {**ITEMS["documents"], "dolma-v1_5/books": [[70.0, 11]]}}
        for other in (fewer, larger, {"questions": ITEMS["questions"]}):
            with self.assertRaisesRegex(ValueError, "same items"):
                definition.compared(TOY, other, ITEMS, GROUPS)


# A balance of the same sentences in two languages, the first's bits over the second's.
BALANCE = {"name": "toy-balance", "balance": ["ja", "en"]}


class BalanceTest(unittest.TestCase):
    def test_balance_ja_en_reads_flores_devtest_in_japanese_over_english_and_keeps_both_splits_out_of_training(self):
        chosen = definition.load("balance-ja-en-v1")
        self.assertEqual((chosen["flores"]["repository"], chosen["flores"]["revision"]),
                         (flores.REPOSITORY, flores.REVISION))
        self.assertEqual((chosen["flores"]["split"], chosen["balance"]), ("devtest", ["jpn_Jpan", "eng_Latn"]))
        self.assertEqual(chosen["overlap"]["flores"], {"dev": chosen["overlap"]["flores"]["dev"],
                                                       "devtest": chosen["flores"]["text"]})

    def test_a_balance_is_the_bits_of_the_first_languages_sentences_over_the_seconds(self):
        self.assertEqual(definition.balance(BALANCE, {"ja": [3.0, 6.0], "en": [2.0, 6.0]})["balance"], 9.0 / 8.0)
        with self.assertRaisesRegex(ValueError, "same sentences"):
            definition.balance(BALANCE, {"ja": [3.0], "en": [2.0, 6.0]})

    def test_a_sentence_is_drawn_the_same_in_both_languages(self):
        # every Japanese sentence takes one and a half times the bits of its English one, whichever are drawn
        steady = definition.balance(BALANCE, {"ja": [3.0, 9.0, 1.5, 6.0], "en": [2.0, 6.0, 1.0, 4.0]})
        self.assertAlmostEqual(steady["balance"], 1.5)
        self.assertLess(steady["standard_error"], 1e-12)
        varied = definition.balance(BALANCE, {"ja": [3.0, 6.0, 1.0, 8.0] * 100, "en": [2.0, 6.0, 1.0, 4.0] * 100})
        self.assertGreater(varied["standard_error"], 0)
        self.assertEqual(definition.balance(BALANCE, {"ja": [3.0, 6.0, 1.0, 8.0] * 100,
                                                      "en": [2.0, 6.0, 1.0, 4.0] * 100}), varied)

    def test_two_balances_differ_by_what_varies_with_the_sentences_and_not_by_what_the_models_share(self):
        standing = {"ja": [3.0, 6.0, 1.0, 8.0] * 100, "en": [2.0, 6.0, 1.0, 4.0] * 100}
        halved = {language: [bits / 2 for bits in found] for language, found in standing.items()}
        found = definition.balance_compared(BALANCE, halved, standing)
        self.assertAlmostEqual(found["difference"], 0)  # half the bits in both languages, the same balance
        self.assertLess(found["standard_error"], 1e-12)
        dearer = {"ja": [bits * 1.1 for bits in standing["ja"]], "en": standing["en"]}
        found = definition.balance_compared(BALANCE, dearer, standing)
        balance = definition.balance(BALANCE, standing)
        self.assertAlmostEqual(found["difference"], 0.1 * balance["balance"])
        # with the sentences drawn, the difference varies a tenth as much as the balance does
        self.assertAlmostEqual(found["standard_error"], 0.1 * balance["standard_error"], delta=1e-3)
        with self.assertRaisesRegex(ValueError, "same sentences"):
            definition.balance_compared(BALANCE, {"ja": [3.0], "en": [2.0]}, standing)


class MakeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        asked = lambda count: [dict(request(doc_id, 0, 0, " yes", context=f"Is {doc_id} even?"),
                                    doc={"query": f"Question: is the number {doc_id} even, odd, or neither of them?"})
                               for doc_id in range(count)]
        tasks = {"t1": ({"task_name": "t1", "primary_metric": "bits_per_byte_corr"}, asked(5)),
                 "t2": ({"task_name": "t2", "primary_metric": "bits_per_byte_corr"}, asked(3))}
        self.requests = make_requests(self.directory, tasks, {"first": ["t1", "t2"]})
        self.files = {**members_files("val"), **members_files("test")}
        self.listed = lambda source, split: sorted(p for p in self.files if p.startswith(f"{source}/{split}/"))

    def download(self, path, staging):
        import gzip
        import json
        local = Path(staging) / path.replace("/", "-")
        # no time in the file, as Paloma's files are the same each time they are downloaded
        with gzip.GzipFile(local, "wb", mtime=0) as file:
            file.write("".join(json.dumps({"text": text}) + "\n" for text in self.files[path]).encode("utf-8"))
        return local

    def named(self):
        """A definition of the toy data, with the names that the tools give its artifacts."""
        out = self.directory / "first"
        selection = om.select(self.requests, out, ["first"], per_task=2)
        aside = om.reserve(self.requests, selection, out, per_task=2)
        categories = {"asked": {"olmes": ["first"]}, **{name: {"paloma": parts} for name, parts in CATEGORIES.items()}}
        val = paloma.fetch(out, "val", CATEGORIES, self.download, self.listed)
        test = paloma.fetch(out, "test", CATEGORIES, self.download, self.listed)
        sizes = dict(member_bytes=100, domain_bytes=30, document_bytes=10)
        return {"name": "toy", "compare": "mean_rate", "categories": categories,
                "olmes": {"requests": self.requests.name, "suites": ["first"], "per_task": 2, "seed": 0,
                          "selection": selection.name, "reserve": {"per_task": 2, "seed": 1, "selection": aside.name}},
                "paloma": {"split": "val", "whole": val.name, "part": sizes,
                           "text": paloma.part(val, out, CATEGORIES, **sizes).name,
                           "reserve": {"split": "test", "whole": test.name,
                                       "text": paloma.part(test, out, CATEGORIES, **sizes).name}},
                "overlap": {"paloma": {"val": val.name, "test": test.name},
                            "artifact": overlap.make(out, self.requests, [val, test]).name}}

    def test_makes_every_artifact_that_a_definition_names_under_its_name(self):
        toy = self.named()
        made = definition.make(toy, self.directory / "second", self.requests, self.download, self.listed)
        self.assertEqual(len(made), 7)
        self.assertEqual(set(made), {toy["olmes"]["selection"], toy["olmes"]["reserve"]["selection"],
                                     toy["paloma"]["whole"], toy["paloma"]["text"], toy["paloma"]["reserve"]["whole"],
                                     toy["paloma"]["reserve"]["text"], toy["overlap"]["artifact"]})
        for name, artifact in made.items():
            self.assertEqual(om.check(artifact)["kind"], name.rsplit("-", 1)[0])
        # what is there already is checked and kept: nothing is fetched again
        refuse = lambda *arguments: self.fail("fetched again")
        self.assertEqual(definition.make(toy, self.directory / "second", self.requests, refuse, refuse), made)

    def test_refuses_bytes_of_the_domains_that_are_not_those_of_the_split_whole(self):
        toy = self.named()
        wholes = toy["paloma"]["whole"], toy["paloma"]["reserve"]["whole"]
        sizes = {name: paloma.domain_bytes(self.directory / "first" / name) for name in wholes}
        made = definition.make(dict(toy, domain_bytes=sizes), self.directory / "second", self.requests, self.download,
                               self.listed)
        self.assertIn(wholes[0], made)
        wrong = {**sizes, wholes[1]: {**sizes[wholes[1]], "ptb/ptb": sizes[wholes[1]]["ptb/ptb"] + 1}}
        with self.assertRaisesRegex(ValueError, f"bytes of {wholes[1]}"):
            definition.make(dict(toy, domain_bytes=wrong), self.directory / "second", self.requests, self.download,
                            self.listed)

    def test_refuses_what_comes_out_under_another_name_and_other_requests(self):
        toy = self.named()
        other = dict(toy, olmes=dict(toy["olmes"], selection="selection-0000000000000000"))
        with self.assertRaisesRegex(ValueError, "selection-0000000000000000"):
            definition.make(other, self.directory / "third", self.requests, self.download, self.listed)
        with self.assertRaisesRegex(ValueError, "is made from"):
            definition.make(dict(toy, olmes=dict(toy["olmes"], requests="requests-0")), self.directory / "third",
                            self.requests, self.download, self.listed)


if __name__ == "__main__":
    unittest.main()
