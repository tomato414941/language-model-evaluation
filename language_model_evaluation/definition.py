"""Evaluation definitions: what a project measures by, under a name.

A definition (definitions/NAME.json) says
- the data that is measured, as the tools make it and as the artifacts that they make are named: OLMES's questions
  (olmes_measure.select: the suites that the categories count, or with `selected` those and others that a project
  measures besides; every question or so many of each task), Paloma's text (paloma.fetch: which split; paloma.part:
  how much of it), and what is kept aside from decisions, a reserve, if any; and in Japanese, beside them, the
  questions of japanese.py (`japanese`, selected as OLMES's) and the text of japanese_text.py (`japanese_text`, cut
  as Paloma's);
- the text that nothing measured by it may learn (overlap.make: every question of the requests, Paloma's splits
  whole), by the artifact's name, and the Japanese questions' and text's under `japanese`;
- the categories that a result is divided into, each the mean of its members: OLMES's or the Japanese tasks and
  groups, a named mean of some tasks, or the sources and domains of Paloma's or the Japanese text; or, under
  `paloma_aggregate`, the bits per byte of all of Paloma's text, the domains weighed by their bytes in the split
  whole (definitions/bytes, paloma.domain_bytes), as Paloma's own aggregate weighs them; and
- how two results compare (`improvement`): by the mean of the categories' rates of improvement, with `languages` by
  the mean of each language's, or by their mean weighted by the definition's `weights`, in which a category of
  weight 0 is shown and decides nothing; a score improves as it falls (bits per byte) unless the definition says
  that a higher one is `better`, as the share of prompts whose instructions a model follows (`followed`): of
  BenchMAX's prompts (`instructions`, instructions.py), of IFBench's (`ifbench`, ifbench.py) or of M-IFEval's
  (`mifeval`, mifeval.py).

A definition of a balance (`balance`: two languages) measures instead the same sentences in both, FLORES+'s (`flores`:
flores.fetch's split), with the text that nothing measured by it may learn under `overlap` (flores.make_overlap: both
splits whole). A model's balance is the bits that it takes to read the sentences in the first language over those
that it takes in the second, and two models' balances compare by their difference (`balance_compared`).

A comparison's standard errors (`compared`) come from two measurements item by item, each question's value, each
document's bits and each prompt's result (`scored` makes the categories of one): how much the comparison varies as
the items measured are drawn again. A balance's come from each sentence's bits in each language, the sentences drawn
again the same in both.

A definition that is `within` another measures a part of the other's data and says only what differs, so that the
two agree on everything else and a result by the larger one can be compared on the part: OLMES's questions by
olmes_measure.within, Paloma's text by reading the part.

    python -m language_model_evaluation.definition make NAME OUT [REQUESTS]   # the definition's artifacts, in OUT
    python -m language_model_evaluation.definition show NAME
    python -m language_model_evaluation.definition bytes ARTIFACT   # a split whole's domains' bytes, as definitions/bytes

`make` builds what OUT does not hold yet and checks every name against the definition's, from OLMES's requests
(REQUESTS) for a definition of questions; fetching Paloma and FLORES+ needs HF_TOKEN (paloma.py, flores.py).
`make_japanese` builds the Japanese artifacts alone, from the files of japanese.py's and japanese_text.py's sources,
and `make_flores` FLORES+'s.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import olmes_measure

from . import flores, japanese, japanese_text, overlap, paloma

DEFINITIONS = Path(__file__).parent / "definitions"
AGGREGATE = "paloma_aggregate"  # a category of all of Paloma's text, the domains weighed by their bytes
# The sources of prompts whose instructions a model follows: BenchMAX's (instructions.py), IFBench's (ifbench.py) and
# M-IFEval's (mifeval.py).
PROMPTS = ("instructions", "ifbench", "mifeval")


def load(name):
    """The definition of a name, with what it takes from the one that it is within, and, if a category is Paloma's
    aggregate, the bytes of each domain of each split whole that it reads (`domain_bytes`: {artifact: {domain:
    bytes}})."""
    own = json.loads((DEFINITIONS / f"{name}.json").read_text(encoding="utf-8"))
    if "within" not in own:
        merged = {"name": name, **own}
    else:
        whole = load(own["within"])
        merged = {**whole, **own, "name": name}
        for section in ("olmes", "paloma", "japanese", "japanese_text", *PROMPTS):
            if section in whole:
                merged[section] = {**whole[section], **own.get(section, {})}
    if any(AGGREGATE in parts for parts in merged.get("categories", {}).values()):
        merged["domain_bytes"] = {artifact: json.loads((DEFINITIONS / "bytes" / f"{artifact}.json").read_text())
                                  for artifact in _wholes(merged)}
    return merged


def _wholes(definition):
    """The split wholes of Paloma's text that a definition reads a part of, or all of: its own and its reserve's."""
    text = definition["paloma"]
    return [text["whole"]] + ([text["reserve"]["whole"]] if "reserve" in text else [])


def members(definition, source="paloma"):
    """The members of each category that is read of a source of text (paloma or japanese_text): {category: [member]},
    as paloma.fetch and paloma.part take them."""
    return {category: list(parts[source]) for category, parts in definition["categories"].items() if source in parts}


def _have(out, made, name, expected, build):
    """The artifact that a definition names, in `out`: checked if it is there, made by `build` if not."""
    if (out / expected).exists():
        olmes_measure.check(out / expected)
    else:
        artifact = build()
        if artifact.name != expected:
            raise ValueError(f"{name} names {expected}, and what was made is {artifact.name}")
    made[expected] = out / expected
    return out / expected


def make(name, out, requests, download=paloma.download, listed=paloma.listed):
    """Every artifact that a definition (its name, or one already loaded) names, in the directory `out`, from an
    olmes-measure requests artifact and Paloma's files: {name: its directory}. One that `out` holds already is
    checked and kept; one that is made must come out under the definition's name."""
    definition, out, made = load(name) if isinstance(name, str) else name, Path(out), {}
    name = definition["name"]
    have = lambda expected, build: _have(out, made, name, expected, build)
    olmes, text, kinds = definition["olmes"], definition["paloma"], members(definition)
    if Path(requests).name != olmes["requests"]:
        raise ValueError(f"{name} is made from {olmes['requests']}")
    selection = have(olmes["selection"], lambda: olmes_measure.select(
        requests, out, olmes.get("selected", olmes["suites"]), olmes.get("per_task"), olmes.get("seed", 0)))
    if "reserve" in olmes:
        aside = olmes["reserve"]
        have(aside["selection"], lambda: olmes_measure.reserve(requests, selection, out, aside["per_task"],
                                                               aside["seed"]))
    fetch = lambda split: lambda: paloma.fetch(out, split, kinds, download, listed)
    whole = have(text["whole"], fetch(text["split"]))
    if text.get("part"):
        have(text["text"], lambda: paloma.part(whole, out, kinds, **text["part"]))
    if "reserve" in text:
        aside = text["reserve"]
        other = have(aside["whole"], fetch(aside["split"]))
        have(aside["text"], lambda: paloma.part(other, out, kinds, **text["part"]))
    splits = [have(artifact, fetch(split)) for split, artifact in definition["overlap"]["paloma"].items()]
    have(definition["overlap"]["artifact"], lambda: overlap.make(out, requests, splits))
    for artifact, sizes in definition.get("domain_bytes", {}).items():
        if paloma.domain_bytes(made[artifact]) != sizes:
            raise ValueError(f"{name}'s bytes of {artifact} are not the artifact's")
    return {**made, **make_japanese(definition, out)}


def make_japanese(name, out, raw=japanese.RAW):
    """Every Japanese artifact that a definition names, in the directory `out`, from the files of the Japanese
    questions' and text's sources, kept in `raw`: {name: its directory}, as `make` checks and keeps them."""
    definition, out, made = load(name) if isinstance(name, str) else name, Path(out), {}
    have = lambda expected, build: _have(out, made, definition["name"], expected, build)
    if "japanese" in definition:
        asked = definition["japanese"]
        requests = have(asked["requests"], lambda: japanese.make(out, raw))
        selection = have(asked["selection"], lambda: olmes_measure.select(
            requests, out, asked.get("selected", asked["suites"]), asked.get("per_task"), asked.get("seed", 0)))
        if "reserve" in asked:
            aside = asked["reserve"]
            have(aside["selection"], lambda: olmes_measure.reserve(requests, selection, out, aside["per_task"],
                                                                   aside["seed"]))
    fetch = lambda split: lambda: japanese_text.fetch(out, split, raw)
    if "japanese_text" in definition:
        text, kinds = definition["japanese_text"], members(definition, "japanese_text")
        whole = have(text["whole"], fetch(text["split"]))
        if text.get("part"):
            have(text["text"], lambda: paloma.part(whole, out, kinds, **text["part"]))
        if "reserve" in text:
            aside = text["reserve"]
            other = have(aside["whole"], fetch(aside["split"]))
            have(aside["text"], lambda: paloma.part(other, out, kinds, **text["part"]))
    if "japanese" in definition.get("overlap", {}):
        aside = definition["overlap"]["japanese"]
        splits = [have(artifact, fetch(split)) for split, artifact in aside["text"].items()]
        have(aside["artifact"], lambda: overlap.make(out, requests, splits))
    return made


def make_flores(name, out, download=flores.download):
    """Every artifact of FLORES+ that a definition of a balance names, in the directory `out`: {name: its
    directory}, as `make` checks and keeps them."""
    definition, out, made = load(name) if isinstance(name, str) else name, Path(out), {}
    have = lambda expected, build: _have(out, made, definition["name"], expected, build)
    fetch = lambda split: lambda: flores.fetch(out, split, download)
    have(definition["flores"]["text"], fetch(definition["flores"]["split"]))
    splits = [have(artifact, fetch(split)) for split, artifact in definition["overlap"]["flores"].items()]
    have(definition["overlap"]["artifact"], lambda: flores.make_overlap(out, splits))
    return made


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def answered(definition, groups, scores, partial=False):
    """The score of each category that is answered, from each task's score ({task: score}) and the groups in which
    the selections hold the suites' tasks, OLMES's and the Japanese ones together. A member is a task, a group (the
    mean of its members, as OLMES averages them), or a named mean of some tasks ({name: [task]}). The categories
    count every task of the definition's suites once; a selection that is `partial`, as a reserve is, may lack
    tasks, and its categories are of the others."""

    def tasks(member):
        return list(*member.values()) if isinstance(member, dict) else olmes_measure.leaves(groups, member)

    def score(member):
        if isinstance(member, dict):
            return _mean(scores[task] for task in tasks(member) if task in scores)
        return _mean(score(inner) for inner in groups[member]) if member in groups else scores[member]

    sources = ("olmes", "japanese")
    sets = {category: parts[source] for category, parts in definition["categories"].items()
            for source in sources if source in parts}
    counted = [task for parts in sets.values() for member in parts for task in tasks(member)]
    suites = [task for source in sources if source in definition for suite in definition[source]["suites"]
              if suite in groups for task in olmes_measure.leaves(groups, suite)]
    asked = {task for task in suites if task in scores}
    if len(set(counted)) != len(counted) or not asked <= set(counted):
        raise ValueError("the categories do not count every task of the suites once")
    if not partial and set(counted) != asked:
        raise ValueError("the categories count tasks that the selection does not hold")
    held = lambda member: any(task in scores for task in tasks(member))
    return {category: _mean(score(member) for member in parts if held(member)) for category, parts in sets.items()}


def read(definition, domains, partial=False):
    """The score of each category that is read, from each domain's bits per byte ({SOURCE/DOMAIN: BPB}), of Paloma's
    and of the Japanese text together:
    a member is its one domain or the mean of its source's, and a category the mean of its members. Every member is
    scored, unless the text is `partial`, as a reserve's may be: its categories are of the members that it holds."""

    def member(name):
        source, _, one = name.partition("/")
        found = [bits for key, bits in domains.items()
                 if key.partition("/")[0] == source and one in ("", key.partition("/")[2])]
        if not found and not partial:
            raise ValueError(f"no domain of {name} is scored")
        return sum(found) / len(found) if found else None

    scores = {}
    for category, names in {**members(definition), **members(definition, "japanese_text")}.items():
        found = [bits for bits in map(member, names) if bits is not None]
        if not found:
            raise ValueError(f"no member of {category} is scored")
        scores[category] = sum(found) / len(found)
    for category, parts in definition["categories"].items():
        if AGGREGATE in parts:
            scores[category] = _aggregate(definition, domains, partial)
    return scores


def _aggregate(definition, domains, partial):
    """The bits per byte of all of Paloma's text from its domains' ({SOURCE/DOMAIN: BPB}): their mean weighed by each
    domain's bytes in the split whole, which a part of it estimates. A `partial` text is the reserve's, of the
    reserve's split, and its domains are those that it holds."""
    text = definition["paloma"]
    split = text["reserve"]["whole"] if partial and "reserve" in text else text["whole"]
    sizes = definition["domain_bytes"][split]
    sources = {name.partition("/")[0] for names in members(definition).values() for name in names}
    stray = sorted(key for key in domains if key.partition("/")[0] in sources and key not in sizes)
    if stray:
        raise ValueError(f"{stray[0]} is no domain of {split}")
    found = {key: bits for key, bits in domains.items() if key in sizes}
    missing = sorted(key for key in sizes if key not in found and sizes[key])
    if missing and not partial:
        raise ValueError(f"no score of {missing[0]}")
    total = sum(sizes[key] for key in found)
    if not total:
        raise ValueError("no domain of Paloma's text is scored")
    return sum(sizes[key] * bits for key, bits in found.items()) / total


def prompts_name(source, language):
    """The name of a language's prompts of a source of PROMPTS, by which a measurement holds their results: the
    language's for BenchMAX's, SOURCE/LANGUAGE for another's."""
    return language if source == "instructions" else f"{source}/{language}"


def followed(definition, shares):
    """The score of each category of instructions, from the share of each source's prompts of a language whose every
    instruction a model follows at the source's level ({prompts_name: share}, as each source's `score` gives them)."""
    return {category: _mean(shares[prompts_name(source, language)] for source in PROMPTS
                            for language in parts.get(source, ()))
            for category, parts in definition["categories"].items() if any(source in parts for source in PROMPTS)}


def categories(definition):
    """The categories' names, in the definition's order."""
    return tuple(definition["categories"])


def rates(definition, scores, standing):
    """Each category's rate of improvement of `scores` on `standing` ({category: score}, of every category): the fall
    of the score, 1 - score / standing, or, when a higher score is `better`, its rise, score / standing - 1. Positive:
    `scores` are the better."""
    if definition.get("better") == "higher":
        return {category: scores[category] / standing[category] - 1 for category in categories(definition)}
    return {category: 1 - scores[category] / standing[category] for category in categories(definition)}


def improvement(definition, scores, standing):
    """By how much `scores` improve on `standing` ({category: score}, of every category), from the categories' `rates`:
    under the rule `mean_rate`, their mean, each counting equally; under `weighted_mean_rate`, their mean weighted by
    the definition's `weights` ({category: weight}), in which a category of weight 0 decides nothing; under
    `mean_rate_by_language`, the mean of each language's mean of its categories' (`languages`: {language:
    [category]}), so that each language counts equally however many categories it has. Positive: `scores` are the
    better."""
    found = rates(definition, scores, standing)
    if definition["compare"] == "mean_rate":
        return _mean(found.values())
    if definition["compare"] == "weighted_mean_rate":
        weights = definition["weights"]
        if sorted(weights) != sorted(categories(definition)):
            raise ValueError("the weights are not of every category once")
        return sum(weights[category] * rate for category, rate in found.items()) / sum(weights.values())
    if definition["compare"] == "mean_rate_by_language":
        languages = definition["languages"]
        if sorted(c for names in languages.values() for c in names) != sorted(categories(definition)):
            raise ValueError("the languages do not hold every category once")
        return _mean(_mean(found[category] for category in names) for names in languages.values())
    raise ValueError(f"no rule {definition['compare']}")


def _arrays(items):
    """A measurement's items, each kind's by what it scores (a task, a domain or a language), as (each item's value,
    its size) in arrays: a question's or a prompt's value is of size one, a document's bits are of its bytes."""
    arrays = {}
    for kind, held in items.items():
        if kind == "documents":
            arrays[kind] = {name: tuple(np.array(each, dtype=float).reshape(-1, 2).T) for name, each in held.items()}
        elif kind in ("questions", "prompts"):
            arrays[kind] = {name: (np.array(each, dtype=float), np.ones(len(each))) for name, each in held.items()}
        else:
            raise ValueError(f"no items of the kind {kind}")
    return arrays


def _categorized(definition, groups, scores):
    """Each category's score from the scores of what the items score, of each kind ({kind: {name: score}})."""
    found = {}
    if "questions" in scores:
        found |= answered(definition, groups, scores["questions"])
    if "documents" in scores:
        found |= read(definition, scores["documents"])
    if "prompts" in scores:
        found |= followed(definition, scores["prompts"])
    return {category: found[category] for category in categories(definition)}


def scored(definition, items, groups=None):
    """Each category's score from a model's measurement item by item: `items` holds, of what the definition measures,
    {"questions": {task: [each question's value]}, "documents": {SOURCE/DOMAIN: [[each document's bits, its bytes]]},
    "prompts": {language: [each prompt's result, 1 if its every instruction is followed]}}, and `groups` are those of
    the selections whose tasks it holds. A task's score is the mean of its questions' values, a domain's the bits of
    its documents over their bytes and a language's the mean of its prompts' results, from which `answered`, `read`
    and `followed` make the categories."""
    return _categorized(definition, groups, {
        kind: {name: float(values.sum() / sizes.sum()) for name, (values, sizes) in held.items()}
        for kind, held in _arrays(items).items()})


def _resampled(generator, columns, draws):
    """The sums of columns of the same items over `draws` resamples of the items, each drawn again as many times as
    there are items, with replacement and the same for every column: an array of the draws' sums for each column."""
    size = len(columns[0])
    sums = [np.empty(draws) for _ in columns]
    step = max(1, 2 ** 22 // size)  # draws at a time, so that their items' indices take at most 32 MB
    for start in range(0, draws, step):
        picked = generator.integers(0, size, (min(step, draws - start), size))
        for total, column in zip(sums, columns):
            total[start:start + len(picked)] = column[picked].sum(axis=1)
    return sums


def compared(definition, items, standing, groups=None, draws=1000, seed=0):
    """How a model's measurement compares with another's of the same items (`items` and `standing`, as `scored` takes
    them): the `improvement` and each category's rate (`rates`), with the standard error of each over `draws`
    resamples of the items. A resample draws each task's questions, each domain's documents and each language's
    prompts again, as many as it holds, with replacement and the same for both measurements, so that an error is of
    the difference between the two on the same items: how much it varies with which items happen to be measured, and
    not with how the models happen to be trained. A category whose score does not vary with the items, as one of a
    single document (Paloma's PTB) does not, has no standard error (None), and adds nothing to the improvement's."""
    ours, theirs = _arrays(items), _arrays(standing)
    same = ours.keys() == theirs.keys() and all(
        ours[kind].keys() == theirs[kind].keys()
        and all(np.array_equal(ours[kind][name][1], theirs[kind][name][1]) for name in ours[kind]) for kind in ours)
    if not same:
        raise ValueError("the measurements are not of the same items")
    generator = np.random.default_rng(seed)
    drawn = {kind: {} for kind in ours}, {kind: {} for kind in ours}
    for kind in sorted(ours):
        for name in sorted(ours[kind]):
            (values, sizes), (other_values, _) = ours[kind][name], theirs[kind][name]
            first, second, size = _resampled(generator, (values, other_values, sizes), draws)
            drawn[0][kind][name], drawn[1][kind][name] = first / size, second / size
    resampled = [_categorized(definition, groups, scores) for scores in drawn]
    varied = rates(definition, *resampled)
    error = lambda found: float(np.std(found, ddof=1))
    measured = scored(definition, items, groups), scored(definition, standing, groups)
    return {"improvement": improvement(definition, *measured),
            "standard_error": error(improvement(definition, *resampled)),
            "categories": {category: {"rate": rate, "standard_error": error(varied[category]) if any(
                np.ptp(scores[category]) > 0 for scores in resampled) else None}
                for category, rate in rates(definition, *measured).items()}}


def _sentences(definition, bits):
    """Each sentence's bits in the two languages of a balance ({language: [bits]}), as arrays in the definition's
    order of them."""
    found = [np.array(bits[language], dtype=float) for language in definition["balance"]]
    if len(found[0]) != len(found[1]):
        raise ValueError("the two languages do not hold the same sentences")
    return found


def balance(definition, bits, draws=1000, seed=0):
    """A model's balance by a definition of the same sentences in two languages, from each sentence's bits
    ({language: [bits]}, the sentences in the same order in both): the bits that it takes to read the sentences in the
    first of the definition's two languages over those that it takes in the second, above 1 when the first costs it
    more for the same content. Its standard error is over `draws` resamples of the sentences, each drawn again as many
    times as there are sentences, with replacement and the same in both languages: how much the balance varies with
    which sentences happen to be measured."""
    first, second = _sentences(definition, bits)
    drawn = _resampled(np.random.default_rng(seed), (first, second), draws)
    return {"balance": float(first.sum() / second.sum()), "standard_error": float(np.std(drawn[0] / drawn[1], ddof=1))}


def balance_compared(definition, bits, standing, draws=1000, seed=0):
    """How a model's balance differs from another's, measured on the same sentences (`bits` and `standing`, as
    `balance` takes them): its balance less the other's, with the standard error of that difference over `draws`
    resamples of the sentences, drawn the same in both languages and for both models, so that it is of how the
    difference varies with which sentences happen to be measured, and not with what the models share."""
    ours, theirs = _sentences(definition, bits), _sentences(definition, standing)
    if len(ours[0]) != len(theirs[0]):
        raise ValueError("the measurements are not of the same sentences")
    differ = lambda first, second, other_first, other_second: first / second - other_first / other_second
    drawn = _resampled(np.random.default_rng(seed), (*ours, *theirs), draws)
    return {"difference": float(differ(*(column.sum() for column in (*ours, *theirs)))),
            "standard_error": float(np.std(differ(*drawn), ddof=1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("make")
    made.add_argument("name")
    made.add_argument("out")
    made.add_argument("requests", nargs="?")
    shown = commands.add_parser("show")
    shown.add_argument("name")
    sized = commands.add_parser("bytes", help="the bytes of each domain of a Paloma artifact, as definitions/bytes")
    sized.add_argument("artifact")
    args = parser.parse_args()
    if args.command == "bytes":
        olmes_measure.check(args.artifact)
        print(json.dumps(paloma.domain_bytes(args.artifact), indent=0, sort_keys=True))
        return
    if args.command == "make":
        chosen = load(args.name)
        if "flores" not in chosen and args.requests is None:
            parser.error(f"{args.name} is made from OLMES's requests")
        for artifact in make_flores(chosen, args.out) if "flores" in chosen else make(chosen, args.out, args.requests):
            print(artifact)
    else:
        print(json.dumps(load(args.name), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
