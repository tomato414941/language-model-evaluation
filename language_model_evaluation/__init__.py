"""What the language-model projects share of their evaluations: the tools that measure a published benchmark as its
publisher defines it and that find the text that overlaps it, and the evaluation definitions that projects adopt.

- olmes_measure (beside this package): AI2's OLMES suites, from the requests that the suites' scores read.
- paloma: AI2's Paloma, its text read as Paloma reads it.
- japanese: Japanese tasks asked as OLMES asks English ones for bits per byte, as a requests artifact of its form.
- japanese_text: Japanese text held out of public corpora, kept as Paloma keeps its text.
- instructions: following instructions as BenchMAX asks and scores it in English and Japanese, by the authors' code.
- flores: FLORES+'s sentences, the same in English and Japanese, by which a model's reading of the two compares.
- overlap: the documents that hold a benchmark's questions or text, which nothing that is measured by it may learn.
- definition: evaluation definitions by name (definitions/NAME.json): the data that is measured, how it is divided
  into categories, or of a balance of two languages how it is read, and how two results compare.

The tools know of no definition. A project chooses the definition that it measures by, and how it reads its own
training text; a definition that is `within` another measures a part of the other's data, so that a result by the
larger one can be compared with it on the part.
"""
