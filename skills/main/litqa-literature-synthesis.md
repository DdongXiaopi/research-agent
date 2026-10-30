# litqa-literature-synthesis

Use when a biomedical question must be answered from published literature.

Procedure
1. Extract the entities (gene, protein, organism, drug, assay) and the exact property being asked.
2. Retrieve sources that report that property directly; prefer primary research with a DOI.
3. Compare each answer option against the retrieved passage; discard options only supported by
   surface similarity to the question wording.
4. State the chosen option and the source identifier that supports it.
