# Experiment inputs

The bundled files are the inputs used in the paper. Keep their row order:
it determines the seeded data splits. No new collection or API key is needed.

## Coffee

`coffee/raw/coffee_survey.csv` is the Great American Coffee Taste Test survey
snapshot from [TidyTuesday, May 14, 2024](https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-05-14).
It includes the public survey's demographic fields and optional free text.

The original survey was organized by James Hoffmann and Cometeer in October
2023. TidyTuesday distributes the cleaned snapshot and its data dictionary;
its repository carries a [CC0 1.0 license](https://github.com/rfordatascience/tidytuesday/blob/main/LICENSE).
We preserve the snapshot unchanged and do not apply the experiment code's MIT
license to it. The file's SHA-256 is
`4ae99df492485239269e5dc6562f6f4d418d0ad98ff00d63e9640b2854de98ef`.

The adapter keeps 3,744 respondents with all four coffee ratings and maps
ratings to utilities with `(rating - 1) / 4`. A 375-respondent discovery sample
leaves 3,369 for analysis. Group definitions and seeds are in
[configs/coffee.json](../configs/coffee.json).

## Persona

| File under `persona/` | Contents |
|---|---|
| `ratings.csv` | 9,984 synthetic ratings |
| `study.json` | Answer texts, persona cells, and collection settings |
| `prompt.txt` | Rating prompt |
| `requests.csv` | Collection request design |
| `protocol.json` | Input hashes and analysis protocol |

Claude Haiku 4.5 rated six answers under 16 persona cells, with 104 draws per
cell and answer. Utilities are `(score - 1) / 9`. Independently permuted answer
draws form 1,664 artificial respondents. A cell-stratified discovery sample
reserves 167, leaving 1,497 for analysis. Group selection uses the discovery
sample. These synthetic ratings do not measure human demographic preferences.
Analysis settings are in [configs/persona.json](../configs/persona.json).

These materials were created for this paper by Jacob Brodkey, Roberto Tamez,
and Aaron Roth and are released under [CC BY 4.0](../LICENSE-DATA.md).
The frozen rating file's SHA-256 is
`921963f45f3aed195a1da7ecca17f1548b4b6c625aed432b7f438cbf3160194f`.
The bundled prompts and request design document the collection; rerunning the
analysis uses the saved ratings and does not issue new model API requests.

Some input and result metadata retains identifiers such as Coffee-v4,
Persona fixed-v8, and v10. These identify analysis or collection versions;
the package contains one input dataset for each study.
