# LM Arena (Chatbot Arena)

This folder contains the **LM Arena** data pipeline — producing two TTL variants (trimmed and augmented) from the Chatbot Arena leaderboard data.

## Pipeline

```
LM Arena Leaderboard  ──►  lmarena.csv  ──►  lmarena.ttl          (trimmed)
                           augmented_lmarena.csv  ──►  augmented_lmarena.ttl  (augmented)
```

## Files

| File | Description |
|---|---|
| `lmarena.csv` | The original dataset from the LM Arena leaderboard. |
| `augmented_lmarena.csv` | Enriched version of `lmarena.csv` — supplemented via Claude API with descriptions, modalities, parameter counts, token limits, and context window sizes. Augmentation was done conservatively; not every model received additional data. |
| `generate_trimmed_ttl_from_csv.py` | Generates `lmarena.ttl` from `lmarena.csv`. |
| `generate_augmented_ttl_from_csv.py` | Generates `augmented_lmarena.ttl` from `augmented_lmarena.csv`. |
| `lmarena.ttl` | Triples for the trimmed dataset. |
| `augmented_lmarena.ttl` | Triples for the augmented dataset — includes additional properties for modalities, parameter counts, token limits, and descriptions where available. |
| `lmarena.ttl.bak` | Backup of the previous `lmarena.ttl`. |
| `augmented_lmarena.ttl.bak` | Backup of the previous `augmented_lmarena.ttl`. |
| `lmarena_trimmed_report.html` | Turtlelint validation report for `lmarena.ttl`. |
| `lmarena_augmented_report.html` | Turtlelint validation report for `augmented_lmarena.ttl`. |

## Augmented Fields

The augmented CSV adds the following columns (where data was available):

- `description` — one-sentence model description
- `input_modalities` / `output_modalities` — e.g. Text, Image, Audio
- `num_params` — total parameter count
- `max_input_tokens` / `max_output_tokens` — token limits
- `context_window` — context window size in tokens

## Usage

```bash
python generate_trimmed_ttl_from_csv.py    # generate trimmed TTL
python generate_augmented_ttl_from_csv.py  # generate augmented TTL
```
