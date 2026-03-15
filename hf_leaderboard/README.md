# HuggingFace Open LLM Leaderboard

This folder contains the **HuggingFace Open LLM Leaderboard v2** data pipeline — from raw CSV extraction to validated Turtle (`.ttl`) triples for the Enki knowledge graph.

## Pipeline

```
HF Datasets API  ──►  hf_leaderboard_long_20260221.csv  ──►  hf_leaderboard.ttl
(hf_extraction.py)     (hf_leaderboard_csv_to_ttl.py)        hf_leaderboard_unmatched.ttl
```

## Files

| File | Description |
|---|---|
| `hf_extraction.py` | Pulls the leaderboard dataset via the HuggingFace `datasets` library and writes it to CSV. |
| `hf_leaderboard_long_20260221.csv` | The dataset in CSV form. This version was pulled on 2026-02-21. |
| `hf_leaderboard_csv_to_ttl.py` | Converts the CSV into two TTL files, split by whether the model's manufacturer org is defined in the Enki ontology. Provisioner triples are emitted only for orgs confirmed as `enki:LLMProvider` in `organizations.ttl`. |
| `hf_leaderboard.ttl` | Triples for models whose manufacturer org exists in the ontology (**known-org** split). |
| `hf_leaderboard_unmatched.ttl` | Triples for models whose manufacturer org is not yet in the ontology (**unmatched-org** split). |
| `hf_leaderboard.ttl.bak` | Backup of the previous `hf_leaderboard.ttl`. |
| `hf_leaderboard_unmatched.ttl.bak` | Backup of the previous `hf_leaderboard_unmatched.ttl`. |
| `hf_leaderboard_report.html` | Turtlelint validation report for `hf_leaderboard.ttl`. |
| `hf_leaderboard_unmatched_report.html` | Turtlelint validation report for `hf_leaderboard_unmatched.ttl`. |

## Benchmarks

The leaderboard evaluates 6 benchmarks plus CO₂ cost:

| Key | Benchmark | Capabilities Tested |
|---|---|---|
| IFEval | IFEval | Instruction Following |
| BBH | BIG-Bench Hard | Multi-Step Reasoning, Formal Reasoning, Causal Reasoning |
| MATH_Lvl_5 | MATH Level 5 | Mathematical Reasoning |
| GPQA | GPQA | Expert-Level Problem Solving, Academic Domain Knowledge, Multi-Step Reasoning |
| MUSR | MUSR | Commonsense Reasoning, Causal Reasoning, Multi-Step Reasoning |
| MMLU_PRO | MMLU-PRO | General Knowledge Reasoning, Academic Domain Knowledge, Factual Recall |
| CO2 | CO₂ cost (kg) | Carbon Cost |

## Usage

```bash
python hf_extraction.py              # pull latest CSV from HF
python hf_leaderboard_csv_to_ttl.py  # generate both TTL files
```
