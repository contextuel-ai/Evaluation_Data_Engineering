# Artificial Analysis

This folder handles the **Artificial Analysis** dataset — extracting model data via API and converting it to Turtle triples for the Enki knowledge graph.

## Pipeline

```
Artificial Analysis API  ──►  aa_models_with_pricing_speed_and_benchmarks.csv  ──►  aa_models.ttl
(aa_extraction.py)             (aa_csv_to_ttl.py)
```

## Files

| File | Description |
|---|---|
| `aa_extraction.py` | Extracts the dataset from Artificial Analysis using an API key. Writes `aa_models_with_pricing_speed_and_benchmarks.csv`. |
| `aa_csv_to_ttl.py` | Converts the CSV into Turtle triples. Emits benchmarks for latency, throughput, time-to-first-token, pricing, and quality scores. |
| `aa_models_with_pricing_speed_and_benchmarks.csv` | The dataset in CSV format. Contains 406 models with evaluations for latency, cost, and quality. |
| `aa_models.ttl` | The generated Turtle triples for use in Enki. |
| `aa_models.ttl.bak` | Backup of the previous `aa_models.ttl`. |
| `aa_models_report.html` | Turtlelint validation report for `aa_models.ttl`. |

## Usage

```bash
python aa_extraction.py  # pull latest data from Artificial Analysis API
python aa_csv_to_ttl.py  # generate TTL from CSV
```
