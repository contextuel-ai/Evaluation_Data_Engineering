# Fireworks AI

This folder uses the Fireworks AI API to retrieve model metadata and generate JSON for the current selection of Fireworks-hosted models.

## Files

| File | Description |
|---|---|
| `model_data_list.py` | Python script that queries the Fireworks API and writes both JSON files. Infers modalities and model family from naming conventions. |
| `fireworks_models_raw.json` | Raw model metadata as returned by the Fireworks API. |
| `fireworks_models.json` | Enriched model records with inferred modalities, context length, tool/LoRA support, and links to HuggingFace/GitHub. |

## Setup

```bash
export FIREWORKS_API_KEY="your-key-here"
python model_data_list.py
```

## Data Sources

1. **Fireworks API** (`https://api.fireworks.ai/v1/accounts/fireworks/models`) — lists accessible models with metadata including display name, description, context length, conversation config, and capability flags.
