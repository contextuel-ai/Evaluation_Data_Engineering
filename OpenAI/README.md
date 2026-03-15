# OpenAI

This folder uses the OpenAI API and documentation structure to parse models, collect metadata, and generate JSON for the current selection of OpenAI models.

## Files

| File | Description |
|---|---|
| `model_list_data.py` | Python script that generates both JSON files using the `OPENAI_API_KEY` and by scraping data from OpenAI's documentation pages. |
| `openai_models_raw.json` | Raw model metadata as returned by the OpenAI `/v1/models` API endpoint. |
| `openai_models.json` | Enriched model data scraped from OpenAI documentation. Includes context window size, speed, knowledge cutoff, and more. |

## Setup

```bash
export OPENAI_API_KEY="your-key-here"
python model_list_data.py
```

## Data Sources

1. **OpenAI API** (`https://api.openai.com/v1/models`) — lists accessible models with basic metadata.
2. **OpenAI Documentation** — individual model pages scraped for context window, pricing, knowledge cutoff, and capabilities.
