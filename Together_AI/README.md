# Together AI

This folder uses the Together AI API and documentation to parse models, collect metadata, and generate JSON for the current selection of Together-hosted models.

## Files

| File | Description |
|---|---|
| `models_list_data.py` | Python script that queries the Together API and scrapes documentation pages to produce both JSON files. |
| `together_models_raw.json` | Raw model metadata as returned by the Together `/v1/models` API endpoint. |
| `together_models.json` | Enriched model records combining API metadata with documentation scraping (pricing, context window, modalities, etc.). |

## Setup

```bash
export TOGETHER_API_KEY="your-key-here"
python models_list_data.py
```

## Data Sources

1. **Together API** (`https://api.together.xyz/v1/models`) — lists accessible models with metadata.
2. **Together Docs** (`https://docs.together.ai/docs/serverless-models`) — index page for discovering model documentation.
3. **OpenAI Compatibility Docs** (`https://docs.together.ai/docs/openai-api-compatibility`) — additional compatibility information.
