This folder uses the Mistral API and docs structure to parse models, collect information, and generate JSON for the current selection of Mistral models.

**model_list_data.py** 
Python script that generates both JSON files using the MISTRAL_API_KEY and by scraping model detail pages from docs.mistral.ai. 

**mistral_models_raw.json**
This file contains only the raw information accessible from the Mistral /v1/models API endpoint. 

**mistral_models.json**
This file contains enriched information scraped from each model's documentation page. Interesting data includes: context window size, pricing, parameter counts, modalities, features, license tier (OPEN/PREMIER/LABS), and more.

## Setup

```bash
export MISTRAL_API_KEY="your-key-here"
python model_list_data.py
```

## Data Sources

1. **Mistral API** (`https://api.mistral.ai/v1/models`) — lists accessible models with basic metadata (context length, capabilities, deprecation info, temperature defaults).
2. **Mistral Docs** (`https://docs.mistral.ai/getting-started/models`) — index page used to discover all model doc page URLs.
3. **Individual model pages** (e.g. `https://docs.mistral.ai/models/mistral-large-3-25-12`) — scraped for description, pricing, context window, modalities, features, parameter counts, version, release date, and license tier.
