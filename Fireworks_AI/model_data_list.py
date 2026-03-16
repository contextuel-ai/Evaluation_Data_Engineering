import os
import json
from pathlib import Path
from typing import Any, Dict, List

import requests

# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "fireworks_models.json"
RAW_OUTPUT_PATH = SCRIPT_DIR / "fireworks_models_raw.json"

# =========================================================
# Config
# =========================================================

FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]

API_MODELS_URL = "https://api.fireworks.ai/v1/accounts/fireworks/models"

HEADERS_API = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Accept": "application/json",
}

# =========================================================
# Helpers
# =========================================================

def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================================================
# API access
# =========================================================

def list_accessible_fireworks_models() -> List[Dict[str, Any]]:
    """
    Returns all Fireworks models visible to the API key.
    """

    resp = requests.get(
        API_MODELS_URL,
        headers=HEADERS_API,
        params={"pageSize": 200},
        timeout=(10, 60),
    )

    if resp.status_code == 401:
        raise RuntimeError(f"401 Unauthorized from Fireworks:\n{resp.text}")

    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, dict):

        if "models" in payload:
            return payload["models"]

        if "data" in payload:
            return payload["data"]

    raise RuntimeError(f"Unexpected Fireworks payload structure: {type(payload)}")


# =========================================================
# Metadata inference
# =========================================================

def derive_modalities(model: Dict[str, Any]) -> Dict[str, Any]:

    name = str(model.get("name", "")).lower()

    input_modalities = ["text"]
    output_modalities = ["text"]
    family = "chat_or_completion"

    if "embedding" in name:
        family = "embedding"
        output_modalities = ["embedding"]

    if "vision" in name or "vl" in name:
        family = "multimodal"
        input_modalities = ["text", "image"]

    if "audio" in name or "whisper" in name:
        family = "audio"
        input_modalities = ["audio"]
        output_modalities = ["text"]

    return {
        "model_family_guess": family,
        "input_modalities": input_modalities,
        "output_modalities": output_modalities,
    }


# =========================================================
# Record construction
# =========================================================

def enrich_fireworks_model(model: Dict[str, Any]) -> Dict[str, Any]:

    derived = derive_modalities(model)

    conversation_config = model.get("conversationConfig") or {}

    return {
        "model_id": model.get("name"),
        "display_name": model.get("displayName"),
        "description": model.get("description"),
        "state": model.get("state"),
        "kind": model.get("kind"),
        "context_length": conversation_config.get("contextLength"),
        "supports_image_input": model.get("supportsImageInput"),
        "supports_tools": model.get("supportsTools"),
        "supports_lora": model.get("supportsLora"),
        "tunable": model.get("tunable"),
        "github_url": model.get("githubUrl"),
        "huggingface_url": model.get("huggingFaceUrl"),
        **derived,
    }


def build_fireworks_model_records(api_models: List[Dict[str, Any]]) -> Dict[str, Any]:

    records = []

    for model in api_models:

        enriched = enrich_fireworks_model(model)

        record = {
            "provider": "fireworks_ai",
            "model_id": model.get("name"),
            "api_listed": True,
            "provider_model_url": f"https://api.fireworks.ai/v1/accounts/fireworks/models/{model.get('name')}",
            **enriched,
        }

        records.append(record)

    return {
        "provider": "fireworks_ai",
        "model_list_api_url": API_MODELS_URL,
        "model_count": len(records),
        "models": records,
    }


# =========================================================
# Raw payload
# =========================================================

def build_raw_payload() -> Any:
    return list_accessible_fireworks_models()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    api_models = list_accessible_fireworks_models()

    write_json(api_models, RAW_OUTPUT_PATH)

    enriched_payload = build_fireworks_model_records(api_models)

    write_json(enriched_payload, OUTPUT_PATH)

    print(f"Wrote raw API JSON to: {RAW_OUTPUT_PATH}")
    print(f"Wrote enriched JSON to: {OUTPUT_PATH}")