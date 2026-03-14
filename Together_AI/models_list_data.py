import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# =========================================================
# Paths: write output in the SAME folder as this script
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = SCRIPT_DIR / "together_models.json"
RAW_OUTPUT_PATH = SCRIPT_DIR / "together_models_raw.json"

# =========================================================
# Config
# =========================================================
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

API_MODELS_URL = "https://api.together.xyz/v1/models"
DOCS_INDEX_URL = "https://docs.together.ai/docs/serverless-models"
OPENAI_COMPAT_URL = "https://docs.together.ai/docs/openai-api-compatibility"

HEADERS_API = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Accept": "application/json",
}

HEADERS_WEB = {
    "User-Agent": "Mozilla/5.0 (compatible; EnkiKG/1.0)"
}


# =========================================================
# Helpers
# =========================================================
def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def list_accessible_together_models() -> List[Dict[str, Any]]:
    """
    Returns Together models visible to the provided API key.
    Official endpoint already returns fairly rich metadata.
    """
    resp = requests.get(API_MODELS_URL, headers=HEADERS_API, timeout=(10, 60))

    if resp.status_code == 401:
        raise RuntimeError(f"401 Unauthorized from Together AI:\n{resp.text}")

    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "models" in payload and isinstance(payload["models"], list):
            return payload["models"]

    raise RuntimeError(f"Unexpected Together models payload shape: {type(payload).__name__}")


def fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS_WEB, timeout=30)
        if resp.status_code == 200:
            return resp.text
        return None
    except requests.RequestException:
        return None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_model_id_for_match(model_id: str) -> str:
    return clean_text(model_id).lower()


def parse_yes_no_token(token: str) -> Optional[bool]:
    token = token.strip().lower()
    if token == "yes":
        return True
    if token == "no":
        return False
    return None


def parse_serverless_models_page() -> Dict[str, Dict[str, Any]]:
    """
    Parse Together's shared serverless-models page into a lookup keyed by model_id.

    This is heuristic because the page is not a formal JSON API. We only extract
    values when we can do so conservatively.
    """
    html = fetch_html(DOCS_INDEX_URL)
    if not html:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [clean_text(line) for line in text.splitlines() if clean_text(line)]

    lookup: Dict[str, Dict[str, Any]] = {}

    # Pattern focuses on rows that include:
    # ... API Model String Context length $input $output [Quantization] [Function Calling] [Structured Outputs]
    #
    # Example row form observed in docs:
    # Meta Llama 4 Maverick (17Bx128E) meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 1048576 $0.27 $0.85 FP8 Yes Yes
    row_pattern = re.compile(
        r"""
        (?P<prefix>.*?)
        (?P<model_id>[A-Za-z0-9_\-\.]+/[A-Za-z0-9_\-\.]+)
        \s+
        (?P<context_length>\d+)
        \s*
        \$?(?P<input_price>\d+(?:\.\d+)?)
        \s*
        \$?(?P<output_price>\d+(?:\.\d+)?)
        (?:\s+(?P<quantization>[A-Za-z0-9\-\._]+))?
        (?:\s+(?P<function_calling>Yes|No))?
        (?:\s+(?P<structured_outputs>Yes|No))?
        """,
        flags=re.VERBOSE,
    )

    for line in lines:
        m = row_pattern.search(line)
        if not m:
            continue

        model_id = m.group("model_id")
        key = normalize_model_id_for_match(model_id)

        record: Dict[str, Any] = {
            "overview_url": DOCS_INDEX_URL,
            "summary_excerpt": None,
            "context_length_from_docs": int(m.group("context_length")),
            "input_usd_per_1m_tokens_from_docs": float(m.group("input_price")),
            "output_usd_per_1m_tokens_from_docs": float(m.group("output_price")),
            "quantization": m.group("quantization"),
            "function_calling_from_docs": parse_yes_no_token(m.group("function_calling"))
            if m.group("function_calling")
            else None,
            "structured_outputs_from_docs": parse_yes_no_token(m.group("structured_outputs"))
            if m.group("structured_outputs")
            else None,
        }

        prefix = clean_text(m.group("prefix"))
        if prefix:
            record["summary_excerpt"] = prefix

        lookup[key] = record

    return lookup


def derive_modalities_and_family(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort inference from Together's model metadata.
    """
    model_id = str(model.get("id", "") or "")
    model_type = str(model.get("type", "") or "").lower()
    display_name = str(model.get("display_name", "") or "")
    combined = f"{model_id} {model_type} {display_name}".lower()

    input_modalities: List[str] = []
    output_modalities: List[str] = []
    family = None

    if "embedding" in combined:
        family = "embedding"
        input_modalities = ["text"]
        output_modalities = ["embedding"]
    elif "rerank" in combined:
        family = "rerank"
        input_modalities = ["text"]
        output_modalities = ["scores"]
    elif "moderation" in combined or "guard" in combined:
        family = "moderation"
        input_modalities = ["text"]
        output_modalities = ["classification"]
    elif "image" in combined or "vision" in combined or "vl" in combined:
        family = "vision_or_multimodal"
        input_modalities = ["text", "image"]
        output_modalities = ["text"]
    elif "audio" in combined or "speech" in combined or "whisper" in combined or "voxtral" in combined:
        family = "audio"
        input_modalities = ["audio"]
        output_modalities = ["text"]
    elif "video" in combined or "veo" in combined or "kling" in combined:
        family = "video"
        input_modalities = ["text", "image"]
        output_modalities = ["video"]
    else:
        family = "chat_or_completion"
        input_modalities = ["text"]
        output_modalities = ["text"]

    return {
        "model_family_guess": family,
        "input_modalities": sorted(set(input_modalities)),
        "output_modalities": sorted(set(output_modalities)),
    }


def enrich_together_model(
    model: Dict[str, Any],
    docs_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    model_id = model.get("id")
    key = normalize_model_id_for_match(str(model_id or ""))
    docs_info = docs_lookup.get(key, {})

    pricing = model.get("pricing") if isinstance(model.get("pricing"), dict) else {}

    derived = derive_modalities_and_family(model)

    return {
        "model_id": model_id,
        "docs_found": bool(docs_info),
        "docs_url": docs_info.get("overview_url"),
        "summary_excerpt": docs_info.get("summary_excerpt"),
        "display_name": model.get("display_name"),
        "organization": model.get("organization"),
        "type": model.get("type"),
        "link": model.get("link"),
        "license": model.get("license"),
        "context_length": model.get("context_length"),
        "context_length_from_docs": docs_info.get("context_length_from_docs"),
        "pricing": pricing,
        "hourly_price": pricing.get("hourly"),
        "input_usd_per_1m_tokens": pricing.get("input"),
        "output_usd_per_1m_tokens": pricing.get("output"),
        "base_price": pricing.get("base"),
        "finetune_price": pricing.get("finetune"),
        "input_usd_per_1m_tokens_from_docs": docs_info.get("input_usd_per_1m_tokens_from_docs"),
        "output_usd_per_1m_tokens_from_docs": docs_info.get("output_usd_per_1m_tokens_from_docs"),
        "quantization": docs_info.get("quantization"),
        "function_calling": docs_info.get("function_calling_from_docs"),
        "structured_outputs": docs_info.get("structured_outputs_from_docs"),
        **derived,
    }


def build_together_model_records(api_models):
    docs_lookup = parse_serverless_models_page()
    records = []

    for model in api_models:
        enriched = enrich_together_model(model, docs_lookup)

        record = {
            "provider": "together_ai",
            "model_id": model.get("id"),
            "owned_by": model.get("organization"),
            "created_unix": model.get("created"),
            "api_listed": True,
            "catalog_url": DOCS_INDEX_URL,
            "openai_compatibility_url": OPENAI_COMPAT_URL,
            **enriched,
        }
        records.append(record)

    return {
        "provider": "together_ai",
        "catalog_url": DOCS_INDEX_URL,
        "model_list_api_url": API_MODELS_URL,
        "openai_compatibility_url": OPENAI_COMPAT_URL,
        "model_count": len(records),
        "models": records,
    }


def build_raw_payload() -> Any:
    return list_accessible_together_models()


if __name__ == "__main__":
    api_models = list_accessible_together_models()

    write_json(api_models, RAW_OUTPUT_PATH)

    enriched_payload = build_together_model_records(api_models)
    write_json(enriched_payload, OUTPUT_PATH)

    print(f"Wrote raw API JSON to: {RAW_OUTPUT_PATH}")
    print(f"Wrote enriched JSON to: {OUTPUT_PATH}")