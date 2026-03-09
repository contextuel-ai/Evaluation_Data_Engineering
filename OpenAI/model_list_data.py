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
OUTPUT_PATH = SCRIPT_DIR / "openai_models.json"
RAW_OUTPUT_PATH = SCRIPT_DIR / "openai_models_raw.json"

# =========================================================
# Config
# =========================================================
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

API_MODELS_URL = "https://api.openai.com/v1/models"
DOCS_INDEX_URL = "https://developers.openai.com/api/docs/models"
DOCS_MODEL_URL_TEMPLATE = "https://developers.openai.com/api/docs/models/{slug}"

HEADERS_API = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
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


def list_accessible_openai_models() -> List[Dict[str, Any]]:
    """
    Returns the currently accessible models for the provided API key.
    Official API only returns basic metadata.
    """
    resp = requests.get(API_MODELS_URL, headers=HEADERS_API, timeout=30)

    if resp.status_code == 401:
        raise RuntimeError(f"401 Unauthorized from OpenAI:\n{resp.text}")

    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", [])


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


def find_first_date(text: str) -> Optional[str]:
    # Examples seen in docs: "Jun 01, 2024 knowledge cutoff"
    m = re.search(r"\b([A-Z][a-z]{2} \d{1,2}, \d{4})\b", text)
    return m.group(1) if m else None


def find_first_int_before_label(text: str, label: str) -> Optional[int]:
    # e.g. "200,000 context window"
    m = re.search(rf"([\d,]+)\s+{re.escape(label)}", text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def find_price_pair(text: str) -> Dict[str, Optional[float]]:
    """
    Tries to parse the main displayed price pair such as '$20•$80' or '$0.15•$0.6'
    as input/output USD per 1M tokens.
    """
    m = re.search(r"\$(\d+(?:\.\d+)?)\s*[•·]\s*\$(\d+(?:\.\d+)?)", text)
    if not m:
        return {"input_usd_per_1m_tokens": None, "output_usd_per_1m_tokens": None}
    return {
        "input_usd_per_1m_tokens": float(m.group(1)),
        "output_usd_per_1m_tokens": float(m.group(2)),
    }


def extract_modalities(text: str) -> Dict[str, List[str]]:
    input_modalities = []
    output_modalities = []

    # Try to find “Input Text, image” and “Output Text”
    m_in = re.search(r"Input\s+([A-Za-z,\s]+?)\s+Output", text)
    m_out = re.search(r"Output\s+([A-Za-z,\s]+?)(?:\s+[A-Z][a-z]+|$)", text)

    if m_in:
        raw = clean_text(m_in.group(1))
        input_modalities = [x.strip().lower() for x in raw.split(",") if x.strip()]

    if m_out:
        raw = clean_text(m_out.group(1))
        output_modalities = [x.strip().lower() for x in raw.split(",") if x.strip()]

    # Fallbacks for modality section
    for modality in ["Text", "Image", "Audio", "Video"]:
        if re.search(rf"{modality}\s+Input and output", text, re.IGNORECASE):
            input_modalities.append(modality.lower())
            output_modalities.append(modality.lower())
        elif re.search(rf"{modality}\s+Input only", text, re.IGNORECASE):
            input_modalities.append(modality.lower())
        elif re.search(rf"{modality}\s+Output only", text, re.IGNORECASE):
            output_modalities.append(modality.lower())

    return {
        "input_modalities": sorted(set(input_modalities)),
        "output_modalities": sorted(set(output_modalities)),
    }


def extract_feature_support(text: str) -> Dict[str, Optional[bool]]:
    features = [
        "Streaming",
        "Function calling",
        "Structured outputs",
        "Fine-tuning",
        "Distillation",
        "Predicted outputs",
    ]
    result = {}
    for feature in features:
        key = feature.lower().replace("-", "_").replace(" ", "_")
        m = re.search(
            rf"{re.escape(feature)}\s+(Supported|Not supported)",
            text,
            re.IGNORECASE,
        )
        result[key] = None if not m else (m.group(1).lower() == "supported")
    return result


def extract_endpoint_support(text: str) -> Dict[str, bool]:
    endpoint_map = {
        "responses": r"Responses\s+v1/responses",
        "chat_completions": r"Chat Completions\s+v1/chat/completions",
        "realtime": r"Realtime\s+v1/realtime",
        "assistants": r"Assistants\s+v1/assistants",
        "batch": r"Batch\s+v1/batch",
        "fine_tuning": r"Fine-tuning\s+v1/fine-tuning",
        "embeddings": r"Embeddings\s+v1/embeddings",
        "image_generation": r"Image generation\s+v1/images/generations",
        "image_edit": r"Image edit\s+v1/images/edits",
        "speech_generation": r"Speech generation\s+v1/audio/speech",
        "transcription": r"Transcription\s+v1/audio/transcriptions",
        "translation": r"Translation\s+v1/audio/translations",
        "moderation": r"Moderation\s+v1/moderations",
        "legacy_completions": r"Completions \(legacy\)\s+v1/completions",
    }
    return {
        key: bool(re.search(pattern, text, re.IGNORECASE))
        for key, pattern in endpoint_map.items()
    }


def extract_snapshots(text: str, model_id: str) -> List[str]:
    """
    Pull snapshot-like strings from the snapshots section.
    """
    candidates = set()

    for match in re.findall(rf"\b{re.escape(model_id)}(?:-[A-Za-z0-9\.]+)+\b", text):
        candidates.add(match)

    if model_id in text:
        candidates.add(model_id)

    return sorted(candidates)


def extract_status_flags(text: str) -> Dict[str, Optional[bool]]:
    lowered = text.lower()
    return {
        "default": "default" in lowered,
        "preview": "preview" in lowered,
        "deprecated": "deprecated" in lowered,
        "legacy": "legacy" in lowered,
    }


def scrape_model_page(model_id: str) -> Dict[str, Any]:
    """
    Scrape a likely OpenAI model docs page.
    This is heuristic because the docs page is not a formal API.
    """
    slug = model_id
    url = DOCS_MODEL_URL_TEMPLATE.format(slug=slug)
    html = fetch_html(url)

    if not html:
        return {
            "model_id": model_id,
            "docs_found": False,
            "docs_url": url,
        }

    soup = BeautifulSoup(html, "html.parser")
    text = clean_text(soup.get_text(" ", strip=True))

    context_window = find_first_int_before_label(text, "context window")
    max_output_tokens = find_first_int_before_label(text, "max output tokens")
    knowledge_cutoff = find_first_date(text)
    price_info = find_price_pair(text)
    modalities = extract_modalities(text)
    features = extract_feature_support(text)
    endpoints = extract_endpoint_support(text)
    snapshots = extract_snapshots(text, model_id)
    status_flags = extract_status_flags(text)

    reasoning_token_support = bool(
        re.search(r"Reasoning token support", text, re.IGNORECASE)
    )

    summary = None
    m_summary = re.search(
        rf"{re.escape(model_id)}\s+Default\s+(.+?)\s+Compare Try in Playground",
        text,
        re.IGNORECASE,
    )
    if m_summary:
        summary = clean_text(m_summary.group(1))

    reasoning_label = None
    intelligence_label = None
    speed_label = None

    for label_name in ["Reasoning", "Intelligence", "Performance"]:
        m = re.search(
            rf"{label_name}\s+(Highest|Higher|Average|Low)",
            text,
            re.IGNORECASE,
        )
        if m:
            if label_name.lower() == "reasoning":
                reasoning_label = m.group(1)
            else:
                intelligence_label = m.group(1)

    m_speed = re.search(
        r"Speed\s+(Slowest|Slow|Medium|Fast|Very fast)",
        text,
        re.IGNORECASE,
    )
    if m_speed:
        speed_label = m_speed.group(1)

    return {
        "model_id": model_id,
        "docs_found": True,
        "docs_url": url,
        "summary": summary,
        "reasoning_label": reasoning_label,
        "intelligence_or_performance_label": intelligence_label,
        "speed_label": speed_label,
        "context_window_tokens": context_window,
        "max_output_tokens": max_output_tokens,
        "knowledge_cutoff": knowledge_cutoff,
        "reasoning_token_support": reasoning_token_support,
        **price_info,
        **modalities,
        "features": features,
        "endpoints": endpoints,
        "snapshots": snapshots,
        "status_flags": status_flags,
    }


def build_openai_model_records() -> Dict[str, Any]:
    api_models = list_accessible_openai_models()
    records = []

    for model in api_models:
        model_id = model["id"]
        enriched = scrape_model_page(model_id)

        record = {
            "provider": "openai",
            "model_id": model_id,
            "owned_by": model.get("owned_by"),
            "created_unix": model.get("created"),
            "api_listed": True,
            "catalog_url": DOCS_INDEX_URL,
            **enriched,
        }
        records.append(record)
        time.sleep(0.2)

    return {
        "provider": "openai",
        "catalog_url": DOCS_INDEX_URL,
        "model_list_api_url": API_MODELS_URL,
        "model_count": len(records),
        "models": records,
    }


def build_raw_payload() -> Dict[str, Any]:
    api_models = list_accessible_openai_models()
    return {
        "object": "list",
        "data": api_models,
    }


if __name__ == "__main__":
    raw_payload = build_raw_payload()
    write_json(raw_payload, RAW_OUTPUT_PATH)

    enriched_payload = build_openai_model_records()
    write_json(enriched_payload, OUTPUT_PATH)

    print(f"Wrote raw API JSON to: {RAW_OUTPUT_PATH}")
    print(f"Wrote enriched JSON to: {OUTPUT_PATH}")