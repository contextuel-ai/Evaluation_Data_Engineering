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
OUTPUT_PATH = SCRIPT_DIR / "mistral_models.json"
RAW_OUTPUT_PATH = SCRIPT_DIR / "mistral_models_raw.json"

# =========================================================
# Config
# =========================================================
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]

API_MODELS_URL = "https://api.mistral.ai/v1/models"
DOCS_INDEX_URL = "https://docs.mistral.ai/getting-started/models"
DOCS_MODEL_URL_BASE = "https://docs.mistral.ai/models/"

HEADERS_API = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
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


def list_accessible_mistral_models() -> List[Dict[str, Any]]:
    """
    Returns the currently accessible models for the provided API key
    via the Mistral /v1/models endpoint.
    """
    resp = requests.get(API_MODELS_URL, headers=HEADERS_API, timeout=30)

    if resp.status_code == 401:
        raise RuntimeError(f"401 Unauthorized from Mistral:\n{resp.text}")

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


# =========================================================
# Docs-index scraping — discover model page URLs
# =========================================================
def discover_model_page_slugs() -> Dict[str, str]:
    """
    Scrape the Mistral models index page to build a map of
    slug → full docs URL for every model that has a dedicated page.
    E.g. "mistral-large-3-25-12" → "https://docs.mistral.ai/models/mistral-large-3-25-12"
    """
    html = fetch_html(DOCS_INDEX_URL)
    if not html:
        print("[WARN] Could not fetch models index page")
        return {}

    soup = BeautifulSoup(html, "html.parser")
    slug_map: Dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Match links like /models/mistral-large-3-25-12
        m = re.match(r"(?:https://docs\.mistral\.ai)?/models/([\w\-]+)", href)
        if m:
            slug = m.group(1)
            # Skip non-model slugs (e.g. "labs", "compare")
            if slug in ("labs", "compare"):
                continue
            full_url = f"https://docs.mistral.ai/models/{slug}"
            slug_map[slug] = full_url

    return slug_map


# =========================================================
# Individual model-page scraping
# =========================================================
def extract_context_window(text: str) -> Optional[int]:
    """Extract context window size like 'CONTEXT 256k' or 'CONTEXT 128k'."""
    m = re.search(r"CONTEXT\s+([\d,]+)\s*k", text, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", "")) * 1000
    m = re.search(r"CONTEXT\s+([\d,]+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def extract_price_pair(html: str) -> Dict[str, Optional[float]]:
    """
    Parse input/output pricing from the raw HTML.
    Mistral docs render prices as: $<!-- -->0.5 ... /M Tokens
    We search the raw HTML for this pattern.
    """
    # Match "$<!-- -->X.XX" followed (eventually) by "/M Tokens"
    prices = re.findall(
        r'\$(?:<!--\s*-->)?\s*([\d.]+).*?/M\s*Tokens',
        html,
        re.IGNORECASE | re.DOTALL,
    )
    # De-dup consecutive duplicates (page sometimes renders same block twice)
    unique = []
    for p in prices:
        if not unique or unique[-1] != p:
            unique.append(p)
    if len(unique) >= 2:
        return {
            "input_usd_per_1m_tokens": float(unique[0]),
            "output_usd_per_1m_tokens": float(unique[1]),
        }
    if len(unique) == 1:
        return {
            "input_usd_per_1m_tokens": float(unique[0]),
            "output_usd_per_1m_tokens": None,
        }
    return {
        "input_usd_per_1m_tokens": None,
        "output_usd_per_1m_tokens": None,
    }


def extract_api_id(text: str) -> Optional[str]:
    """
    Extract the API model identifier shown on the page.
    e.g. 'mistral-large-2512' or 'codestral-2508'.
    These appear right after the model title as a standalone slug.
    """
    # Pattern: a standalone model-id like "mistral-large-2512" shown on page
    m = re.search(
        r"\b((?:mistral|codestral|ministral|magistral|pixtral|devstral|voxtral|mathstral|mixtral)"
        r"[\w\-]*\-\d{4})\b",
        text,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def extract_release_date(text: str) -> Optional[str]:
    """Extract the release date like 'December 2, 2025'."""
    m = re.search(r"\b([A-Z][a-z]+ \d{1,2}, \d{4})\b", text)
    return m.group(0) if m else None


def extract_version(text: str) -> Optional[str]:
    """Extract version tag like 'v25.12'."""
    m = re.search(r"\bv(\d{2}\.\d{2})\b", text)
    return f"v{m.group(1)}" if m else None


def extract_license_tier(text: str) -> Optional[str]:
    """
    Extract the license/access tier: OPEN, PREMIER, or LABS.
    These appear as labels on each model page.
    """
    for tier in ("OPEN", "PREMIER", "LABS"):
        if re.search(rf"\b{tier}\b", text):
            return tier
    return None


def extract_description(soup: BeautifulSoup, text: str) -> Optional[str]:
    """
    The model description on Mistral docs pages is typically inside an
    <article> tag that is a sibling of the <h1> title.
    """
    h1 = soup.find("h1")
    if not h1:
        return None

    # Primary: look for an <article> sibling after the H1
    for sibling in h1.find_next_siblings():
        if sibling.name == "article":
            # The article contains the description plus the full model card.
            # The description is the initial text before the stats/features.
            article_text = clean_text(sibling.get_text(" ", strip=True))
            # Truncate at known section markers
            for marker in ["SPEED", "PERFORMANCE", "MODALITIES", "CONTEXT",
                           "PRICE", "FEATURES", "WEIGHTS", "FINETUNING"]:
                idx = article_text.find(marker)
                if idx > 0:
                    article_text = article_text[:idx].strip()
                    break
            if article_text and len(article_text) > 10:
                return article_text
        if sibling.name in ("h2", "h3"):
            break

    # Fallback: look for a <p> sibling
    for sibling in h1.find_next_siblings():
        if sibling.name == "p":
            desc = clean_text(sibling.get_text(" ", strip=True))
            if desc and len(desc) > 10:
                return desc
        if sibling.name in ("h2", "h3", "section"):
            break

    return None


def extract_modality_icons(text: str, api_capabilities: Optional[Dict] = None) -> Dict[str, List[str]]:
    """
    Determine modalities primarily from API capabilities, with doc-text fallback.
    API capabilities dict may contain: vision, audio, audio_transcription, ocr.
    """
    input_modalities = ["text"]  # all models accept text
    output_modalities = ["text"]  # all generative models output text

    if api_capabilities:
        if api_capabilities.get("vision"):
            input_modalities.append("image")
        if api_capabilities.get("audio") or api_capabilities.get("audio_transcription"):
            input_modalities.append("audio")
        if api_capabilities.get("ocr") and "image" not in input_modalities:
            input_modalities.append("image")
    else:
        # Fallback: infer from the model description text only (not full page)
        lower = text.lower()

        if "multimodal" in lower or "vision" in lower:
            input_modalities.append("image")
        if "audio input" in lower or lower.startswith("voxtral"):
            input_modalities.append("audio")
        if "image generation" in lower:
            output_modalities.append("image")

    return {
        "input_modalities": sorted(set(input_modalities)),
        "output_modalities": sorted(set(output_modalities)),
    }


def extract_features(soup: BeautifulSoup) -> List[str]:
    """
    Extract supported feature names from the Features section.
    These appear as links: 'Chat Completions', 'Function Calling',
    'Structured Outputs', 'Predicted Outputs', 'FIM', 'OCR', etc.
    """
    features = []
    features_heading = None

    for tag in soup.find_all(["h3", "h2"]):
        if "features" in tag.get_text(strip=True).lower():
            features_heading = tag
            break

    if not features_heading:
        return features

    # Gather all links in the Features section
    container = features_heading.find_parent()
    if not container:
        container = features_heading

    for a in container.find_all("a", href=True):
        link_text = clean_text(a.get_text(" ", strip=True))
        # Filter out navigation / irrelevant links
        if not link_text or len(link_text) < 3:
            continue
        # Extract just the feature name (before any endpoint path)
        # e.g. "Chat Completions /v1/chat/completions" → "Chat Completions"
        feature_name = re.split(r"\s+/v\d", link_text)[0].strip()
        if feature_name and feature_name not in features:
            features.append(feature_name)

    return features


def extract_speed_performance(html: str) -> Dict[str, Optional[int]]:
    """
    Extract speed and performance ratings from bar indicators on the page.

    Each metric is rendered as a row of 4 bars (divs with class
    ``bg-primary-soft``).  Filled bars have ``opacity-100``; unfilled
    bars have ``opacity-30``.  We count filled bars to get a rating
    from 0–4.
    """
    result: Dict[str, Optional[int]] = {
        "speed_rating": None,
        "performance_rating": None,
    }

    for label, key in [("Speed", "speed_rating"),
                       ("Performance", "performance_rating")]:
        # Find the section header, e.g. ">Speed</span>"
        idx = html.find(f">{label}</span>")
        if idx < 0:
            continue
        # Look at the next ~800 chars for the bar divs
        chunk = html[idx:idx + 800]
        bars = re.findall(r"bg-primary-soft[^>]*opacity-(\d+)", chunk)
        if bars:
            filled = sum(1 for b in bars if b == "100")
            result[key] = filled

    return result


def extract_parameter_info(text: str) -> Dict[str, Optional[Any]]:
    """
    Extract parameter counts from description text.
    e.g. '41B active parameters and 675B total parameters'
    or 'features 24B parameters'
    """
    total_params = None
    active_params = None

    # "XB total parameters"
    m = re.search(r"([\d.]+)\s*B\s+total\s+param", text, re.IGNORECASE)
    if m:
        total_params = float(m.group(1))

    # "XB active parameters"
    m = re.search(r"([\d.]+)\s*B\s+active\s+param", text, re.IGNORECASE)
    if m:
        active_params = float(m.group(1))

    # Generic "features XB parameters" or just "XB parameters"
    if total_params is None:
        m = re.search(r"([\d.]+)\s*B\s+param", text, re.IGNORECASE)
        if m:
            total_params = float(m.group(1))

    return {
        "total_parameters_b": total_params,
        "active_parameters_b": active_params,
    }


def scrape_model_page(slug: str, url: str,
                      api_capabilities: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Scrape an individual Mistral model documentation page for metadata.
    api_capabilities, if provided, comes from the API /v1/models response.
    """
    html = fetch_html(url)

    if not html:
        return {
            "slug": slug,
            "docs_found": False,
            "docs_url": url,
        }

    soup = BeautifulSoup(html, "html.parser")
    text = clean_text(soup.get_text(" ", strip=True))

    # Extract the page title (model name)
    h1 = soup.find("h1")
    model_name = clean_text(h1.get_text(strip=True)) if h1 else slug

    api_id = extract_api_id(text)
    description = extract_description(soup, text)
    context_window = extract_context_window(text)
    price_info = extract_price_pair(html)
    release_date = extract_release_date(text)
    version = extract_version(text)
    license_tier = extract_license_tier(text)
    modalities = extract_modality_icons(
        description or model_name, api_capabilities
    )
    features = extract_features(soup)
    params = extract_parameter_info(text)
    speed_perf = extract_speed_performance(html)

    return {
        "slug": slug,
        "model_name": model_name,
        "api_model_id": api_id,
        "docs_found": True,
        "docs_url": url,
        "description": description,
        "license_tier": license_tier,
        "version": version,
        "release_date": release_date,
        "context_window_tokens": context_window,
        **price_info,
        **modalities,
        **params,
        **speed_perf,
        "features": features,
    }


# =========================================================
# Match API models → doc pages
# =========================================================
def match_api_model_to_slug(model_id: str, slug_map: Dict[str, str]) -> Optional[str]:
    """
    Try to find the best matching doc page slug for an API model id.
    API ids look like 'mistral-large-2512', slugs like 'mistral-large-3-25-12'.
    """
    # Direct match
    if model_id in slug_map:
        return model_id

    # Normalise: strip '-latest' suffix
    base = re.sub(r"-latest$", "", model_id)

    # Try converting API date format (YYMM) → slug format (YY-MM)
    # e.g. "mistral-large-2512" → try "mistral-large-*-25-12"
    m = re.match(r"^(.+?)[-_](\d{2})(\d{2})$", base)
    if m:
        prefix, yy, mm = m.group(1), m.group(2), m.group(3)
        # Try various slug patterns
        for slug in slug_map:
            if slug.endswith(f"-{yy}-{mm}") and slug.startswith(prefix.split("-")[0]):
                # Check if the slug contains the key parts of the prefix
                slug_core = re.sub(r"-\d{2}-\d{2}$", "", slug)
                prefix_core = prefix.replace("-", "")
                slug_core_flat = slug_core.replace("-", "")
                if prefix_core in slug_core_flat or slug_core_flat in prefix_core:
                    return slug

    # Fuzzy: look for slugs that share a long common prefix
    best_slug = None
    best_overlap = 0
    for slug in slug_map:
        overlap = len(os.path.commonprefix([model_id, slug]))
        if overlap > best_overlap and overlap >= 8:
            best_overlap = overlap
            best_slug = slug

    return best_slug


# =========================================================
# Build enriched records
# =========================================================
def build_mistral_model_records() -> Dict[str, Any]:
    """
    1. Fetch model list from Mistral API (raw metadata).
    2. Discover all model doc page URLs from the docs index.
    3. For each API model, try to match it to a doc page and scrape it.
    4. Also scrape any doc pages that weren't matched (non-API / legacy models).
    """
    api_models = list_accessible_mistral_models()
    slug_map = discover_model_page_slugs()

    print(f"API models: {len(api_models)}")
    print(f"Doc page slugs discovered: {len(slug_map)}")

    records = []
    scraped_slugs = set()

    # Step 1: For each API model, find and scrape its doc page
    for model in api_models:
        model_id = model.get("id", "")
        matched_slug = match_api_model_to_slug(model_id, slug_map)

        doc_info: Dict[str, Any] = {}
        if matched_slug:
            doc_url = slug_map[matched_slug]
            doc_info = scrape_model_page(
                matched_slug, doc_url,
                api_capabilities=model.get("capabilities"),
            )
            scraped_slugs.add(matched_slug)
            time.sleep(0.3)

        record = {
            "provider": "mistral",
            "model_id": model_id,
            "owned_by": model.get("owned_by"),
            "created_unix": model.get("created"),
            "object_type": model.get("object"),
            "api_listed": True,
            "max_context_length": model.get("max_context_length"),
            "default_model_temperature": model.get("default_model_temperature"),
            "type": model.get("type"),
            "capabilities": model.get("capabilities", {}),
            "deprecation": model.get("deprecation"),
            **doc_info,
        }
        records.append(record)

    # Step 2: Scrape any remaining doc pages not matched to API models
    for slug, url in sorted(slug_map.items()):
        if slug in scraped_slugs:
            continue

        doc_info = scrape_model_page(slug, url)
        record = {
            "provider": "mistral",
            "model_id": doc_info.get("api_model_id") or slug,
            "owned_by": "mistralai",
            "created_unix": None,
            "object_type": None,
            "api_listed": False,
            "max_context_length": None,
            "default_model_temperature": None,
            "type": None,
            "capabilities": {},
            "deprecation": None,
            **doc_info,
        }
        records.append(record)
        scraped_slugs.add(slug)
        time.sleep(0.3)

    # De-duplicate by model_id (prefer api_listed entries)
    seen_ids: Dict[str, int] = {}
    deduped = []
    for i, rec in enumerate(records):
        mid = rec["model_id"]
        if mid in seen_ids:
            existing_idx = seen_ids[mid]
            # Keep the one that is API-listed
            if rec["api_listed"] and not deduped[existing_idx]["api_listed"]:
                deduped[existing_idx] = rec
            continue
        seen_ids[mid] = len(deduped)
        deduped.append(rec)

    return {
        "provider": "mistral",
        "catalog_url": DOCS_INDEX_URL,
        "model_list_api_url": API_MODELS_URL,
        "model_count": len(deduped),
        "models": deduped,
    }


def build_raw_payload() -> Dict[str, Any]:
    api_models = list_accessible_mistral_models()
    return {
        "object": "list",
        "data": api_models,
    }


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    raw_payload = build_raw_payload()
    write_json(raw_payload, RAW_OUTPUT_PATH)

    enriched_payload = build_mistral_model_records()
    write_json(enriched_payload, OUTPUT_PATH)

    print(f"\nWrote raw API JSON to:  {RAW_OUTPUT_PATH}")
    print(f"Wrote enriched JSON to: {OUTPUT_PATH}")
    print(f"Total models: {enriched_payload['model_count']}")
