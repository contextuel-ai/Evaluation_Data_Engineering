import os
import requests
import pandas as pd

AA_API_KEY = os.getenv("AA_API_KEY")
if not AA_API_KEY:
    raise RuntimeError("AA_API_KEY env var not set. Run: export AA_API_KEY='...'")


URL = "https://artificialanalysis.ai/api/v2/data/llms/models"
HEADERS = {"x-api-key": AA_API_KEY}

resp = requests.get(URL, headers=HEADERS, timeout=60)
resp.raise_for_status()

payload = resp.json()

# AA typically wraps results like {"data": [...]}
models = payload.get("data", payload if isinstance(payload, list) else [])

def zeroish_to_none(value):
    """
    Treat 0 / 0.0 / '0' / '0.0' as missing placeholders.
    Leaves other values unchanged.
    """
    if value is None:
        return None

    # catch strings like "0", "0.0", "0.00", " 0.0 "
    if isinstance(value, str):
        s = value.strip()
        if s in {"0", "0.0", "0.00", "0.000"}:
            return None
        return value

    # catch numeric zeros
    if isinstance(value, (int, float)) and value == 0:
        return None

    return value

rows = []
for m in models:
    creator = m.get("model_creator") or {}
    pricing = m.get("pricing") or {}
    evaluations = m.get("evaluations") or {}

    row = {
        # IDs / names
        "model_id": m.get("id"),
        "model_name": m.get("name"),
        "model_slug": m.get("slug"),

        # creator
        "creator": creator.get("name"),
        "creator_id": creator.get("id"),
        "creator_slug": creator.get("slug"),

        "output_tokens_per_second": zeroish_to_none(m.get("median_output_tokens_per_second")),
        "time_to_first_token": zeroish_to_none(m.get("median_time_to_first_token_seconds")),
        "median_time_to_first_answer_token": zeroish_to_none(m.get("median_time_to_first_answer_token")),

        "price_1m_input_tokens": zeroish_to_none(pricing.get("price_1m_input_tokens")),
        "price_1m_output_tokens": zeroish_to_none(pricing.get("price_1m_output_tokens")),
        "price_1m_blended_3_to_1": zeroish_to_none(pricing.get("price_1m_blended_3_to_1"))

    }

    # Keep evaluation scores unchanged
    for k, v in evaluations.items():
        row[k] = v

    rows.append(row)

df = pd.DataFrame(rows)

# Optional: stable column ordering (keeps your core columns first)
core_cols = [
    "model_id", "model_name", "model_slug",
    "creator", "creator_id", "creator_slug",
    "output_tokens_per_second",
    "time_to_first_token",
    "median_time_to_first_answer_token",
    "price_1m_input_tokens",
    "price_1m_output_tokens",
    "price_1m_blended_3_to_1",
]
other_cols = [c for c in df.columns if c not in core_cols]
df = df[core_cols + sorted(other_cols)]

out_path = "aa_models_with_pricing_speed_and_benchmarks.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {out_path} with {len(df)} rows and {len(df.columns)} columns.")