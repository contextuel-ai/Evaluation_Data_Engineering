"""
Mistral JSON → Turtle (TTL) conversion for the Contextuel.ai knowledge graph.

Reads the enriched JSON produced by model_list_data.py (mistral_models.json)
and emits Turtle triples covering:
  - LLM instances with metadata (params, context window, modalities, description)
  - LLMProvisioning triples (Mistral as provider for each model)
  - EvaluatedMetric triples for input/output pricing (metricAssesses → Cost)

All models are manufactured and provisioned by Mistral.
Org / license / capability definitions live in the base ontology.
"""

from pathlib import Path
from collections import OrderedDict, defaultdict
import json
import re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parent
SOURCE_JSON = ROOT / "mistral_models.json"
OUTPUT_TTL = ROOT / "mistral_models.ttl"
BACKUP = OUTPUT_TTL.with_suffix(".ttl.bak")

C = Namespace("https://www.contextuel.ai/spec/Contextuel/")
ENKI = Namespace("https://www.contextuel.ai/spec/Enki/")
DCT = Namespace("http://purl.org/dc/terms/")

HEADER_BLOCK = """@base <https://www.contextuel.ai/spec/Contextuel/> .
@prefix enki: <https://www.contextuel.ai/spec/Enki/> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix contextuel: <https://www.contextuel.ai/spec/Contextuel/> .

"""

# _________ CLAIMED METRICS __________

PRICE_METRICS = {
    "Price_1M_Input_Tokens":  {"field": "input_usd_per_1m_tokens",
                                "label": "Price per 1M Input Tokens ($)",
                                "criterion": "Cost"},
    "Price_1M_Output_Tokens": {"field": "output_usd_per_1m_tokens",
                                "label": "Price per 1M Output Tokens ($)",
                                "criterion": "Cost"},
}

RATING_METRICS = {
    "Speed_Rating":       {"field": "speed_rating",
                           "label": "Speed Rating (1-5)",
                           "criterion": "Latency"},
    "Performance_Rating": {"field": "performance_rating",
                           "label": "Performance Rating (1-5)",
                           "criterion": "Quality"},
}

# _________ MODALITY MAPPING __________
# Map JSON modality labels (lowercase) to ontology URI locals from modalities.ttl.
MODALITY_MAP = {
    "text":  "TextModality",
    "image": "ImageModality",
    "audio": "AudioModality",
    "video": "VideoModality",
}

# _________ HELPERS __________

def _sanitize(s: str) -> str:
    """Shared sanitisation: replace separators with _, strip non-alphanum."""
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def local_from_label(s: str) -> str:
    """For general labels. No org stripping."""
    s = (s or "").strip()
    s = _sanitize(s)
    if not s:
        return "Unnamed"
    parts = s.split("_")
    cap_parts = []
    for p in parts:
        if not p:
            continue
        if any(c.isupper() for c in p):
            cap_parts.append(p)
        else:
            cap_parts.append(p[0].upper() + p[1:])
    return "_".join(cap_parts) or "Unnamed"


def local_from_model(s: str) -> str:
    """For model names. Strips org prefix if present before sanitising."""
    s = (s or "").strip()
    if "/" in s:
        s = s.split("/", 1)[1]
    s = _sanitize(s)
    if not s:
        return "Unnamed"
    parts = s.split("_")
    cap_parts = []
    for p in parts:
        if not p:
            continue
        if any(c.isupper() for c in p):
            cap_parts.append(p)
        else:
            cap_parts.append(p[0].upper() + p[1:])
    return "_".join(cap_parts) or "Unnamed"


def parse_release_date(s: str) -> str | None:
    """Parse 'December 2, 2025' or 'May 7, 2025' → '2025-12' (xsd:gYearMonth).
    Returns None if unparseable."""
    if not s:
        return None
    MONTHS = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    m = re.match(r'(\w+)\s+\d{1,2},?\s+(\d{4})', s.strip())
    if not m:
        return None
    month_str = m.group(1).lower()
    year = m.group(2)
    month_num = MONTHS.get(month_str)
    if not month_num:
        return None
    return f"{year}-{month_num}"


# _________ ORGANISATION __________

# Mistral is the sole manufacturer and provider.
ORG_LOCAL = "Mistral"

# _________ CURL TEMPLATES __________

CURL_MODELS = (
    'curl https://api.mistral.ai/v1/models '
    '-H "Authorization: Bearer $MISTRAL_API_KEY"'
)

# Per-endpoint curl templates.  {model_id} is substituted at emission time.
# Use double-brace {{ }} to produce literal braces after .format().
CURL_CHAT = (
    'curl https://api.mistral.ai/v1/chat/completions '
    '-H "Authorization: Bearer $MISTRAL_API_KEY" '
    '-H "Content-Type: application/json" '
    """-d '{{"model":"{model_id}","messages":[{{"role":"user","content":"Hello"}}]}}'"""
)

CURL_FIM = (
    'curl https://api.mistral.ai/v1/fim/completions '
    '-H "Authorization: Bearer $MISTRAL_API_KEY" '
    '-H "Content-Type: application/json" '
    """-d '{{"model":"{model_id}","prompt":"def fibonacci(n):","suffix":"return result"}}'"""
)

CURL_EMBED = (
    'curl https://api.mistral.ai/v1/embeddings '
    '-H "Authorization: Bearer $MISTRAL_API_KEY" '
    '-H "Content-Type: application/json" '
    """-d '{{"model":"{model_id}","input":["Hello world"]}}'"""
)


def curl_for_model(model_id: str, caps: dict, api_listed: bool) -> str | None:
    """Return the appropriate curl template for a model, or None if not API-accessible."""
    if not api_listed:
        return None
    mid_lower = model_id.lower()
    # Embedding models use /v1/embeddings
    if "embed" in mid_lower:
        return CURL_EMBED.format(model_id=model_id)
    # FIM-capable models (Codestral) — prefer FIM endpoint if not also chat
    if caps.get("completion_fim") and not caps.get("completion_chat"):
        return CURL_FIM.format(model_id=model_id)
    # Chat completion (the vast majority)
    if caps.get("completion_chat"):
        return CURL_CHAT.format(model_id=model_id)
    # Fallback for API-listed models without known caps (use chat)
    return CURL_CHAT.format(model_id=model_id)

# _________ READ JSON __________

def read_models(path: Path) -> list[dict]:
    """Read the enriched Mistral JSON and normalise into row dicts."""
    if not path.exists():
        print(f"[ERROR] JSON not found: {path}")
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    print(f"Loaded {len(models)} models from {path.name}")

    rows = []
    seen_ids = set()
    for m in models:
        model_id = (m.get("model_id") or "").strip()
        if not model_id:
            continue
        # Deduplicate by model_id
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        # Use model_name (human label) if available, fall back to model_id
        display_name = m.get("model_name") or m.get("display_name") or ""
        label = display_name if display_name else model_id

        # Description
        description = (m.get("description") or "").strip()

        # Context window — prefer docs value, fall back to API max_context_length
        context_window = m.get("context_window_tokens") or m.get("max_context_length")
        if context_window:
            try:
                context_window = int(context_window)
                if context_window <= 0:
                    context_window = None
            except (ValueError, TypeError):
                context_window = None

        # Parameter counts
        total_params_b = m.get("total_parameters_b")
        active_params_b = m.get("active_parameters_b")

        # Modalities
        input_modalities = m.get("input_modalities") or []
        output_modalities = m.get("output_modalities") or []

        # Pricing
        price = {}
        for pm_key, pm_info in PRICE_METRICS.items():
            v = m.get(pm_info["field"])
            if v is not None:
                try:
                    v = float(v)
                    if v >= 0:
                        price[pm_key] = v
                except (ValueError, TypeError):
                    pass

        # Release date → gYearMonth
        release_date = parse_release_date(m.get("release_date") or "")

        # Docs URL for identifier
        docs_url = m.get("docs_url") or ""

        # Speed & performance ratings (1-5 scale, self-reported by Mistral)
        ratings = {}
        for rm_key, rm_info in RATING_METRICS.items():
            v = m.get(rm_info["field"])
            if v is not None:
                try:
                    v = int(v)
                    if 1 <= v <= 5:
                        ratings[rm_key] = v
                except (ValueError, TypeError):
                    pass

        # API availability and capabilities (for curl generation)
        api_listed = bool(m.get("api_listed"))
        capabilities = m.get("capabilities") or {}

        rows.append({
            "model_id": model_id,
            "label": label,
            "description": description,
            "context_window": context_window,
            "total_params_b": total_params_b,
            "active_params_b": active_params_b,
            "input_modalities": input_modalities,
            "output_modalities": output_modalities,
            "price": price,
            "ratings": ratings,
            "release_date": release_date,
            "docs_url": docs_url,
            "api_listed": api_listed,
            "capabilities": capabilities,
        })

    print(f"After dedup: {len(rows)} unique models")
    return rows


# _________ BUILD GRAPH __________

def build_graph(rows: list[dict]) -> tuple[Graph, dict[URIRef, list[URIRef]]]:
    g = Graph()
    g.bind("enki", ENKI)
    g.bind("contextuel", C)
    g.bind("dct", DCT)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    org_uri = URIRef(str(C) + ORG_LOCAL)
    model_uris: list[URIRef] = []
    provisioning_uris: list[URIRef] = []
    metrics_by_llm: dict[URIRef, list[URIRef]] = defaultdict(list)

    # Provider-level: curl to list all models
    g.add((org_uri, ENKI.curlModel, Literal(CURL_MODELS)))

    for r in rows:
        local = local_from_model(r["model_id"])
        model_uri = URIRef(str(C) + local)

        g.add((model_uri, RDF.type, ENKI.LLM))
        g.add((model_uri, RDFS.label, Literal(r["label"])))

        # Identifier (Mistral docs URL or API model ID)
        if r["docs_url"]:
            g.add((model_uri, DCT.identifier, Literal(r["docs_url"])))
        else:
            g.add((model_uri, DCT.identifier, Literal(r["model_id"])))

        # Description
        if r["description"]:
            g.add((model_uri, DCT.description, Literal(r["description"])))

        # Context window
        if r["context_window"]:
            g.add((model_uri, ENKI.hasMaxContextWindowTokens,
                   Literal(str(r["context_window"]), datatype=XSD.nonNegativeInteger)))

        # Parameter count (total, in raw count as nonNegativeInteger)
        if r["total_params_b"] is not None:
            try:
                params_b = float(r["total_params_b"])
                if params_b > 0:
                    raw_count = int(params_b * 1_000_000_000)
                    g.add((model_uri, ENKI.hasTotalParameterCount,
                           Literal(str(raw_count), datatype=XSD.nonNegativeInteger)))
            except (ValueError, TypeError):
                pass

        # Input modalities
        for mod in r["input_modalities"]:
            mod_local = MODALITY_MAP.get(mod.lower())
            if mod_local:
                g.add((model_uri, ENKI.hasInputModality, URIRef(str(C) + mod_local)))

        # Output modalities
        for mod in r["output_modalities"]:
            mod_local = MODALITY_MAP.get(mod.lower())
            if mod_local:
                g.add((model_uri, ENKI.hasOutputModality, URIRef(str(C) + mod_local)))

        # Release date as knowledge cutoff (xsd:gYearMonth)
        if r["release_date"]:
            g.add((model_uri, ENKI.hasKnowledgeCutoff,
                   Literal(r["release_date"], datatype=XSD.gYearMonth)))

        # Manufacturer: Mistral manufactures this model
        g.add((org_uri, ENKI.manufactures, model_uri))

        # Provisioning: Mistral provisions this model via API
        prov_local = f"{ORG_LOCAL}_Provisioning_{local}"
        prov_uri = URIRef(str(C) + prov_local)
        g.add((prov_uri, RDF.type, ENKI.LLMProvisioning))
        g.add((prov_uri, RDFS.label, Literal(f"Mistral provisioning of {r['label']}")))
        g.add((prov_uri, ENKI.provider, org_uri))
        g.add((prov_uri, ENKI.provides, model_uri))

        # Curl example for API-listed models
        curl_cmd = curl_for_model(r["model_id"], r["capabilities"], r["api_listed"])
        if curl_cmd:
            g.add((prov_uri, ENKI.curlExample, Literal(curl_cmd)))

        provisioning_uris.append(prov_uri)

        # Price metrics as ClaimedMetric → Cost
        for pm_key, pm_val in r["price"].items():
            pm_info = PRICE_METRICS[pm_key]
            metric_uri = URIRef(str(C) + f"{local}_{pm_key}")
            g.add((metric_uri, RDF.type, ENKI.ClaimedMetric))
            g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + pm_info["criterion"])))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, org_uri))
            g.add((metric_uri, ENKI.metricUnit, Literal(pm_info["label"])))
            g.add((metric_uri, ENKI.metricValue, Literal(pm_val)))
            metrics_by_llm[model_uri].append(metric_uri)

        # Speed / performance ratings as ClaimedMetric → Latency / Quality
        for rm_key, rm_val in r["ratings"].items():
            rm_info = RATING_METRICS[rm_key]
            metric_uri = URIRef(str(C) + f"{local}_{rm_key}")
            g.add((metric_uri, RDF.type, ENKI.ClaimedMetric))
            g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + rm_info["criterion"])))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, org_uri))
            g.add((metric_uri, ENKI.metricUnit, Literal(rm_info["label"])))
            g.add((metric_uri, ENKI.metricValue, Literal(rm_val)))
            metrics_by_llm[model_uri].append(metric_uri)

        model_uris.append(model_uri)

    # Sort metrics for deterministic output
    for model_uri in metrics_by_llm:
        metrics_by_llm[model_uri] = sorted(metrics_by_llm[model_uri], key=lambda s: str(s).lower())

    # Stats
    with_params = sum(1 for r in rows if r["total_params_b"] is not None)
    with_ctx = sum(1 for r in rows if r["context_window"] is not None)
    with_price = sum(1 for r in rows if r["price"])
    with_ratings = sum(1 for r in rows if r["ratings"])
    with_desc = sum(1 for r in rows if r["description"])
    with_modalities = sum(1 for r in rows if r["input_modalities"])
    with_date = sum(1 for r in rows if r["release_date"])
    with_curl = sum(1 for r in rows if curl_for_model(r["model_id"], r["capabilities"], r["api_listed"]))
    total_metrics = sum(len(v) for v in metrics_by_llm.values())
    print(f"Models: {len(rows)}")
    print(f"  with params:     {with_params}")
    print(f"  with context:    {with_ctx}")
    print(f"  with pricing:    {with_price}")
    print(f"  with ratings:    {with_ratings}")
    print(f"  with description:{with_desc}")
    print(f"  with modalities: {with_modalities}")
    print(f"  with release:    {with_date}")
    print(f"  with curl:       {with_curl}")
    print(f"Provisionings: {len(provisioning_uris)}")
    print(f"Claimed metrics: {total_metrics} (price: {sum(len(r['price']) for r in rows)}, ratings: {sum(len(r['ratings']) for r in rows)})")

    return g, metrics_by_llm


# _________ SERIALISATION __________

def reorder_and_serialize(g: Graph, out_path: Path, metrics_by_llm: dict[URIRef, list[URIRef]]):
    PREFIX_MAP = OrderedDict([
        (str(C),     "contextuel:"),
        (str(ENKI),  "enki:"),
        (str(DCT),   "dct:"),
        (str(RDF),   "rdf:"),
        (str(RDFS),  "rdfs:"),
        (str(XSD),   "xsd:"),
    ])

    def curie(uri):
        u = str(uri)
        for ns, prefix in PREFIX_MAP.items():
            if u.startswith(ns):
                return prefix + u[len(ns):]
        return f"<{u}>"

    def format_obj(o):
        if isinstance(o, URIRef):
            return curie(o)
        if isinstance(o, Literal):
            if o.datatype == XSD.integer:
                return str(int(o))
            if o.datatype == XSD.nonNegativeInteger:
                return f'"{int(o)}"^^xsd:nonNegativeInteger'
            if o.datatype == XSD.gYearMonth:
                return f'"{o}"^^xsd:gYearMonth'
            if o.datatype in (XSD.float, XSD.double):
                return str(float(o))
            if o.datatype == XSD.decimal:
                return str(o)
            # Plain numeric literals (no explicit datatype) — emit bare value
            val_str = str(o)
            if o.datatype is None:
                try:
                    fv = float(val_str)
                    if fv == int(fv) and "." not in val_str:
                        return val_str
                    return val_str
                except (ValueError, TypeError):
                    pass
            escaped = str(o).replace("\\", "\\\\").replace('"', '\\"')
            if o.language:
                return f'"{escaped}"@{o.language}'
            if o.datatype:
                return f'"{escaped}"^^{curie(o.datatype)}'
            return f'"{escaped}"'
        return f'"{o}"'

    def serialize_subjects(subject_list):
        block_lines = []
        for s in subject_list:
            preds = sorted(set(g.predicates(s)), key=lambda p: str(p))
            block = []
            types = sorted(set(g.objects(s, RDF.type)), key=lambda o: str(o))
            has_type = bool(types)
            if types:
                if len(types) == 1:
                    block.append(f"{curie(s)} a {curie(types[0])}")
                else:
                    type_strs = ",\n        ".join(curie(t) for t in types)
                    block.append(f"{curie(s)} a {type_strs}")
            for p in preds:
                if p == RDF.type:
                    continue
                objs = sorted(set(g.objects(s, p)), key=lambda o: str(o))
                for o in objs:
                    block.append(f"    {curie(p)} {format_obj(o)}")
            if not has_type and block:
                result = f"{curie(s)}\n" + " ;\n".join(block) + " ."
            else:
                result = " ;\n".join(block) + " ."
            block_lines.append(result)
            block_lines.append("")
        return block_lines

    seen_subjects = set()

    # Gather subject categories
    org_uri = URIRef(str(C) + ORG_LOCAL)
    provisionings = sorted(set(g.subjects(RDF.type, ENKI.LLMProvisioning)), key=lambda s: str(s).lower())
    llms = sorted(set(g.subjects(RDF.type, ENKI.LLM)), key=lambda s: str(s).lower())

    ordered_subjects = []

    def add_if_new(s):
        if s not in seen_subjects:
            ordered_subjects.append(s)
            seen_subjects.add(s)

    # 1. Org (manufacturer triples)
    add_if_new(org_uri)

    # 2. Provisionings
    for p in provisionings:
        add_if_new(p)

    # 3. LLMs, each followed by their metrics
    for llm in llms:
        add_if_new(llm)
        for m in metrics_by_llm.get(llm, []):
            add_if_new(m)

    # 4. Anything remaining
    for s in sorted(set(g.subjects()), key=lambda s: str(s).lower()):
        add_if_new(s)

    lines = serialize_subjects(ordered_subjects)

    if out_path.exists():
        BACKUP.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(HEADER_BLOCK)
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"Wrote Mistral TTL to {out_path} (backup at {BACKUP})")


# _________ MAIN __________

def main():
    rows = read_models(SOURCE_JSON)
    if not rows:
        print("No models read. Exiting.")
        return

    print(f"\nBuilding graph from {len(rows)} models...")
    g, metrics_by_llm = build_graph(rows)
    reorder_and_serialize(g, OUTPUT_TTL, metrics_by_llm)


if __name__ == "__main__":
    main()
