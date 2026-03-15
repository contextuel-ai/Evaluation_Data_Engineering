from pathlib import Path
from collections import OrderedDict, defaultdict
import pandas as pd
import re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "hf_leaderboard_long_20260221.csv"
OUTPUT_TTL = ROOT / "hf_leaderboard.ttl"
OUTPUT_UNMATCHED_TTL = ROOT / "hf_leaderboard_unmatched.ttl"
BACKUP = OUTPUT_TTL.with_suffix(".ttl.bak")
BACKUP_UNMATCHED = OUTPUT_UNMATCHED_TTL.with_suffix(".ttl.bak")

ONTOLOGY_ORGS_TTL = ROOT.parent / "enki" / "backend" / "ontology" / "organizations.ttl"

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

# The 6 benchmarks + CO2 on the HF Open LLM Leaderboard v2
BENCHMARKS = {
    "IFEval":      {"col": "IFEval",       "label": "IFEval"},
    "BBH":         {"col": "BBH",          "label": "BBH"},
    "MATH_Lvl_5":  {"col": "MATH Lvl 5",   "label": "MATH Lvl 5"},
    "GPQA":        {"col": "GPQA",         "label": "GPQA"},
    "MUSR":        {"col": "MUSR",         "label": "MUSR"},
    "MMLU_PRO":    {"col": "MMLU-PRO",     "label": "MMLU-PRO"},
    "CO2":         {"col": "CO₂ cost (kg)", "label": "CO₂ cost (kg)"},
}

# Capabilities tested by each benchmark
BENCHMARK_CAPABILITIES = {
    "IFEval": [
        ("InstructionFollowing", "Instruction Following",
         "The ability of an LLM to understand and correctly execute user-provided instructions."),
    ],
    "BBH": [
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
        ("FormalReasoning", "Formal Reasoning",
         "The ability of an LLM to apply structured logical rules and simulate procedural execution, including syllogistic reasoning, Boolean logic evaluation, truth-table analysis, state tracking, and step-by-step algorithmic operations."),
        ("CausalReasoning", "Causal Reasoning",
         "The ability of an LLM to identify, evaluate, and reason about cause-and-effect relationships between events, actions, and outcomes, including counterfactual reasoning and distinguishing correlation from causation."),
    ],
    "MATH_Lvl_5": [
        ("MathematicalReasoning", "Mathematical Reasoning",
         "The ability of an LLM to solve mathematical problems requiring formal symbolic manipulation, numerical computation, and multi-step logical derivation across domains such as algebra, geometry, number theory, combinatorics, and calculus."),
    ],
    "GPQA": [
        ("ExpertLevelProblemSolving", "Expert-Level Problem Solving",
         "The ability of an LLM to solve problems that require graduate-level or professional-level expertise in a specialized domain, integrating deep domain knowledge with sophisticated reasoning that resists simple lookup or surface-level understanding."),
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "MUSR": [
        ("CommonsenseReasoning", "Commonsense Reasoning",
         "The ability of an LLM to draw plausible inferences based on implicit world knowledge, everyday physical and social understanding, and narrative comprehension, even when the reasoning steps are not formally specified."),
        ("CausalReasoning", "Causal Reasoning",
         "The ability of an LLM to identify, evaluate, and reason about cause-and-effect relationships between events, actions, and outcomes, including counterfactual reasoning and distinguishing correlation from causation."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "MMLU_PRO": [
        ("GeneralKnowledgeReasoning", "General Knowledge Reasoning",
         "The ability of an LLM to apply broad academic knowledge across multiple disciplines to correctly answer domain-specific questions."),
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
        ("FactualRecall", "Factual Recall",
         "The ability of an LLM to accurately retrieve and reproduce learned factual information from its training data."),
    ],
}

# _________ HELPERS __________

def _sanitize(s: str) -> str:
    """Shared sanitisation: replace separators with _, strip non-alphanum."""
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def local_from_label(s: str) -> str:
    """For general labels (evaluations, capabilities, licenses, orgs). No org stripping."""
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
    """For model fullnames. Strips org prefix before sanitising."""
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

def extract_params_billion(name: str) -> float | None:
    """
    Extract the total parameter count (in billions) from a model name.

    Recognises patterns like '70B', '1.5B', '350M', '270M'.
    Returns None if no parameter count is found.

    Heuristics:
      - Match <number>(B|b) or <number>(M|m) at word boundaries.
      - Skip MoE active-parameter tokens like 'A22B', 'x7b'.
      - If multiple matches, take the largest (total params > active params).
    """
    matches = re.findall(r'(?<![AaXx])(\d+(?:\.\d+)?)\s*([BbMm])\b', name)
    if not matches:
        return None
    candidates = []
    for num_str, suffix in matches:
        try:
            val = float(num_str)
        except ValueError:
            continue
        if suffix.upper() == 'B':
            candidates.append(val)
        elif suffix.upper() == 'M':
            candidates.append(val / 1000.0)
    return max(candidates) if candidates else None

def normalize_license_local(lic: str) -> str:
    if not lic:
        return ""
    l = lic.strip()
    if l in LICENSE_RENAME:
        return local_from_label(LICENSE_RENAME[l])
    if re.match(r'(?i)^apache[ _\-.]?2[ _\-.]?0$', l):
        return local_from_label("Apache 2.0")
    return local_from_label(l)

# --- Org / license normalisation mappings ---
# Org URI renames (match against raw org labels from CSV)
ORG_RENAME = {
    "google":                "GoogleAI",
    "meta-llama":            "MetaAI",
    "microsoft":             "MicrosoftAI",
    "nvidia":                "NvidiaNGC",
    "Salesforce":            "SalesforceAI",
    "allenai":               "Allen_AI",
    "01-ai":                 "01_AI",
    "deepseek-ai":           "DeepSeek_AI",
    "stabilityai":           "Stability_AI",
    "cognitivecomputations": "Cognitive_Computations",
    "h2oai":                 "h2oai",
    "openchat":              "OpenChat",
    "lmsys":                 "LMSYS",
    "PrimeIntellect":        "Prime_Intellect",
    "ai21labs":              "AI21_Labs",
    "gradientai":            "GradientAI",
    "mosaicml":              "MosaicML",
    "HuggingFaceH4":         "HuggingFace",
    "argilla-warehouse":     "Argilla",
}

# Org merge (multiple raw labels → single canonical local)
ORG_MERGE: dict[str, str] = {}

# License renames (raw CSV label → canonical label for local_from_label)
LICENSE_RENAME: dict[str, str] = {
    "Non-commercial": "CC_BY_NC_4_0",
    "Qianwen LICENSE": "Apache_2_0",
    "Yi License": "Apache_2_0",
}


def org_uri_local(org_label: str) -> str:
    """Return the canonical URI local name for an org."""
    if org_label in ORG_MERGE:
        return ORG_MERGE[org_label]
    if org_label in ORG_RENAME:
        return ORG_RENAME[org_label]
    return local_from_label(org_label)

def extract_org(fullname: str) -> str:
    if "/" in fullname:
        return fullname.split("/")[0].strip()
    return ""

def extract_base_model_org(base_model: str) -> str:
    """Extract the org from a base model string like 'meta-llama/Meta-Llama-3-8B'.
    Returns '' if not parseable (e.g. 'Removed', merge strings, empty)."""
    if not base_model:
        return ""
    bm = base_model.strip()
    # Skip non-model values
    if bm.lower() in ("removed", "nan", "", "-"):
        return ""
    # Skip merge indicators (contain '(Merge)' or similar)
    if "(Merge)" in bm or "(merge)" in bm:
        return ""
    if "/" in bm:
        return bm.split("/")[0].strip()
    return ""

# _________ ONTOLOGY ORG LOOKUP __________

def load_known_org_locals(ont_path: Path) -> set[str]:
    """Parse organizations.ttl and return the set of org local names
    (i.e. the part after contextuel:) that are typed as LLMManufacturer or LLMProvider."""
    if not ont_path.exists():
        print(f"[WARNING] Ontology orgs file not found: {ont_path}")
        return set()
    text = ont_path.read_text(encoding="utf-8")
    orgs = set()
    for block in re.split(r'\n(?=contextuel:\w+\s)', text):
        m = re.match(r'contextuel:(\w+)', block)
        if m and ('enki:LLMManufacturer' in block or 'enki:LLMProvider' in block):
            orgs.add(m.group(1))
    return orgs


def load_known_provider_locals(ont_path: Path) -> set[str]:
    """Parse organizations.ttl and return the set of org local names
    that are typed specifically as enki:LLMProvider."""
    if not ont_path.exists():
        print(f"[WARNING] Ontology orgs file not found: {ont_path}")
        return set()
    text = ont_path.read_text(encoding="utf-8")
    providers = set()
    for block in re.split(r'\n(?=contextuel:\w+\s)', text):
        m = re.match(r'contextuel:(\w+)', block)
        if m and 'enki:LLMProvider' in block:
            providers.add(m.group(1))
    return providers


def classify_row_org(row: dict, known_org_locals: set[str]) -> bool:
    """Return True if the row's manufacturer org is in the ontology.
    A row is 'known' when the org that would be emitted as the manufacturer
    resolves (via org_uri_local) to a local name present in organizations.ttl."""
    org = row.get("org", "")
    if not org:
        # Models with no org get no manufacturer triple; put them in unmatched.
        return False

    base_org = row.get("base_model_org", "")
    # Mirror the manufacturer logic in build_graph:
    if base_org and base_org.lower() != org.lower():
        mfr_local = org_uri_local(base_org)
    else:
        mfr_local = org_uri_local(org)

    return mfr_local in known_org_locals

# _________ READ CSV __________

def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}")
        return rows

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Deduplicate: keep only the first occurrence of each fullname
    df = df.drop_duplicates(subset=["fullname"], keep="first")
    print(f"After dedup by fullname: {len(df)} rows")

    for _, r in df.iterrows():
        fullname = str(r.get("fullname", "")).strip()
        if not fullname:
            continue

        org = extract_org(fullname)

        lic = str(r.get("Hub License", "")).strip()
        if lic == "nan":
            lic = ""

        params = r.get("#Params (B)", None)
        params_str = ""
        if pd.notna(params):
            try:
                params_str = str(float(params))
            except (ValueError, TypeError):
                pass

        benchmarks = {}
        for bm_key, bm_info in BENCHMARKS.items():
            val = r.get(bm_info["col"], None)
            if pd.notna(val):
                try:
                    benchmarks[bm_key] = float(val)
                except (ValueError, TypeError):
                    pass

        # Extract base model to determine the true manufacturer
        base_model_raw = str(r.get("Base Model", "")).strip()
        if base_model_raw == "nan":
            base_model_raw = ""
        base_model_org = extract_base_model_org(base_model_raw)

        row = {
            "fullname": fullname,
            "label": fullname,
            "org": org,
            "license": lic,
            "params_b": params_str,
            "benchmarks": benchmarks,
            "base_model_raw": base_model_raw,
            "base_model_org": base_model_org,
        }
        rows.append(row)

    print(f"Total models: {len(rows)}")

    return rows

# _________ BUILD GRAPH __________

ORG_LOCAL = "Hugging_Face"

def build_graph(rows: list[dict], known_provider_locals: set[str]) -> tuple[Graph, dict[URIRef, list[URIRef]]]:
    g = Graph()
    g.bind("enki", ENKI)
    g.bind("contextuel", C)
    g.bind("dct", DCT)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # HuggingFace as evaluator org (definitional triples live in organizations.ttl)
    hf_org = URIRef(str(C) + ORG_LOCAL)

    # Pre-collect all model URI locals so we can detect license/model collisions
    model_locals = {local_from_model(r["fullname"]) for r in rows if r["fullname"]}

    # Track which license locals needed collision-avoidance prefixing
    prefixed_licenses = set()

    # Collect manufacturer and provisioner relationships
    manufacturer_models = defaultdict(set)
    provisioner_provisionings = defaultdict(list)

    # model_uri → [metric_uri, …] for serialisation ordering
    metrics_by_llm: dict[URIRef, list[URIRef]] = defaultdict(list)

    for r in rows:
        local = local_from_model(r["fullname"])
        model_uri = URIRef(str(C) + local)

        g.add((model_uri, RDF.type, ENKI.LLM))
        model_name = r["fullname"]
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]
        g.add((model_uri, RDFS.label, Literal(model_name)))

        # Identifier (HuggingFace link)
        hf_link = f"https://huggingface.co/{r['fullname']}"
        g.add((model_uri, DCT.identifier, Literal(hf_link)))

        # License (definitional triples live in licenses.ttl — only emit dct:license link)
        if r["license"] and r["license"] != "-":
            lic_local = normalize_license_local(r["license"])
            # Prefix with License_ only when the license local collides with a model local
            if lic_local in model_locals:
                prefixed_licenses.add((r["license"], lic_local))
                lic_local = f"License_{lic_local}"
            lic_uri = URIRef(str(C) + lic_local)
            g.add((model_uri, DCT.license, lic_uri))

        # Parameters (CSV value first, then name-based extraction as fallback)
        params_b = None
        if r["params_b"]:
            try:
                params_b = float(r["params_b"])
            except (ValueError, TypeError):
                pass
        if params_b is None:
            params_b = extract_params_billion(r["fullname"])
        if params_b is not None and params_b > 0:
            raw_count = int(params_b * 1_000_000_000)
            g.add((model_uri, ENKI.hasTotalParameterCount,
                   Literal(str(raw_count), datatype=XSD.nonNegativeInteger)))

        # Benchmark scores as EvaluatedMetric
        for bm_key, bm_val in r["benchmarks"].items():
            bm_label = BENCHMARKS[bm_key]["label"]
            metric_uri = URIRef(str(C) + f"{local}_{bm_key}")
            g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
            if bm_key == "CO2":
                g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + "CarbonCost")))
            else:
                g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + "Quality")))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, hf_org))
            g.add((metric_uri, ENKI.metricUnit, Literal(bm_label)))
            g.add((metric_uri, ENKI.metricValue, Literal(bm_val)))
            metrics_by_llm[model_uri].append(metric_uri)

        # --- Manufacturer logic (all models with an org) ---
        if not r["org"]:
            continue

        submitting_org = r["org"]
        base_org = r["base_model_org"]

        # Determine who the manufacturer is:
        #   - If base_model_org is parseable and differs from the submitting org,
        #     the base_model_org is the manufacturer.
        #   - If base_model_org matches the submitting org, or is empty/unparseable
        #     (self-submission or original model), the submitting org is the manufacturer.
        if base_org and base_org.lower() != submitting_org.lower():
            manufacturer_models[base_org].add(model_uri)
        else:
            manufacturer_models[submitting_org].add(model_uri)

        # --- Provisioner logic (only orgs confirmed as LLMProvider in ontology) ---
        # Determine which org would be the provisioner for this model:
        #   - If base_model_org differs → submitting org is the provisioner
        #   - Otherwise → submitting org is also the provisioner (self-hosted)
        if base_org and base_org.lower() != submitting_org.lower():
            candidate_prov = submitting_org
        else:
            candidate_prov = submitting_org

        candidate_prov_local = org_uri_local(candidate_prov)
        if candidate_prov_local in known_provider_locals:
            provisioner_provisionings[candidate_prov].append(
                (model_uri, local, model_name)
            )

    # Add manufacturer triples (definitional triples live in organizations.ttl)
    skipped_org_collisions = []
    for org_name, model_uris in manufacturer_models.items():
        org_local = org_uri_local(org_name)
        if org_local in model_locals:
            skipped_org_collisions.append(org_name)
            continue
        org_uri = URIRef(str(C) + org_local)
        for m_uri in sorted(model_uris, key=lambda u: str(u).lower()):
            g.add((org_uri, ENKI.manufactures, m_uri))

    # Add LLMProvisioning triples (no LLMProvider type — that lives in organizations.ttl)
    for org_name, prov_list in provisioner_provisionings.items():
        org_local = org_uri_local(org_name)
        if org_local in model_locals:
            # Already reported in the manufacturer loop (or will be below)
            continue
        org_uri = URIRef(str(C) + org_local)

        for model_uri, model_local, model_label in prov_list:
            prov_local = f"{org_local}_Provisioning_{model_local}"
            prov_uri = URIRef(str(C) + prov_local)
            g.add((prov_uri, RDF.type, ENKI.LLMProvisioning))
            g.add((prov_uri, RDFS.label, Literal(f"{org_name} provisioning of {model_label}")))
            g.add((prov_uri, ENKI.provider, org_uri))
            g.add((prov_uri, ENKI.provides, model_uri))

    # Stats
    mfr_count = sum(len(m) for m in manufacturer_models.values())
    prov_count = sum(len(p) for p in provisioner_provisionings.values())
    dual_orgs = set(manufacturer_models.keys()) & set(provisioner_provisionings.keys())
    prov_only = set(provisioner_provisionings.keys()) - set(manufacturer_models.keys())
    mfr_only = set(manufacturer_models.keys()) - set(provisioner_provisionings.keys())
    print(f"Manufacturers: {len(manufacturer_models)} orgs ({mfr_count} models)")
    print(f"Provisioners:  {len(provisioner_provisionings)} orgs ({prov_count} provisionings)")
    print(f"Dual (mfr+prov): {len(dual_orgs)} | Mfr-only: {len(mfr_only)} | Prov-only: {len(prov_only)}")
    if prefixed_licenses:
        print(f"License URI collisions resolved ({len(prefixed_licenses)}):")
        for lic_label, lic_local in sorted(prefixed_licenses):
            print(f"  '{lic_label}' ({lic_local}) → License_{lic_local}")
    else:
        print("No license/model URI collisions detected.")
    if skipped_org_collisions:
        print(f"Org/model URI collisions — skipped org triples ({len(skipped_org_collisions)}):")
        for org_name in sorted(skipped_org_collisions):
            print(f"  '{org_name}' collides with model URI contextuel:{local_from_label(org_name)}")

    # Sort each model's metrics for deterministic output
    for model_uri in metrics_by_llm:
        metrics_by_llm[model_uri] = sorted(metrics_by_llm[model_uri], key=lambda s: str(s).lower())

    return g, metrics_by_llm

# _________ EVALUATIONS __________

def add_evaluations(g: Graph, rows: list[dict]):
    """Add Evaluation triples (evaluees, metricEvaluated, testsCapability)."""
    hf_org = URIRef(str(C) + ORG_LOCAL)

    eval_uris = {}
    for bm_key, bm_info in BENCHMARKS.items():
        eval_label = f"HF Open LLM Leaderboard {bm_info['label']}"
        eval_uri = URIRef(str(C) + local_from_label(eval_label))
        g.add((eval_uri, RDF.type, ENKI.Evaluation))
        g.add((eval_uri, RDFS.label, Literal(eval_label)))
        g.add((eval_uri, ENKI.evaluator, hf_org))

        # testsCapability
        for cap_local, _, _ in BENCHMARK_CAPABILITIES.get(bm_key, []):
            g.add((eval_uri, ENKI.testsCapability, URIRef(str(C) + cap_local)))

        eval_uris[bm_key] = eval_uri

    # Link models as evaluees + metricEvaluated
    for r in rows:
        local = local_from_model(r["fullname"])
        model_uri = URIRef(str(C) + local)

        for bm_key in r["benchmarks"]:
            g.add((eval_uris[bm_key], ENKI.evaluee, model_uri))
            metric_uri = URIRef(str(C) + f"{local}_{bm_key}")
            g.add((eval_uris[bm_key], ENKI.metricEvaluated, metric_uri))

# _________ SERIALISATION __________

def reorder_and_serialize(g: Graph, out_path: Path, metrics_by_llm: dict[URIRef, list[URIRef]], backup_path: Path | None = None):
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

    # Org subjects: identified by having enki:manufactures predicate
    org_subjects = sorted(
        set(g.subjects(ENKI.manufactures, None)),
        key=lambda s: str(s).lower(),
    )
    provisionings = sorted(set(g.subjects(RDF.type, ENKI.LLMProvisioning)), key=lambda s: str(s).lower())
    evaluations = sorted(set(g.subjects(RDF.type, ENKI.Evaluation)), key=lambda s: str(s).lower())
    llms = sorted(set(g.subjects(RDF.type, ENKI.LLM)), key=lambda s: str(s).lower())

    # Group provisionings by their provider org
    provisionings_by_org = defaultdict(list)
    for p in provisionings:
        for org_uri in g.objects(p, ENKI.provider):
            provisionings_by_org[org_uri].append(p)
    for org_uri in provisionings_by_org:
        provisionings_by_org[org_uri] = sorted(
            provisionings_by_org[org_uri], key=lambda s: str(s).lower()
        )

    ordered_subjects = []

    def add_block(subjects):
        for s in subjects:
            if s not in seen_subjects:
                ordered_subjects.append(s)
                seen_subjects.add(s)

    # 1. Orgs (manufactures + provisionings)
    for org in org_subjects:
        if org not in seen_subjects:
            ordered_subjects.append(org)
            seen_subjects.add(org)
        for p in provisionings_by_org.get(org, []):
            if p not in seen_subjects:
                ordered_subjects.append(p)
                seen_subjects.add(p)

    # 2. Any remaining provisionings
    add_block(provisionings)

    # 3. Evaluations
    add_block(evaluations)

    # 4. LLMs, each followed by their metrics
    for llm in llms:
        if llm in seen_subjects:
            continue
        ordered_subjects.append(llm)
        seen_subjects.add(llm)
        for m in metrics_by_llm.get(llm, []):
            if m not in seen_subjects:
                ordered_subjects.append(m)
                seen_subjects.add(m)

    # 5. Anything remaining
    for s in sorted(set(g.subjects()), key=lambda s: str(s).lower()):
        if s not in seen_subjects:
            ordered_subjects.append(s)
            seen_subjects.add(s)

    lines = serialize_subjects(ordered_subjects)

    if out_path.exists():
        bp = backup_path or out_path.with_suffix(".ttl.bak")
        bp.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(HEADER_BLOCK)
        fh.write("\n".join(lines))
        fh.write("\n")

    bp = backup_path or out_path.with_suffix(".ttl.bak")
    print(f"Wrote HF leaderboard TTL to {out_path} (backup at {bp})")

# _________ MAIN __________

def main():
    rows = read_rows(SOURCE_CSV)
    if not rows:
        print("No rows read. Exiting.")
        return

    # Load known orgs from the ontology
    known_org_locals = load_known_org_locals(ONTOLOGY_ORGS_TTL)
    known_provider_locals = load_known_provider_locals(ONTOLOGY_ORGS_TTL)
    print(f"Loaded {len(known_org_locals)} known orgs from ontology ({len(known_provider_locals)} are LLMProvider)")

    # Split rows into known-org and unknown-org sets
    known_rows = []
    unmatched_rows = []
    for r in rows:
        if classify_row_org(r, known_org_locals):
            known_rows.append(r)
        else:
            unmatched_rows.append(r)

    print(f"Known-org models: {len(known_rows)}, Unmatched-org models: {len(unmatched_rows)}")

    # Build and serialise the known-org file
    print(f"\n=== Building known-org graph from {len(known_rows)} models ===")
    g_known, metrics_known = build_graph(known_rows, known_provider_locals)
    add_evaluations(g_known, known_rows)
    reorder_and_serialize(g_known, OUTPUT_TTL, metrics_known, BACKUP)

    # Build and serialise the unmatched-org file
    print(f"\n=== Building unmatched-org graph from {len(unmatched_rows)} models ===")
    g_unmatched, metrics_unmatched = build_graph(unmatched_rows, known_provider_locals)
    add_evaluations(g_unmatched, unmatched_rows)
    reorder_and_serialize(g_unmatched, OUTPUT_UNMATCHED_TTL, metrics_unmatched, BACKUP_UNMATCHED)

if __name__ == "__main__":
    main()
