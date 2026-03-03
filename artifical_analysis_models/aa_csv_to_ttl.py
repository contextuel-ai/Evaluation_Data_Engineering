"""
Artificial Analysis CSV → Turtle (TTL) conversion for the Contextuel.ai knowledge graph.

Changes from the HF leaderboard script (hf_leaderboard_csv_to_ttl.py):
  - Uses csv module (not pandas) since the AA CSV is simple and well-formed.
  - Model URI derived from model_name via local_from_label (no org prefix to strip).
  - All models are API-provisioned: every creator is both LLMManufacturer and LLMProvider.
    No "Official Providers" / "Base Model" columns needed.
  - Evaluator org is Artificial Analysis (not HuggingFace).
  - 15 quality benchmarks (vs HF's 6), plus 3 speed and 3 price operational metrics.
  - Speed metrics use metricAssesses → contextuel:Latency (enki:Criterion).
  - Price metrics use metricAssesses → contextuel:Cost (enki:Criterion).
  - dct:identifier stores the AA page URL (https://artificialanalysis.ai/models/{slug}).
  - No license data in the AA CSV, so no dct:License triples.
  - Capabilities mapped per-benchmark following the same (local, label, description) tuple format.
"""

from pathlib import Path
from collections import OrderedDict, defaultdict
import csv
import re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "aa_models_with_pricing_speed_and_benchmarks.csv"
OUTPUT_TTL = ROOT / "aa_models.ttl"
BACKUP = OUTPUT_TTL.with_suffix(".ttl.bak")

C = Namespace("https://www.contextuel.ai/spec/Contextuel/")
ENKI = Namespace("https://www.contextuel.ai/spec/Enki/")
DCT = Namespace("http://purl.org/dc/terms/")

HEADER_BLOCK = """@base <https://www.contextuel.ai/spec/Enki/> .
@prefix enki: <https://www.contextuel.ai/spec/Enki/> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix contextuel: <https://www.contextuel.ai/spec/Contextuel/> .

"""

# _________ BENCHMARKS & OPERATIONAL METRICS __________

# Quality benchmarks (metricAssesses → Quality)
BENCHMARKS = {
    "AIME":          {"col": "aime",          "label": "AIME"},
    "AIME_25":       {"col": "aime_25",       "label": "AIME 2025"},
    "AA_Coding_Index": {"col": "artificial_analysis_coding_index",
                        "label": "Artificial Analysis Coding Index"},
    "AA_Intelligence_Index": {"col": "artificial_analysis_intelligence_index",
                              "label": "Artificial Analysis Intelligence Index"},
    "AA_Math_Index": {"col": "artificial_analysis_math_index",
                      "label": "Artificial Analysis Math Index"},
    "GPQA":          {"col": "gpqa",          "label": "GPQA"},
    "HLE":           {"col": "hle",           "label": "HLE"},
    "IFBench":       {"col": "ifbench",       "label": "IFBench"},
    "LCR":           {"col": "lcr",           "label": "LCR"},
    "LiveCodeBench": {"col": "livecodebench", "label": "LiveCodeBench"},
    "MATH_500":      {"col": "math_500",      "label": "MATH 500"},
    "MMLU_PRO":      {"col": "mmlu_pro",      "label": "MMLU-PRO"},
    "SciCode":       {"col": "scicode",       "label": "SciCode"},
    "Tau2":          {"col": "tau2",          "label": "Tau2"},
    "TerminalBench_Hard": {"col": "terminalbench_hard", "label": "TerminalBench Hard"},
}

# Speed / latency metrics (metricAssesses → Latency)
SPEED_METRICS = {
    "Output_Tokens_Per_Second":  {"col": "output_tokens_per_second",
                                   "label": "Output Tokens per Second"},
    "Time_To_First_Token":       {"col": "time_to_first_token",
                                   "label": "Time to First Token (s)"},
    "Median_TTFAT":              {"col": "median_time_to_first_answer_token",
                                   "label": "Median Time to First Answer Token (s)"},
}

# Price / cost metrics (metricAssesses → Cost)
PRICE_METRICS = {
    "Price_1M_Input_Tokens":   {"col": "price_1m_input_tokens",
                                 "label": "Price per 1M Input Tokens ($)"},
    "Price_1M_Output_Tokens":  {"col": "price_1m_output_tokens",
                                 "label": "Price per 1M Output Tokens ($)"},
    "Price_1M_Blended_3_To_1": {"col": "price_1m_blended_3_to_1",
                                 "label": "Price per 1M Blended 3:1 Tokens ($)"},
}

# _________ CAPABILITY MAPPING __________

BENCHMARK_CAPABILITIES = {
    "AIME": [
        ("MathematicalReasoning", "Mathematical Reasoning",
         "The ability of an LLM to solve mathematical problems requiring formal symbolic manipulation, numerical computation, and multi-step logical derivation across domains such as algebra, geometry, number theory, combinatorics, and calculus."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "AIME_25": [
        ("MathematicalReasoning", "Mathematical Reasoning",
         "The ability of an LLM to solve mathematical problems requiring formal symbolic manipulation, numerical computation, and multi-step logical derivation across domains such as algebra, geometry, number theory, combinatorics, and calculus."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "AA_Coding_Index": [
        ("CodeGeneration", "Code Generation",
         "The ability of an LLM to produce syntactically correct, functional, and idiomatic source code from natural-language specifications, partial implementations, or test cases."),
    ],
    "AA_Intelligence_Index": [
        ("GeneralKnowledgeReasoning", "General Knowledge Reasoning",
         "The ability of an LLM to apply broad academic knowledge across multiple disciplines to correctly answer domain-specific questions."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "AA_Math_Index": [
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
    "HLE": [
        ("ExpertLevelProblemSolving", "Expert-Level Problem Solving",
         "The ability of an LLM to solve problems that require graduate-level or professional-level expertise in a specialized domain, integrating deep domain knowledge with sophisticated reasoning that resists simple lookup or surface-level understanding."),
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
    ],
    "IFBench": [
        ("InstructionFollowing", "Instruction Following",
         "The ability of an LLM to understand and correctly execute user-provided instructions."),
    ],
    "LCR": [
        ("LongContextReasoning", "Long-Context Reasoning",
         "The ability of an LLM to accurately retrieve, synthesise, and reason over information distributed across long input contexts, including multi-document comprehension, needle-in-a-haystack recall, and cross-passage inference."),
    ],
    "LiveCodeBench": [
        ("CodeGeneration", "Code Generation",
         "The ability of an LLM to produce syntactically correct, functional, and idiomatic source code from natural-language specifications, partial implementations, or test cases."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "MATH_500": [
        ("MathematicalReasoning", "Mathematical Reasoning",
         "The ability of an LLM to solve mathematical problems requiring formal symbolic manipulation, numerical computation, and multi-step logical derivation across domains such as algebra, geometry, number theory, combinatorics, and calculus."),
    ],
    "MMLU_PRO": [
        ("GeneralKnowledgeReasoning", "General Knowledge Reasoning",
         "The ability of an LLM to apply broad academic knowledge across multiple disciplines to correctly answer domain-specific questions."),
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
        ("FactualRecall", "Factual Recall",
         "The ability of an LLM to accurately retrieve and reproduce learned factual information from its training data."),
    ],
    "SciCode": [
        ("CodeGeneration", "Code Generation",
         "The ability of an LLM to produce syntactically correct, functional, and idiomatic source code from natural-language specifications, partial implementations, or test cases."),
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
    ],
    "Tau2": [
        ("InstructionFollowing", "Instruction Following",
         "The ability of an LLM to understand and correctly execute user-provided instructions."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "TerminalBench_Hard": [
        ("CodeGeneration", "Code Generation",
         "The ability of an LLM to produce syntactically correct, functional, and idiomatic source code from natural-language specifications, partial implementations, or test cases."),
        ("ToolUse", "Tool Use",
         "The ability of an LLM to correctly select, parameterise, and sequence external tools, APIs, or command-line utilities to accomplish a task that cannot be completed with text generation alone."),
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
    """For general labels (evaluations, capabilities, licenses, orgs).
    Does NOT strip org prefixes. Capitalises each word part, preserves existing uppercase."""
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


def safe_float(val: str) -> float | None:
    """Parse a CSV string to float, returning None for empty/missing."""
    if not val or val.strip() in ("", "-", "nan"):
        return None
    try:
        return float(val.strip())
    except (ValueError, TypeError):
        return None


# _________ EVALUATOR ORG __________

ORG_LOCAL = "Artificial_Analysis"


# _________ READ CSV __________

def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}")
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            model_name = (r.get("model_name") or "").strip()
            if not model_name:
                continue

            creator = (r.get("creator") or "").strip()
            slug = (r.get("model_slug") or "").strip()

            # Quality benchmarks
            benchmarks: dict[str, float] = {}
            for bm_key, bm_info in BENCHMARKS.items():
                v = safe_float(r.get(bm_info["col"], ""))
                if v is not None:
                    benchmarks[bm_key] = v

            # Speed metrics
            speed: dict[str, float] = {}
            for sm_key, sm_info in SPEED_METRICS.items():
                v = safe_float(r.get(sm_info["col"], ""))
                if v is not None:
                    speed[sm_key] = v

            # Price metrics
            price: dict[str, float] = {}
            for pm_key, pm_info in PRICE_METRICS.items():
                v = safe_float(r.get(pm_info["col"], ""))
                if v is not None:
                    price[pm_key] = v

            rows.append({
                "model_name": model_name,
                "creator": creator,
                "slug": slug,
                "model_id": (r.get("model_id") or "").strip(),
                "benchmarks": benchmarks,
                "speed": speed,
                "price": price,
            })

    print(f"Loaded {len(rows)} models from {path.name}")
    creators = set(r["creator"] for r in rows if r["creator"])
    print(f"Unique creators: {len(creators)}")
    return rows


# _________ BUILD GRAPH __________

def build_graph(rows: list[dict]) -> Graph:
    g = Graph()
    g.bind("enki", ENKI)
    g.bind("contextuel", C)
    g.bind("dct", DCT)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Artificial Analysis as evaluator org
    aa_org = URIRef(str(C) + ORG_LOCAL)
    g.add((aa_org, RDF.type, ENKI.AIOrganization))
    g.add((aa_org, RDFS.label, Literal("Artificial Analysis")))
    g.add((aa_org, DCT.identifier, Literal("https://artificialanalysis.ai/")))
    g.add((aa_org, DCT.description, Literal(
        "Artificial Analysis provides independent benchmarks, speed tests, and pricing data for LLM API providers."
    )))

    # Collect manufacturer and provisioner relationships
    # In AA, every creator is both manufacturer and provider (all API-served)
    manufacturer_models = defaultdict(set)
    provisioner_provisionings = defaultdict(list)

    for r in rows:
        local = local_from_label(r["model_name"])
        model_uri = URIRef(str(C) + local)

        g.add((model_uri, RDF.type, ENKI.LLM))
        g.add((model_uri, RDFS.label, Literal(r["model_name"])))

        # Identifier (Artificial Analysis page)
        if r["slug"]:
            aa_link = f"https://artificialanalysis.ai/models/{r['slug']}"
            g.add((model_uri, DCT.identifier, Literal(aa_link)))

        # Quality benchmark scores as EvaluatedMetric
        for bm_key, bm_val in r["benchmarks"].items():
            bm_label = BENCHMARKS[bm_key]["label"]
            metric_uri = URIRef(str(C) + f"{local}_{bm_key}")
            g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
            g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + "Quality")))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, aa_org))
            g.add((metric_uri, ENKI.metricUnit, Literal(bm_label)))
            g.add((metric_uri, ENKI.metricValue, Literal(bm_val)))

        # Speed metrics as EvaluatedMetric → Latency
        for sm_key, sm_val in r["speed"].items():
            sm_label = SPEED_METRICS[sm_key]["label"]
            metric_uri = URIRef(str(C) + f"{local}_{sm_key}")
            g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
            g.add((metric_uri, ENKI.metricAssesses, latency_uri))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, aa_org))
            g.add((metric_uri, ENKI.metricUnit, Literal(sm_label)))
            g.add((metric_uri, ENKI.metricValue, Literal(sm_val)))

        # Price metrics as EvaluatedMetric → Cost
        for pm_key, pm_val in r["price"].items():
            pm_label = PRICE_METRICS[pm_key]["label"]
            metric_uri = URIRef(str(C) + f"{local}_{pm_key}")
            g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
            g.add((metric_uri, ENKI.metricAssesses, cost_uri))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, aa_org))
            g.add((metric_uri, ENKI.metricUnit, Literal(pm_label)))
            g.add((metric_uri, ENKI.metricValue, Literal(pm_val)))

        # Manufacturer and provisioner (all AA models are API-provisioned)
        if r["creator"]:
            creator_name = r["creator"]
            manufacturer_models[creator_name].add(model_uri)
            provisioner_provisionings[creator_name].append(
                (model_uri, local, r["model_name"])
            )

    # Add LLMManufacturer instances
    for org_name, model_uris in manufacturer_models.items():
        org_local = local_from_label(org_name)
        org_uri = URIRef(str(C) + org_local)
        g.add((org_uri, RDF.type, ENKI.LLMManufacturer))
        g.add((org_uri, RDFS.label, Literal(org_name)))
        for m_uri in sorted(model_uris, key=lambda u: str(u).lower()):
            g.add((org_uri, ENKI.manufactures, m_uri))

    # Add LLMProvider + LLMProvisioning instances
    for org_name, prov_list in provisioner_provisionings.items():
        org_local = local_from_label(org_name)
        org_uri = URIRef(str(C) + org_local)
        g.add((org_uri, RDF.type, ENKI.LLMProvider))

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
    print(f"Manufacturers: {len(manufacturer_models)} orgs ({mfr_count} models)")
    print(f"Provisioners:  {len(provisioner_provisionings)} orgs ({prov_count} provisionings)")

    return g


# _________ CAPABILITIES __________

def add_capabilities(g: Graph):
    """Add enki:Capability instances (quality benchmarks only — speed/price are not capabilities)."""
    seen = set()
    for bm_key, caps in BENCHMARK_CAPABILITIES.items():
        for cap_local, cap_label, cap_desc in caps:
            if cap_local in seen:
                continue
            seen.add(cap_local)
            cap_uri = URIRef(str(C) + cap_local)
            g.add((cap_uri, RDF.type, ENKI.Capability))
            g.add((cap_uri, RDFS.label, Literal(cap_label)))
            g.add((cap_uri, DCT.description, Literal(cap_desc)))


# _________ EVALUATIONS __________

def add_evaluations(g: Graph, rows: list[dict]):
    aa_org = URIRef(str(C) + ORG_LOCAL)

    # Quality benchmark evaluations
    eval_uris: dict[str, URIRef] = {}
    for bm_key, bm_info in BENCHMARKS.items():
        eval_label = f"Artificial Analysis {bm_info['label']}"
        eval_uri = URIRef(str(C) + local_from_label(eval_label))
        g.add((eval_uri, RDF.type, ENKI.Evaluation))
        g.add((eval_uri, RDFS.label, Literal(eval_label)))
        g.add((eval_uri, ENKI.evaluator, aa_org))

        # testsCapability
        for cap_local, _, _ in BENCHMARK_CAPABILITIES.get(bm_key, []):
            g.add((eval_uri, ENKI.testsCapability, URIRef(str(C) + cap_local)))

        eval_uris[bm_key] = eval_uri

    # Speed metric evaluations (no testsCapability — these measure latency, not a capability)
    for sm_key, sm_info in SPEED_METRICS.items():
        eval_label = f"Artificial Analysis {sm_info['label']}"
        eval_uri = URIRef(str(C) + local_from_label(eval_label))
        g.add((eval_uri, RDF.type, ENKI.Evaluation))
        g.add((eval_uri, RDFS.label, Literal(eval_label)))
        g.add((eval_uri, ENKI.evaluator, aa_org))
        eval_uris[sm_key] = eval_uri

    # Price metric evaluations (no testsCapability — these measure cost, not a capability)
    for pm_key, pm_info in PRICE_METRICS.items():
        eval_label = f"Artificial Analysis {pm_info['label']}"
        eval_uri = URIRef(str(C) + local_from_label(eval_label))
        g.add((eval_uri, RDF.type, ENKI.Evaluation))
        g.add((eval_uri, RDFS.label, Literal(eval_label)))
        g.add((eval_uri, ENKI.evaluator, aa_org))
        eval_uris[pm_key] = eval_uri

    # Link models as evaluees + metricEvaluated
    for r in rows:
        local = local_from_label(r["model_name"])
        model_uri = URIRef(str(C) + local)

        for bm_key in r["benchmarks"]:
            g.add((eval_uris[bm_key], ENKI.evaluee, model_uri))
            metric_uri = URIRef(str(C) + f"{local}_{bm_key}")
            g.add((eval_uris[bm_key], ENKI.metricEvaluated, metric_uri))

        for sm_key in r["speed"]:
            g.add((eval_uris[sm_key], ENKI.evaluee, model_uri))
            metric_uri = URIRef(str(C) + f"{local}_{sm_key}")
            g.add((eval_uris[sm_key], ENKI.metricEvaluated, metric_uri))

        for pm_key in r["price"]:
            g.add((eval_uris[pm_key], ENKI.evaluee, model_uri))
            metric_uri = URIRef(str(C) + f"{local}_{pm_key}")
            g.add((eval_uris[pm_key], ENKI.metricEvaluated, metric_uri))


# _________ SERIALISATION __________

def reorder_and_serialize(g: Graph, out_path: Path):
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
            if o.datatype in (XSD.float, XSD.double):
                return str(float(o))
            if o.datatype == XSD.decimal:
                return str(o)
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
            if types:
                if len(types) == 1:
                    block.append(f"{curie(s)} a {curie(types[0])}")
                else:
                    type_strs = ",\n        ".join(curie(t) for t in types)
                    block.append(f"{curie(s)} a {type_strs}")
            else:
                block.append(f"{curie(s)}")
            for p in preds:
                if p == RDF.type:
                    continue
                objs = sorted(set(g.objects(s, p)), key=lambda o: str(o))
                for o in objs:
                    block.append(f"    {curie(p)} {format_obj(o)}")
            block_lines.append(" ;\n".join(block) + " .")
            block_lines.append("")
        return block_lines

    seen_subjects = set()

    licenses = sorted(set(g.subjects(RDF.type, DCT.License)), key=lambda s: str(s).lower())
    criteria = sorted(set(g.subjects(RDF.type, ENKI.Criterion)), key=lambda s: str(s).lower())
    manufacturers = sorted(set(g.subjects(RDF.type, ENKI.LLMManufacturer)), key=lambda s: str(s).lower())
    provisionings = sorted(set(g.subjects(RDF.type, ENKI.LLMProvisioning)), key=lambda s: str(s).lower())
    evaluations = sorted(set(g.subjects(RDF.type, ENKI.Evaluation)), key=lambda s: str(s).lower())
    capabilities = sorted(set(g.subjects(RDF.type, ENKI.Capability)), key=lambda s: str(s).lower())
    ai_orgs = sorted(set(g.subjects(RDF.type, ENKI.AIOrganization)), key=lambda s: str(s).lower())
    llms = sorted(set(g.subjects(RDF.type, ENKI.LLM)), key=lambda s: str(s).lower())
    metrics = set(g.subjects(RDF.type, ENKI.EvaluatedMetric))

    def local_name_of(u):
        s = str(u)
        return s.rstrip("/").split("/")[-1]

    # Group provisionings by their provider org
    provisionings_by_org = defaultdict(list)
    for p in provisionings:
        for org_uri in g.objects(p, ENKI.provider):
            provisionings_by_org[org_uri].append(p)
    for org_uri in provisionings_by_org:
        provisionings_by_org[org_uri] = sorted(
            provisionings_by_org[org_uri], key=lambda s: str(s).lower()
        )

    metrics_by_llm = defaultdict(list)

    # Sort LLMs by local name length DESCENDING so longer names match first
    llms_by_length = sorted(llms, key=lambda u: len(local_name_of(u)), reverse=True)
    claimed_metrics = set()

    for llm in llms_by_length:
        llm_loc = local_name_of(llm)
        for m in metrics:
            if m in claimed_metrics:
                continue
            mloc = local_name_of(m)
            if mloc.startswith(llm_loc + "_"):
                metrics_by_llm[llm].append(m)
                claimed_metrics.add(m)
        metrics_by_llm[llm] = sorted(set(metrics_by_llm[llm]), key=lambda s: str(s).lower())

    ordered_subjects = []

    def add_block(subjects):
        for s in subjects:
            if s not in seen_subjects:
                ordered_subjects.append(s)
                seen_subjects.add(s)

    # 1. Licenses (none in AA, but kept for consistency)
    add_block(licenses)

    # 2. Criteria (Latency, Cost)
    add_block(criteria)

    # 3. Manufacturers/Providers, each followed by their provisionings
    for mfr in manufacturers:
        if mfr not in seen_subjects:
            ordered_subjects.append(mfr)
            seen_subjects.add(mfr)
        for p in provisionings_by_org.get(mfr, []):
            if p not in seen_subjects:
                ordered_subjects.append(p)
                seen_subjects.add(p)

    # 4. Any remaining provisionings (safety net)
    add_block(provisionings)

    # 5. Evaluations
    add_block(evaluations)

    # 6. Capabilities
    add_block(capabilities)

    # 7. AI Orgs
    add_block(ai_orgs)

    # 8. LLMs, each followed by their metrics
    for llm in llms:
        if llm in seen_subjects:
            continue
        ordered_subjects.append(llm)
        seen_subjects.add(llm)
        for m in metrics_by_llm.get(llm, []):
            if m not in seen_subjects:
                ordered_subjects.append(m)
                seen_subjects.add(m)

    # 9. Anything remaining
    for s in sorted(set(g.subjects()), key=lambda s: str(s).lower()):
        if s not in seen_subjects:
            ordered_subjects.append(s)
            seen_subjects.add(s)

    lines = serialize_subjects(ordered_subjects)

    if out_path.exists():
        BACKUP.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(HEADER_BLOCK)
        fh.write("\n".join(lines))
        fh.write("\n")

    print(f"Wrote AA TTL to {out_path} (backup at {BACKUP})")


# _________ MAIN __________

def main():
    rows = read_rows(SOURCE_CSV)
    if not rows:
        print("No rows read. Exiting.")
        return

    print(f"Building graph from {len(rows)} models...")
    g = build_graph(rows)
    add_capabilities(g)
    add_evaluations(g, rows)
    reorder_and_serialize(g, OUTPUT_TTL)


if __name__ == "__main__":
    main()
