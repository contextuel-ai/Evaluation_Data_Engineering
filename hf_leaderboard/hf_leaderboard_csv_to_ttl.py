from pathlib import Path
from collections import OrderedDict, defaultdict
import pandas as pd
import re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "hf_leaderboard_long_20260221.csv"
OUTPUT_TTL = ROOT / "hf_leaderboard.ttl"
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

# The 6 benchmarks on the HF Open LLM Leaderboard v2
BENCHMARKS = {
    "IFEval":      {"col": "IFEval",      "label": "IFEval"},
    "BBH":         {"col": "BBH",         "label": "BBH"},
    "MATH_Lvl_5":  {"col": "MATH Lvl 5",  "label": "MATH Lvl 5"},
    "GPQA":        {"col": "GPQA",        "label": "GPQA"},
    "MUSR":        {"col": "MUSR",        "label": "MUSR"},
    "MMLU_PRO":    {"col": "MMLU-PRO",    "label": "MMLU-PRO"},
    "CO2":         {"col": "CO₂ cost (kg)", "label": "CO₂ cost (kg)"},
}

# Capabilities tested by each benchmark (mirroring lmarena_trimmed.ttl pattern)
BENCHMARK_CAPABILITIES = {
    "IFEval": [
        ("InstructionFollowing", "Instruction Following",
         "The ability of an LLM to understand and correctly execute user-provided instructions."),
        ("OutputFormatCompliance", "Output Format Compliance",
         "The ability of an LLM to produce outputs that conform to specified formatting constraints."),
    ],
    "BBH": [
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
        ("AlgorithmicReasoning", "Algorithmic Reasoning",
         "The ability of an LLM to follow and execute step-by-step algorithmic procedures."),
        ("LogicalDeduction", "Logical Deduction",
         "The ability of an LLM to derive conclusions from given premises using formal logical rules."),
    ],
    "MATH_Lvl_5": [
        ("AdvancedMathematicalReasoning", "Advanced Mathematical Reasoning",
         "The ability of an LLM to solve competition-level mathematics problems requiring deep mathematical knowledge."),
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
    ],
    "GPQA": [
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
        ("GraduateLevelReasoning", "Graduate-Level Reasoning",
         "The ability of an LLM to reason at the level expected of domain experts in graduate-level science."),
    ],
    "MUSR": [
        ("MultiStepReasoning", "Multi-Step Reasoning",
         "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."),
        ("LongContextUnderstanding", "Long Context Understanding",
         "The ability of an LLM to comprehend and reason over long narrative passages."),
    ],
    "MMLU_PRO": [
        ("AcademicDomainKnowledge", "Academic Domain Knowledge",
         "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."),
        ("FactualRecall", "Factual Recall",
         "The ability of an LLM to accurately retrieve and reproduce learned factual information from its training data."),
        ("GeneralKnowledgeReasoning", "General Knowledge Reasoning",
         "The ability of an LLM to apply broad academic knowledge across multiple disciplines to correctly answer domain-specific questions."),
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

def normalize_license_local(lic: str) -> str:
    if not lic:
        return ""
    l = lic.strip()
    if re.match(r'(?i)^apache[ _\-.]?2[ _\-.]?0$', l):
        return local_from_label("Apache 2.0")
    return local_from_label(l)

def extract_org(fullname: str) -> str:
    if "/" in fullname:
        return fullname.split("/")[0].strip()
    return ""

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

        row = {
            "fullname": fullname,
            "label": fullname,
            "org": org,
            "license": lic,
            "params_b": params_str,
            "benchmarks": benchmarks,
        }
        rows.append(row)

    return rows

# _________ BUILD GRAPH __________

ORG_LOCAL = "Hugging_Face"

def build_graph(rows: list[dict]) -> Graph:
    g = Graph()
    g.bind("enki", ENKI)
    g.bind("contextuel", C)
    g.bind("dct", DCT)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    hf_org = URIRef(str(C) + ORG_LOCAL)
    g.add((hf_org, RDF.type, ENKI.AIOrganization))
    g.add((hf_org, RDFS.label, Literal("Hugging Face")))
    g.add((hf_org, DCT.identifier, Literal("https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard")))
    g.add((hf_org, DCT.description, Literal(
        "The Open LLM Leaderboard is a HuggingFace community benchmark that evaluates open-source LLMs across multiple standardised tasks."
    )))

    for r in rows:
        local = local_from_model(r["fullname"])
        model_uri = URIRef(str(C) + local)

        g.add((model_uri, RDF.type, ENKI.LLM))
        # Label is just the model name without org prefix
        model_name = r["fullname"]
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]
        g.add((model_uri, RDFS.label, Literal(model_name)))

        # Identifier (HuggingFace link)
        hf_link = f"https://huggingface.co/{r['fullname']}"
        g.add((model_uri, DCT.identifier, Literal(hf_link)))

        # License
        if r["license"] and r["license"] != "-":
            lic_local = normalize_license_local(r["license"])
            lic_uri = URIRef(str(C) + lic_local)
            g.add((lic_uri, RDF.type, DCT.License))
            g.add((lic_uri, RDFS.label, Literal(r["license"])))
            g.add((model_uri, DCT.license, lic_uri))

        # Parameters
        if r["params_b"]:
            try:
                g.add((model_uri, ENKI.hasTotalParameterCount,
                       Literal(float(r["params_b"]), datatype=XSD.double)))
            except (ValueError, TypeError):
                pass

        # Benchmark scores as EvaluatedMetric
        for bm_key, bm_val in r["benchmarks"].items():
            bm_label = BENCHMARKS[bm_key]["label"]
            metric_uri = URIRef(str(C) + f"{local}_{bm_key}")
            g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
            # CO2 assesses CarbonCost, all others assess Quality
            if bm_key == "CO2":
                g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + "CarbonCost")))
            else:
                g.add((metric_uri, ENKI.metricAssesses, URIRef(str(C) + "Quality")))
            g.add((metric_uri, ENKI.metricAssignee, model_uri))
            g.add((metric_uri, ENKI.metricAssigner, hf_org))
            g.add((metric_uri, ENKI.metricUnit, Literal(bm_label)))
            g.add((metric_uri, ENKI.metricValue, Literal(bm_val)))

    return g

# _________ CAPABILITIES __________

def add_capabilities(g: Graph):
    """Add enki:Capability instances, same pattern as lmarena_trimmed.ttl."""
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
    evaluations = sorted(set(g.subjects(RDF.type, ENKI.Evaluation)), key=lambda s: str(s).lower())
    capabilities = sorted(set(g.subjects(RDF.type, ENKI.Capability)), key=lambda s: str(s).lower())
    ai_orgs = sorted(set(g.subjects(RDF.type, ENKI.AIOrganization)), key=lambda s: str(s).lower())
    llms = sorted(set(g.subjects(RDF.type, ENKI.LLM)), key=lambda s: str(s).lower())
    metrics = set(g.subjects(RDF.type, ENKI.EvaluatedMetric))

    def local_name_of(u):
        s = str(u)
        return s.rstrip("/").split("/")[-1]

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

    add_block(licenses)
    add_block(evaluations)
    add_block(capabilities)
    add_block(ai_orgs)

    for llm in llms:
        if llm in seen_subjects:
            continue
        ordered_subjects.append(llm)
        seen_subjects.add(llm)
        for m in metrics_by_llm.get(llm, []):
            if m not in seen_subjects:
                ordered_subjects.append(m)
                seen_subjects.add(m)

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

    print(f"Wrote HF leaderboard TTL to {out_path} (backup at {BACKUP})")

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
