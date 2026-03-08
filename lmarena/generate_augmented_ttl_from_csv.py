from pathlib import Path
import csv
import re
from collections import OrderedDict, defaultdict
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "augmented_lmarena.csv"
OUT_TTL = ROOT / "augmented_lmarena.ttl"
BACKUP = OUT_TTL.with_suffix(".ttl.bak")

ENKI = Namespace("https://www.contextuel.ai/spec/Enki/")
LM = Namespace("https://www.contextuel.ai/spec/Enki/lmarena_triples/")
DCT = Namespace("http://purl.org/dc/terms/")

HEADER_BLOCK = """@base <https://www.contextuel.ai/spec/Enki/> .
@prefix enki: <https://www.contextuel.ai/spec/Enki/> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix contextuel: <https://www.contextuel.ai/spec/Contextuel/> .

"""

# _________ PROVISIONING DETECTION __________

KNOWN_API_PROVIDERS = {
    "openai", "anthropic", "google", "mistral", "mistralai",
    "cohere", "perplexity ai", "amazon", "alibaba",
    "01 ai", "deepseek ai", "deepseek", "zhipu ai",
    "xai", "nvidia", "ibm", "ai21 labs",
    "meta",
}

def is_provisioned(row: dict) -> bool:
    lic = (row.get("license") or "").strip().lower()
    org = (row.get("org") or "").strip().lower()
    if lic == "proprietary":
        return True
    if org in KNOWN_API_PROVIDERS:
        return True
    return False

# _________ HELPERS __________

def local_from_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "Unnamed"

def normalize_license_local(lic: str) -> str:
    if not lic:
        return ""
    l = lic.strip()
    if re.match(r'(?i)^apache[ _\-.]?2[ _\-.]?0$', l):
        return local_from_label("Apache 2.0")
    return local_from_label(l)

def safe_int(val) -> int | None:
    """Parse a value to int, handling floats like '30000000000.0'."""
    if not val or str(val).strip() in ("", "-", "nan"):
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None

def safe_billions(val) -> float | None:
    """Parse a raw parameter count and convert to billions."""
    if not val or str(val).strip() in ("", "-", "nan"):
        return None
    try:
        raw = float(str(val).strip())
        return round(raw / 1e9, 2)
    except (ValueError, TypeError):
        return None

def parse_modalities(val: str) -> list[str]:
    """Parse comma or semicolon separated modalities."""
    if not val or val.strip() in ("", "-", "nan"):
        return []
    parts = re.split(r"[;,]\s*", val.strip())
    return [p.strip() for p in parts if p.strip() and p.strip() != "-"]

# _________ READ CSV __________

def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        print(f"[ERROR] CSV not found: {path}")
        return rows

    with path.open(newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            key = (r.get("key") or "").strip()
            model = (r.get("Model") or "").strip()
            label = model or key

            mt = (r.get("MT-bench (score)") or "").strip()
            mmlu = (r.get("MMLU") or "").strip()
            cutoff = (r.get("Knowledge cutoff date") or "").strip()
            lic = (r.get("License") or "").strip()
            org = (r.get("Organization") or "").strip()
            link = (r.get("Link") or "").strip()
            desc = (r.get("description") or "").strip()

            total_params = safe_billions(r.get("total_parameter_count"))
            max_input = safe_int(r.get("max_input_tokens"))

            input_modalities = parse_modalities(r.get("input_modality", ""))
            output_modalities = parse_modalities(r.get("output_modality", ""))

            rows.append({
                "id": key or label,
                "label": label,
                "mtbench": mt,
                "mmlu": mmlu,
                "cutoff": cutoff,
                "license": lic,
                "org": org,
                "link": link,
                "description": desc,
                "total_parameter_count": total_params,
                "max_input_tokens": max_input,
                "max_output_tokens": safe_int(r.get("max_output_tokens")),
                "context_window_tokens": safe_int(r.get("context_window_tokens")),
                "input_modalities": input_modalities,
                "output_modalities": output_modalities,
            })

    print(f"Loaded {len(rows)} rows from {path.name}")
    return rows

# _________ BUILD GRAPH __________

def build_graph(rows: list[dict]) -> Graph:
    g = Graph()
    g.bind("enki", ENKI)
    g.bind("contextuel", LM)
    g.bind("dct", DCT)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # LM Arena org
    lm_arena = URIRef(str(LM) + local_from_label("LM Arena"))
    g.add((lm_arena, RDF.type, ENKI.AIOrganization))
    g.add((lm_arena, RDFS.label, Literal("LM Arena")))
    g.add((lm_arena, DCT.identifier, Literal("https://arena.ai/")))
    g.add((lm_arena, DCT.description, Literal(
        "LM Arena is crowdsourced platform for comparing LLMs across various benchmarks."
    )))

    # Collect provisioning data
    provider_provisionings = defaultdict(list)

    for r in rows:
        local = local_from_label(r["label"] or r["id"])
        model_uri = URIRef(str(LM) + local)
        model_label = r["label"] or r["id"]

        g.add((model_uri, RDF.type, ENKI.LLM))
        g.add((model_uri, RDFS.label, Literal(model_label)))

        # Identifier
        if r["link"]:
            g.add((model_uri, DCT.identifier, Literal(r["link"])))

        # Description
        if r["description"] and r["description"] not in ("-", "nan"):
            g.add((model_uri, DCT.description, Literal(r["description"])))

        # License
        if r["license"] and r["license"] not in ("-", "nan"):
            lic_local = normalize_license_local(r["license"])
            lic_uri = URIRef(str(LM) + lic_local)
            g.add((lic_uri, RDF.type, DCT.License))
            g.add((lic_uri, RDFS.label, Literal(r["license"])))
            g.add((model_uri, DCT.license, lic_uri))

        # Knowledge cutoff
        if r["cutoff"] and r["cutoff"] not in ("-", "nan", "Online"):
            g.add((model_uri, ENKI.hasKnowledgeCutoff, Literal(r["cutoff"])))

        # Total parameter count (in billions)
        if r["total_parameter_count"] is not None:
            g.add((model_uri, ENKI.hasTotalParameterCount,
                   Literal(r["total_parameter_count"], datatype=XSD.double)))

        # Max context window tokens
        if r["context_window_tokens"] is not None:
            g.add((model_uri, ENKI.hasMaxContextWindowTokens,
                   Literal(r["context_window_tokens"], datatype=XSD.nonNegativeInteger)))

        # Max input tokens
        if r["max_input_tokens"] is not None:
            g.add((model_uri, ENKI.hasMaxInputTokens,
                   Literal(r["max_input_tokens"], datatype=XSD.nonNegativeInteger)))

        # Max output tokens
        if r["max_output_tokens"] is not None:
            g.add((model_uri, ENKI.hasMaxOutputTokens,
                   Literal(r["max_output_tokens"], datatype=XSD.nonNegativeInteger)))

        # Input modalities
        for mod in r["input_modalities"]:
            mod_local = local_from_label(mod)
            mod_uri = URIRef(str(LM) + mod_local)
            g.add((mod_uri, RDF.type, ENKI.Modality))
            g.add((mod_uri, RDFS.label, Literal(mod)))
            g.add((model_uri, ENKI.hasInputModality, mod_uri))

        # Output modalities
        for mod in r["output_modalities"]:
            mod_local = local_from_label(mod)
            mod_uri = URIRef(str(LM) + mod_local)
            g.add((mod_uri, RDF.type, ENKI.Modality))
            g.add((mod_uri, RDFS.label, Literal(mod)))
            g.add((model_uri, ENKI.hasOutputModality, mod_uri))

        # Manufacturer
        if r["org"] and r["org"] not in ("-", "nan"):
            org_label = r["org"].strip()
            org_local = local_from_label(org_label)
            org_uri = URIRef(str(LM) + org_local)
            g.add((org_uri, RDF.type, ENKI.LLMManufacturer))
            g.add((org_uri, RDFS.label, Literal(org_label)))
            g.add((org_uri, ENKI.manufactures, model_uri))

            # Provisioning
            if is_provisioned(r):
                provider_provisionings[org_label].append((model_uri, local, model_label))

        # MT-Bench metric
        mt = r["mtbench"]
        if mt and mt not in ("-", "nan"):
            try:
                val = float(mt)
                metric_uri = URIRef(str(LM) + f"{local}_MT_Bench")
                g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
                g.add((metric_uri, ENKI.metricValue, Literal(val)))
                g.add((metric_uri, ENKI.metricAssignee, model_uri))
                g.add((metric_uri, ENKI.metricAssigner, lm_arena))
                g.add((metric_uri, ENKI.metricUnit, Literal("MT-Bench")))
                g.add((metric_uri, ENKI.metricAssesses, LM.Quality))
            except (ValueError, TypeError):
                pass

        # MMLU metric
        mmlu = r["mmlu"]
        if mmlu and mmlu not in ("-", "nan"):
            try:
                val = float(mmlu)
                metric_uri = URIRef(str(LM) + f"{local}_MMLU")
                g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
                g.add((metric_uri, ENKI.metricValue, Literal(val)))
                g.add((metric_uri, ENKI.metricAssignee, model_uri))
                g.add((metric_uri, ENKI.metricAssigner, lm_arena))
                g.add((metric_uri, ENKI.metricUnit, Literal("MMLU")))
                g.add((metric_uri, ENKI.metricAssesses, LM.Quality))
            except (ValueError, TypeError):
                pass

    # Add LLMProvider + LLMProvisioning triples
    for org_label, models in provider_provisionings.items():
        org_local = local_from_label(org_label)
        org_uri = URIRef(str(LM) + org_local)
        g.add((org_uri, RDF.type, ENKI.LLMProvider))

        for model_uri, model_local, model_label in models:
            prov_local = f"{org_local}_Provisioning_{model_local}"
            prov_uri = URIRef(str(LM) + prov_local)
            g.add((prov_uri, RDF.type, ENKI.LLMProvisioning))
            g.add((prov_uri, RDFS.label, Literal(f"{org_label} provisioning of {model_label}")))
            g.add((prov_uri, ENKI.provider, org_uri))
            g.add((prov_uri, ENKI.provides, model_uri))

    provisioned_count = sum(len(m) for m in provider_provisionings.values())
    print(f"Provisioned models: {provisioned_count} across {len(provider_provisionings)} providers")

    return g

# _________ CAPABILITIES + EVALUATIONS __________

def add_arena_evaluations(g: Graph, rows: list[dict]):
    lm_arena = URIRef(str(LM) + local_from_label("LM Arena"))

    e_mt = URIRef(str(LM) + "LM_Arena_MT_Bench")
    e_mmlu = URIRef(str(LM) + "LM_Arena_MMLU")
    g.add((e_mt, RDF.type, ENKI.Evaluation))
    g.add((e_mt, RDFS.label, Literal("LM Arena MT-Bench")))
    g.add((e_mmlu, RDF.type, ENKI.Evaluation))
    g.add((e_mmlu, RDFS.label, Literal("LM Arena MMLU")))

    g.add((e_mt, ENKI.evaluator, lm_arena))
    g.add((e_mmlu, ENKI.evaluator, lm_arena))

    # --- Capability URIs (instances declared in proposed_capabilities.ttl) ---
    # MMLU
    c1 = URIRef(str(LM) + local_from_label("GeneralKnowledgeReasoning"))
    c2 = URIRef(str(LM) + local_from_label("MultiStepReasoning"))
    c3 = URIRef(str(LM) + local_from_label("FactualRecall"))
    c4 = URIRef(str(LM) + local_from_label("AcademicDomainKnowledge"))

    # MT-Bench
    c5 = URIRef(str(LM) + local_from_label("InstructionFollowing"))
    c6 = URIRef(str(LM) + local_from_label("ConversationalReasoning"))
    c7 = URIRef(str(LM) + local_from_label("MultiTurnDialogue"))
    c8 = URIRef(str(LM) + local_from_label("ContextRetention"))
    c9 = URIRef(str(LM) + local_from_label("ResponseHelpfulness"))

    g.add((e_mmlu, ENKI.testsCapability, c1))
    g.add((e_mmlu, ENKI.testsCapability, c2))
    g.add((e_mmlu, ENKI.testsCapability, c3))
    g.add((e_mmlu, ENKI.testsCapability, c4))
    g.add((e_mt, ENKI.testsCapability, c5))
    g.add((e_mt, ENKI.testsCapability, c6))
    g.add((e_mt, ENKI.testsCapability, c7))
    g.add((e_mt, ENKI.testsCapability, c8))
    g.add((e_mt, ENKI.testsCapability, c9))

    for r in rows:
        label = r.get("label") or r.get("id") or "Unnamed"
        local = local_from_label(label)
        model_uri = URIRef(str(LM) + local)

        mt_str = r.get("mtbench", "").strip()
        mmlu_str = r.get("mmlu", "").strip()

        if mt_str and mt_str not in ("-", "nan"):
            metric_uri = URIRef(str(LM) + f"{local}_MT_Bench")
            g.add((e_mt, ENKI.evaluee, model_uri))
            g.add((e_mt, ENKI.metricEvaluated, metric_uri))

        if mmlu_str and mmlu_str not in ("-", "nan"):
            metric_uri = URIRef(str(LM) + f"{local}_MMLU")
            g.add((e_mmlu, ENKI.evaluee, model_uri))
            g.add((e_mmlu, ENKI.metricEvaluated, metric_uri))

# _________ SERIALISATION __________

def reorder_and_serialize(g: Graph, out_path: Path):
    from urllib.parse import urldefrag

    def local_name_of(u: URIRef) -> str:
        s = str(u)
        s, _frag = urldefrag(s)
        if "#" in str(u):
            return str(u).split("#")[-1]
        return s.rstrip("/").split("/")[-1]

    PREFIX_MAP = OrderedDict([
        (str(LM),   "contextuel:"),
        (str(ENKI), "enki:"),
        (str(DCT),  "dct:"),
        (str(RDF),  "rdf:"),
        (str(RDFS), "rdfs:"),
        (str(XSD),  "xsd:"),
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
    modalities = sorted(set(g.subjects(RDF.type, ENKI.Modality)), key=lambda s: str(s).lower())
    manufacturers = sorted(set(g.subjects(RDF.type, ENKI.LLMManufacturer)), key=lambda s: str(s).lower())
    provisionings = sorted(set(g.subjects(RDF.type, ENKI.LLMProvisioning)), key=lambda s: str(s).lower())
    evaluations = sorted(set(g.subjects(RDF.type, ENKI.Evaluation)), key=lambda s: str(s).lower())

    ai_orgs_all = sorted(set(g.subjects(RDF.type, ENKI.AIOrganization)), key=lambda s: str(s).lower())
    lm_arena_uri = URIRef(str(LM) + "LM_Arena")
    ai_orgs = []
    if lm_arena_uri in ai_orgs_all:
        ai_orgs.append(lm_arena_uri)
        ai_orgs.extend([s for s in ai_orgs_all if s != lm_arena_uri])
    else:
        ai_orgs = ai_orgs_all

    llms = sorted(set(g.subjects(RDF.type, ENKI.LLM)), key=lambda s: str(s).lower())
    metrics = set(g.subjects(RDF.type, ENKI.EvaluatedMetric))

    # Group provisionings by provider org
    provisionings_by_org = defaultdict(list)
    for p in provisionings:
        for org_uri in g.objects(p, ENKI.provider):
            provisionings_by_org[org_uri].append(p)
    for org_uri in provisionings_by_org:
        provisionings_by_org[org_uri] = sorted(
            provisionings_by_org[org_uri], key=lambda s: str(s).lower()
        )

    # Group metrics by LLM
    metrics_by_llm = defaultdict(list)
    metric_locals = {m: local_name_of(m) for m in metrics}
    metric_items = list(metric_locals.items())
    llms_by_length = sorted(llms, key=lambda u: len(local_name_of(u)), reverse=True)
    claimed_metrics = set()

    for llm in llms_by_length:
        llm_loc = local_name_of(llm)
        for m, mloc in metric_items:
            if m in claimed_metrics:
                continue
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

    # 1. Licenses
    add_block(licenses)

    # 2. Modalities
    add_block(modalities)

    # 3. Manufacturers/Providers, each followed by their provisionings
    for mfr in manufacturers:
        if mfr not in seen_subjects:
            ordered_subjects.append(mfr)
            seen_subjects.add(mfr)
        for p in provisionings_by_org.get(mfr, []):
            if p not in seen_subjects:
                ordered_subjects.append(p)
                seen_subjects.add(p)

    # 4. Any remaining provisionings
    add_block(provisionings)

    # 5. Evaluations
    add_block(evaluations)

    # 6. AI Orgs
    add_block(ai_orgs)

    # 7. LLMs, each followed by their metrics
    for llm in llms:
        if llm in seen_subjects:
            continue
        ordered_subjects.append(llm)
        seen_subjects.add(llm)
        for m in metrics_by_llm.get(llm, []):
            if m not in seen_subjects:
                ordered_subjects.append(m)
                seen_subjects.add(m)

    # 8. Anything remaining
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

    print(f"Wrote augmented TTL to {out_path} (backup at {BACKUP})")

# _________ MAIN __________

def main():
    rows = read_rows(SOURCE_CSV)
    if not rows:
        print("No rows read. Exiting.")
        return

    print(f"Building graph from {len(rows)} models...")
    g = build_graph(rows)
    add_arena_evaluations(g, rows)
    reorder_and_serialize(g, OUT_TTL)

if __name__ == "__main__":
    main()