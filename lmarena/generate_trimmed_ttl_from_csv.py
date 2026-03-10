from pathlib import Path
import csv
import re
from collections import OrderedDict, defaultdict
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "lmarena.csv"
OUT_TTL = ROOT / "lmarena.ttl"
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

# Orgs known to serve their own models (including open-weight ones) via a hosted API.
# If a model's org is in this set, it gets provisioning triples regardless of license.
KNOWN_API_PROVIDERS = {
    "openai", "anthropic", "google", "mistral", "mistralai",
    "cohere", "perplexity ai", "amazon", "alibaba",
    "01 ai", "deepseek ai", "deepseek", "zhipu ai",
    "xai", "nvidia", "ibm", "ai21 labs",
    "meta",  # Meta serves Llama via llama-api
}

def is_provisioned(row: dict) -> bool:
    """
    Determine if a model is API-provisioned by its manufacturer.
    Rules:
      1. Proprietary license → always provisioned (no other way to use it)
      2. Org is a known API provider → provisioned (they serve it via their API)
      3. Otherwise → not provisioned (download-only)
    """
    lic = (row.get("license") or "").strip().lower()
    org = (row.get("org") or "").strip().lower()

    # Rule 1: proprietary models are always served via API
    if lic == "proprietary":
        return True

    # Rule 2: known API providers serve their own models
    if org in KNOWN_API_PROVIDERS:
        return True

    return False

# _________ HELPERS __________

def key_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return (s or "unnamed").lower()

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

# NEW: preserve case from rdfs:label when producing a safe local name
def local_from_label(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "Unnamed"

def local_name(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    if not s:
        return "Unnamed"
    parts = s.split("_")
    parts_cap = []
    for p in parts:
        if p.isdigit():
            parts_cap.append(p)
        else:
            # preserve "AI" as uppercase when the token is "ai"
            if p.lower() == "ai":
                parts_cap.append("AI")
            else:
                parts_cap.append(p[:1].upper() + p[1:].lower() if p else p)
    return "_".join(parts_cap)

def read_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline='', encoding='utf-8') as fh:
        sample = fh.read(4096)
        fh.seek(0)
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            id_col = next((h for h in reader.fieldnames if h.lower() in ("key","id","model","identifier")), reader.fieldnames[0])
            label_col = next((h for h in reader.fieldnames if h.lower() in ("model","model_name","display_name","name")), reader.fieldnames[1] if len(reader.fieldnames)>1 else id_col)
            mt_col = next((h for h in reader.fieldnames if "mt-bench" in h.lower() or "mt_bench" in h.lower()), None)
            mmlu_col = next((h for h in reader.fieldnames if h.lower() in ("mmlu","mmlu_score")), None)
            cutoff_col = next((h for h in reader.fieldnames if "knowledge" in h.lower() and "cutoff" in h.lower()), None)
            license_col = next((h for h in reader.fieldnames if "license" in h.lower()), None)
            org_col = next((h for h in reader.fieldnames if h.lower() in ("organization","org","owner","manufacturer")), None)
            link_col = next((h for h in reader.fieldnames if h.lower() in ("link","url","identifier")), None)

            fh.seek(0)
            reader = csv.DictReader(fh)
            for r in reader:
                idv = (r.get(id_col) or "").strip()
                label = (r.get(label_col) or "").strip() or idv
                mt = (r.get(mt_col) or "").strip() if mt_col else ""
                mmlu = (r.get(mmlu_col) or "").strip() if mmlu_col else ""
                cutoff = (r.get(cutoff_col) or "").strip() if cutoff_col else ""
                lic = (r.get(license_col) or "").strip() if license_col else ""
                org = (r.get(org_col) or "").strip() if org_col else ""
                link = (r.get(link_col) or "").strip() if link_col else ""
                rows.append({
                    "id": idv or label,
                    "label": label,
                    "mtbench": mt,
                    "mmlu": mmlu,
                    "cutoff": cutoff,
                    "license": lic,
                    "org": org,
                    "link": link
                })
        else:
            fh.seek(0)
            rdr = csv.reader(fh)
            for cols in rdr:
                if not cols:
                    continue
                idv = cols[0].strip()
                label = cols[1].strip() if len(cols) > 1 else idv
                mt = cols[2].strip() if len(cols) > 2 else ""
                mmlu = cols[3].strip() if len(cols) > 3 else ""
                cutoff = cols[4].strip() if len(cols) > 4 else ""
                lic = cols[5].strip() if len(cols) > 5 else ""
                org = cols[6].strip() if len(cols) > 6 else ""
                link = cols[7].strip() if len(cols) > 7 else ""
                rows.append({
                    "id": idv or label,
                    "label": label,
                    "mtbench": mt,
                    "mmlu": mmlu,
                    "cutoff": cutoff,
                    "license": lic,
                    "org": org,
                    "link": link
                })
    return rows

def normalize_license_local(lic: str) -> str:
    if not lic:
        return ""
    l = lic.strip()
    # Normalize all variants of "Apache 2.0" / "Apache-2.0" to one canonical form
    if re.match(r'(?i)^apache[ _\-.]?2[ _\-.]?0$', l):
        return local_from_label("Apache 2.0")
    # For all other licenses, derive the URI local directly from the label text
    # preserving the original capitalization
    return local_from_label(l)

def build_graph(rows) -> tuple[Graph, dict[URIRef, list[URIRef]]]:
    g = Graph()
    g.bind("enki", ENKI); g.bind("contextuel", LM); g.bind("dct", DCT)
    g.bind("rdfs", RDFS); g.bind("xsd", XSD)

    # Metrics-by-LLM map — populated as metrics are created
    metrics_by_llm: dict[URIRef, list[URIRef]] = defaultdict(list)

    lm_arena = URIRef(str(LM) + local_from_label("LM Arena"))
    g.add((lm_arena, RDF.type, ENKI.AIOrganization))
    g.add((lm_arena, RDFS.label, Literal("LM Arena")))
    g.add((lm_arena, DCT.identifier, Literal("https://arena.ai/")))
    g.add((lm_arena, DCT.description, Literal("LM Arena is crowdsourced platform for comparing LLMs across various benchmarks.")))

    # Collect provisioning data as we iterate
    # provider_org_label → [(model_uri, model_local, model_label)]
    provider_provisionings = defaultdict(list)

    for r in rows:
        local = local_from_label(r["label"] or r["id"])
        model_uri = URIRef(str(LM) + local)
        model_label = r["label"] or r["id"]
        g.add((model_uri, RDF.type, ENKI.LLM))
        g.add((model_uri, RDFS.label, Literal(r["label"])))

        if r["link"]:
            try:
                g.add((model_uri, DCT.identifier, Literal(r["link"])))
            except Exception:
                pass

        # Parameters (extracted from model name)
        params_b = extract_params_billion(model_label)
        if params_b is not None:
            g.add((model_uri, ENKI.hasTotalParameterCount,
                   Literal(int(params_b * 1e9), datatype=XSD.nonNegativeInteger)))

        if r["license"] and r["license"].strip() and r["license"].strip() != "-":
            lic_local = normalize_license_local(r["license"])
            lic_uri = URIRef(str(LM) + lic_local)
            g.add((lic_uri, RDF.type, DCT.License))
            g.add((lic_uri, RDFS.label, Literal(r["license"])))
            g.add((model_uri, DCT.license, lic_uri))

        if r["cutoff"] and r["cutoff"].strip() and r["cutoff"].strip() != "-":
            g.add((model_uri, ENKI.hasKnowledgeCutoff, Literal(r["cutoff"])))

        if r["org"] and r["org"].strip() and r["org"].strip() != "-":
            org_label = r["org"].strip()
            org_local = local_from_label(org_label)
            org_uri = URIRef(str(LM) + org_local)
            g.add((org_uri, RDF.type, ENKI.LLMManufacturer))
            g.add((org_uri, RDFS.label, Literal(org_label)))
            g.add((org_uri, ENKI.manufactures, model_uri))

            # Check provisioning dynamically
            if is_provisioned(r):
                provider_provisionings[org_label].append((model_uri, local, model_label))

        mt = r["mtbench"]
        if mt and mt.strip() and mt.strip() != "-":
            try:
                val = float(mt)
                metric_uri = URIRef(str(LM) + f"{local}_MT_Bench")
                g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
                g.add((metric_uri, ENKI.metricValue, Literal(val)))
                g.add((metric_uri, ENKI.metricAssignee, model_uri))
                g.add((metric_uri, ENKI.metricAssigner, lm_arena))
                g.add((metric_uri, ENKI.metricUnit, Literal("MT-Bench")))
                g.add((metric_uri, ENKI.metricAssesses, LM.Quality))
                metrics_by_llm[model_uri].append(metric_uri)
            except Exception:
                pass

        mmlu = r["mmlu"]
        if mmlu and mmlu.strip() and mmlu.strip() != "-":
            try:
                val = float(mmlu)
                metric_uri = URIRef(str(LM) + f"{local}_MMLU")
                g.add((metric_uri, RDF.type, ENKI.EvaluatedMetric))
                g.add((metric_uri, ENKI.metricValue, Literal(val)))
                g.add((metric_uri, ENKI.metricAssignee, model_uri))
                g.add((metric_uri, ENKI.metricAssigner, lm_arena))
                g.add((metric_uri, ENKI.metricUnit, Literal("MMLU")))
                g.add((metric_uri, ENKI.metricAssesses, LM.Quality))
                metrics_by_llm[model_uri].append(metric_uri)
            except Exception:
                pass

    # Add LLMProvider + LLMProvisioning triples
    for org_label, models in provider_provisionings.items():
        org_local = local_from_label(org_label)
        org_uri = URIRef(str(LM) + org_local)

        # Add LLMProvider type (URI already exists as LLMManufacturer)
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

    # Sort each model's metrics for deterministic output
    for model_uri in metrics_by_llm:
        metrics_by_llm[model_uri] = sorted(metrics_by_llm[model_uri], key=lambda s: str(s).lower())

    return g, metrics_by_llm

def add_arena_evaluations(g, rows):
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

        has_mt = mt_str and mt_str != "-"
        has_mmlu = mmlu_str and mmlu_str != "-"

        if has_mt:
            metric_uri = URIRef(str(LM) + f"{local}_MT_Bench")
            g.add((e_mt, ENKI.evaluee, model_uri))
            g.add((e_mt, ENKI.metricEvaluated, metric_uri))

        if has_mmlu:
            metric_uri = URIRef(str(LM) + f"{local}_MMLU")
            g.add((e_mmlu, ENKI.evaluee, model_uri))
            g.add((e_mmlu, ENKI.metricEvaluated, metric_uri))

def reorder_and_serialize(g: Graph, out_path: Path, metrics_by_llm: dict[URIRef, list[URIRef]]):

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
            if o.datatype in (XSD.integer, XSD.nonNegativeInteger):
                return str(int(o))
            if o.datatype in (XSD.float, XSD.double):
                return str(float(o))
            if o.datatype == XSD.decimal:
                return str(o)
            escaped = str(o).replace("\\", "\\\\").replace('"', '\\"')
            if o.language:
                return f"\"{escaped}\"@{o.language}"
            if o.datatype:
                return f"\"{escaped}\"^^{curie(o.datatype)}"
            return f"\"{escaped}\""
        return f"\"{o}\""

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
    # Manufacturers/Providers (same URIs when dual-typed)
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

    # 1. Licenses
    add_block(licenses)

    # 2. Manufacturers/Providers, each followed by their provisionings
    for mfr in manufacturers:
        if mfr not in seen_subjects:
            ordered_subjects.append(mfr)
            seen_subjects.add(mfr)
        for p in provisionings_by_org.get(mfr, []):
            if p not in seen_subjects:
                ordered_subjects.append(p)
                seen_subjects.add(p)

    # 3. Any remaining provisionings
    add_block(provisionings)

    # 4. Evaluations
    add_block(evaluations)

    # 5. AI Orgs
    add_block(ai_orgs)

    # 6. LLMs, each followed by their metrics
    for llm in llms:
        if llm in seen_subjects:
            continue
        ordered_subjects.append(llm)
        seen_subjects.add(llm)
        for m in metrics_by_llm.get(llm, []):
            if m not in seen_subjects:
                ordered_subjects.append(m)
                seen_subjects.add(m)

    # 7. Anything remaining
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

    print("Wrote trimmed TTL to", out_path, "(backup at", BACKUP, ")")

def main():
    rows = read_rows(SOURCE_CSV)
    if not rows:
        print("No source CSV found at", SOURCE_CSV, "or file is empty. Exiting.")
        return
    g, metrics_by_llm = build_graph(rows)
    add_arena_evaluations(g, rows)
    reorder_and_serialize(g, OUT_TTL, metrics_by_llm)

if __name__ == "__main__":
    main()