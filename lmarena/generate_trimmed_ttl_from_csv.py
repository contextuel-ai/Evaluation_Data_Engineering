from pathlib import Path
import csv
import re
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "lmarena.csv"
OUT_TTL = ROOT / "lmarena_trimmed.ttl"
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

def key_slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[.\-\/\s]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return (s or "unnamed").lower()

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

def build_graph(rows):
    g = Graph()
    g.bind("enki", ENKI); g.bind("contextuel", LM); g.bind("dct", DCT)
    g.bind("rdfs", RDFS); g.bind("xsd", XSD)

    lm_arena = URIRef(str(LM) + local_from_label("LM Arena"))
    eval_mt = URIRef(str(LM) + local_from_label("LM Arena MT-Bench"))
    eval_mmlu = URIRef(str(LM) + local_from_label("LM Arena MMLU"))

    lm_arena = URIRef(str(LM) + local_from_label("LM Arena"))
    g.add((lm_arena, RDF.type, ENKI.AIOrganization))
    g.add((lm_arena, RDFS.label, Literal("LM Arena")))
    g.add((lm_arena, DCT.identifier, Literal("https://arena.ai/")))
    g.add((lm_arena, DCT.description, Literal("LM Arena is crowdsourced platform for comparing LLMs across various benchmarks.")))

    for r in rows:
        # use the label to create the URI local so capitalization matches the rdfs:label
        local = local_from_label(r["label"] or r["id"])
        model_uri = URIRef(str(LM) + local)
        g.add((model_uri, RDF.type, ENKI.LLM))
        g.add((model_uri, RDFS.label, Literal(r["label"])))

        if r["link"]:
            try:
                g.add((model_uri, DCT.identifier, Literal(r["link"])))
            except Exception:
                pass

        if r["license"] and r["license"].strip() and r["license"].strip() != "-":
            lic_local = normalize_license_local(r["license"])
            lic_uri = URIRef(str(LM) + lic_local)
            g.add((lic_uri, RDF.type, DCT.License))
            g.add((lic_uri, RDFS.label, Literal(r["license"])))
            g.add((model_uri, DCT.license, lic_uri))

        if r["cutoff"] and r["cutoff"].strip() and r["cutoff"].strip() != "-":
            g.add((model_uri, ENKI.hasKnowledgeCutoff, Literal(r["cutoff"])))

        if r["org"] and r["org"].strip() and r["org"].strip() != "-":
            org_local = local_from_label(r["org"])
            org_uri = URIRef(str(LM) + org_local)
            g.add((org_uri, RDF.type, ENKI.LLMManufacturer))
            g.add((org_uri, RDFS.label, Literal(r["org"])))
            g.add((org_uri, ENKI.manufactures, model_uri))

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
            except Exception:
                pass

    return g

def add_arena_evaluations(g, rows):
    lm_arena = URIRef(str(LM) + local_from_label("LM Arena"))

    # create evaluation URIs from rdfs:label so local parts preserve label capitalization
    e_mt = URIRef(str(LM) + "LM_Arena_MT_Bench")
    e_mmlu = URIRef(str(LM) + "LM_Arena_MMLU")
    g.add((e_mt, RDF.type, ENKI.Evaluation))
    g.add((e_mt, RDFS.label, Literal("LM Arena MT-Bench")))
    g.add((e_mmlu, RDF.type, ENKI.Evaluation))
    g.add((e_mmlu, RDFS.label, Literal("LM Arena MMLU")))

    g.add((e_mt, ENKI.evaluator, lm_arena))
    g.add((e_mmlu, ENKI.evaluator, lm_arena))

            # --- MMLU Capabilities ---
    c1 = URIRef(str(LM) + local_from_label("GeneralKnowledgeReasoning"))
    c2 = URIRef(str(LM) + local_from_label("MultiStepReasoning"))
    c3 = URIRef(str(LM) + local_from_label("FactualRecall"))
    c4 = URIRef(str(LM) + local_from_label("AcademicDomainKnowledge"))

    g.add((c1, RDF.type, ENKI.Capability))
    g.add((c1, RDFS.label, Literal("General Knowledge Reasoning")))
    g.add((c1, DCT.description, Literal(
        "The ability of an LLM to apply broad academic knowledge across multiple disciplines to correctly answer domain-specific questions."
    )))

    g.add((c2, RDF.type, ENKI.Capability))
    g.add((c2, RDFS.label, Literal("Multi-Step Reasoning")))
    g.add((c2, DCT.description, Literal(
        "The ability of an LLM to perform sequential logical operations to arrive at a correct conclusion."
    )))

    g.add((c3, RDF.type, ENKI.Capability))
    g.add((c3, RDFS.label, Literal("Factual Recall")))
    g.add((c3, DCT.description, Literal(
        "The ability of an LLM to accurately retrieve and reproduce learned factual information from its training data."
    )))

    g.add((c4, RDF.type, ENKI.Capability))
    g.add((c4, RDFS.label, Literal("Academic Domain Knowledge")))
    g.add((c4, DCT.description, Literal(
        "The ability of an LLM to demonstrate understanding of specialized subject matter such as law, medicine, physics, or history."
    )))

    # --- MT-Bench Capabilities ---
    c5 = URIRef(str(LM) + local_from_label("InstructionFollowing"))
    c6 = URIRef(str(LM) + local_from_label("ConversationalReasoning"))
    c7 = URIRef(str(LM) + local_from_label("MultiTurnDialogue"))
    c8 = URIRef(str(LM) + local_from_label("ContextRetention"))
    c9 = URIRef(str(LM) + local_from_label("ResponseHelpfulness"))

    g.add((c5, RDF.type, ENKI.Capability))
    g.add((c5, RDFS.label, Literal("Instruction Following")))
    g.add((c5, DCT.description, Literal(
        "The ability of an LLM to understand and correctly execute user-provided instructions."
    )))

    g.add((c6, RDF.type, ENKI.Capability))
    g.add((c6, RDFS.label, Literal("Conversational Reasoning")))
    g.add((c6, DCT.description, Literal(
        "The ability of an LLM to maintain logical consistency and coherence throughout a dialogue."
    )))

    g.add((c7, RDF.type, ENKI.Capability))
    g.add((c7, RDFS.label, Literal("Multi-Turn Dialogue")))
    g.add((c7, DCT.description, Literal(
        "The ability of an LLM to sustain contextually appropriate responses across multiple conversational exchanges."
    )))

    g.add((c8, RDF.type, ENKI.Capability))
    g.add((c8, RDFS.label, Literal("Context Retention")))
    g.add((c8, DCT.description, Literal(
        "The ability of an LLM to remember and appropriately use prior information within an ongoing interaction."
    )))

    g.add((c9, RDF.type, ENKI.Capability))
    g.add((c9, RDFS.label, Literal("Response Helpfulness")))
    g.add((c9, DCT.description, Literal(
        "The ability of an LLM to generate responses that are useful, relevant, and aligned with user intent."
    )))

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
            local = local_from_label(r["label"] or r["id"])
            metric_uri = URIRef(str(LM) + f"{local}_MT_Bench")
            g.add((e_mt, ENKI.evaluee, model_uri))
            g.add((e_mt, ENKI.metricEvaluated, metric_uri))

        if has_mmlu:
            local = local_from_label(r["label"] or r["id"])
            metric_uri = URIRef(str(LM) + f"{local}_MMLU")
            g.add((e_mmlu, ENKI.evaluee, model_uri))
            g.add((e_mmlu, ENKI.metricEvaluated, metric_uri))

def serialize_graph(g: Graph, out_path: Path):
    ttl = g.serialize(format="turtle")
    final = HEADER_BLOCK + "\n" + "\n".join([ln for ln in ttl.splitlines() if not ln.startswith("@prefix") and not ln.startswith("@base")]) + "\n"
    if out_path.exists():
        BACKUP.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
    out_path.write_text(final, encoding="utf-8")
    print("Wrote trimmed TTL to", out_path, "(backup at", BACKUP, ")")

def reorder_and_serialize(g: Graph, out_path: Path):
    """
    Write TTL in this order:
      1) Licenses
      2) LLMManufacturers
      3) Evaluations
      4) Capabilities
      5) LM_Arena AIOrganization (and any other AIOrganization subjects, with LM_Arena first if present)
      6) Each LLM immediately followed by its corresponding EvaluatedMetrics
      7) Remaining subjects (anything not covered above)
    """
    from collections import OrderedDict, defaultdict
    from urllib.parse import urldefrag

    # ---------- helpers ----------
    def local_name(u: URIRef) -> str:
        """Best-effort local name extraction for grouping (keeps underscores, etc.)."""
        s = str(u)
        s, _frag = urldefrag(s)
        if "#" in str(u):
            return str(u).split("#")[-1]
        return s.rstrip("/").split("/")[-1]

    # Prefix map ordered longest-first so contextuel: matches before enki:
    PREFIX_MAP = OrderedDict([
        (str(LM),   "contextuel:"),  # LM is your contextuel namespace in your script
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
            # preserve datatype-aware rendering
            if o.datatype == XSD.integer:
                return str(int(o))
            if o.datatype in (XSD.float, XSD.double):
                return str(float(o))
            if o.datatype == XSD.decimal:
                return str(o)
            escaped = str(o).replace("\\", "\\\\").replace('"', '\\"')
            # keep language tags/datatype if present and non-numeric
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

            # types first (comma-separated after 'a')
            types = sorted(set(g.objects(s, RDF.type)), key=lambda o: str(o))
            if types:
                if len(types) == 1:
                    block.append(f"{curie(s)} a {curie(types[0])}")
                else:
                    type_strs = ",\n        ".join(curie(t) for t in types)
                    block.append(f"{curie(s)} a {type_strs}")
            else:
                # subjects with no rdf:type still get a header line
                block.append(f"{curie(s)}")

            # remaining predicates alphabetically
            for p in preds:
                if p == RDF.type:
                    continue
                objs = sorted(set(g.objects(s, p)), key=lambda o: str(o))
                for o in objs:
                    block.append(f"    {curie(p)} {format_obj(o)}")

            block_lines.append(" ;\n".join(block) + " .")
            block_lines.append("")
        return block_lines

    # ---------- collect typed subjects ----------
    seen_subjects = set()

    licenses = sorted(set(g.subjects(RDF.type, DCT.License)), key=lambda s: str(s).lower())
    manufacturers = sorted(set(g.subjects(RDF.type, ENKI.LLMManufacturer)), key=lambda s: str(s).lower())
    evaluations = sorted(set(g.subjects(RDF.type, ENKI.Evaluation)), key=lambda s: str(s).lower())
    capabilities = sorted(set(g.subjects(RDF.type, ENKI.Capability)), key=lambda s: str(s).lower())

    # AIOrganization: put LM_Arena first if present, then the rest
    ai_orgs_all = sorted(set(g.subjects(RDF.type, ENKI.AIOrganization)), key=lambda s: str(s).lower())
    lm_arena_uri = URIRef(str(LM) + "LM_Arena")  # assumes LM == contextuel namespace
    ai_orgs = []
    if lm_arena_uri in ai_orgs_all:
        ai_orgs.append(lm_arena_uri)
        ai_orgs.extend([s for s in ai_orgs_all if s != lm_arena_uri])
    else:
        ai_orgs = ai_orgs_all

    # ---------- LLMs + their metrics interleaving ----------
    llms = sorted(set(g.subjects(RDF.type, ENKI.LLM)), key=lambda s: str(s).lower())
    metrics = set(g.subjects(RDF.type, ENKI.EvaluatedMetric))

    # Build a fast index from metric local-name prefix -> metric subjects.
    # We’ll group metrics by "the LLM local name" inferred from metric subject local name.
    # Expected patterns:
    #   <Model>_MMLU, <Model>_MT_Bench, ... (suffix is benchmark/metric family)
    # If your model local names themselves contain underscores, this still works
    # because we match by "startswith(llm_local + '_')".
    metrics_by_llm = defaultdict(list)
    metric_locals = {m: local_name(m) for m in metrics}

    # Precompute for speed: list of (metric_subject, local_string)
    metric_items = list(metric_locals.items())

    for llm in llms:
        llm_loc = local_name(llm)
        # 1) strict: metrics whose local starts with "{llm_loc}_"
        for m, mloc in metric_items:
            if mloc.startswith(llm_loc + "_"):
                metrics_by_llm[llm].append(m)

        # 2) optional fallback: if nothing matched, try label-based matching
        if not metrics_by_llm[llm]:
            llm_labels = {str(o).lower() for o in g.objects(llm, RDFS.label)}
            if llm_labels:
                for m in metrics:
                    m_labels = {str(o).lower() for o in g.objects(m, RDFS.label)}
                    if m_labels and (llm_labels & m_labels):
                        metrics_by_llm[llm].append(m)

        # stable order for metrics under each LLM
        metrics_by_llm[llm] = sorted(set(metrics_by_llm[llm]), key=lambda s: str(s).lower())

    # ---------- build final subject ordering ----------
    ordered_subjects = []

    def add_block(subjects):
        for s in subjects:
            if s not in seen_subjects:
                ordered_subjects.append(s)
                seen_subjects.add(s)

    add_block(licenses)
    add_block(manufacturers)
    add_block(evaluations)
    add_block(capabilities)
    add_block(ai_orgs)

    # Interleave: LLM, its metrics, next LLM, its metrics...
    for llm in llms:
        if llm in seen_subjects:
            continue
        ordered_subjects.append(llm)
        seen_subjects.add(llm)
        for m in metrics_by_llm.get(llm, []):
            if m not in seen_subjects:
                ordered_subjects.append(m)
                seen_subjects.add(m)

    # ---------- remaining subjects (covers “all the triple types” you didn’t enumerate) ----------
    # This includes any: datasets, providers, benchmark entities, blank-node subjects, etc.
    remaining = []
    for s in sorted(set(g.subjects()), key=lambda s: str(s).lower()):
        if s not in seen_subjects:
            remaining.append(s)
            seen_subjects.add(s)

    # ---------- write ----------
    lines = []
    lines.extend(serialize_subjects(ordered_subjects))
    if remaining:
        lines.extend(serialize_subjects(remaining))

    # backup (assumes BACKUP + HEADER_BLOCK exist in your script; keep your prior logic)
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
    g = build_graph(rows)
    add_arena_evaluations(g, rows)
    # write once using the ordered serializer (don't call serialize_graph first)
    reorder_and_serialize(g, OUT_TTL)

if __name__ == "__main__":
    main()