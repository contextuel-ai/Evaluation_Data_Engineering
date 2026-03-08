"""
Compare provisioned-model overlap across the three evaluation datasets:
  - HuggingFace Open LLM Leaderboard  (hf_leaderboard.ttl)
  - Artificial Analysis               (aa_models.ttl)
  - LMArena / Chatbot Arena           (lmarena.ttl)

Only models that appear in at least one LLMProvisioning triple are considered.

Matching strategy
-----------------
Model labels differ across sources (e.g. AA uses spaces, HF uses hyphens,
word order varies).  We build a *canonical key* per label:
  1. Lower-case.
  2. Strip parenthetical suffixes — "(Reasoning)", "(V1)", "(Jun '24)", etc.
  3. Replace all separators (.-_/ and whitespace) with a single space.
  4. Remove filler words: "instruct", "chat", "hf", "preview", "v0.1", etc.
  5. Sort the remaining tokens alphabetically so word order doesn't matter.
  6. Rejoin with "-".

This is intentionally aggressive: it will *over-match* slightly, so the
output includes the original labels side-by-side for manual verification.
"""

from pathlib import Path
import re
from collections import defaultdict
from rdflib import Graph, Namespace, RDF, RDFS

ROOT = Path(__file__).resolve().parent

ENKI = Namespace("https://www.contextuel.ai/spec/Enki/")
C = Namespace("https://www.contextuel.ai/spec/Contextuel/")
LM = Namespace("https://www.contextuel.ai/spec/Enki/lmarena_triples/")

HF_TTL = ROOT / "hf_leaderboard" / "hf_leaderboard.ttl"
AA_TTL = ROOT / "artifical_analysis_models" / "aa_models.ttl"
LM_TTL = ROOT / "lmarena" / "lmarena.ttl"

# _________ EXTRACT PROVISIONED MODELS __________

def provisioned_labels(ttl_path: Path) -> set[str]:
    """Return the set of rdfs:label strings for models referenced by
    at least one enki:LLMProvisioning via enki:provides."""
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    labels: set[str] = set()
    for prov in g.subjects(RDF.type, ENKI.LLMProvisioning):
        for model in g.objects(prov, ENKI.provides):
            for label in g.objects(model, RDFS.label):
                labels.add(str(label))
    return labels


# _________ CANONICAL KEY __________

# Tokens that carry no discriminating information for model identity.
_FILLER = {
    "hf", "sft", "dpo",
    "rlhf", "gptq", "awq", "gguf", "fp16", "bf16", "int4", "int8",
    "meta",  # HF uses "Meta-Llama-3-…", AA uses "Llama 3 …"
}

# Patterns to strip before tokenising.
_PAREN_RE = re.compile(r"\s*\(.*?\)")                       # (Reasoning), (V1) …
_DATE_RE = re.compile(                                        # Jun '24, Oct 24 …
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
    r"\s*'?\d{2,4}\b", re.IGNORECASE)


def canonical_key(label: str) -> str:
    """Build an order-independent canonical key from a model label."""
    s = label.lower().strip()
    s = _PAREN_RE.sub("", s)
    s = _DATE_RE.sub("", s)
    # Normalise separators to spaces
    s = re.sub(r"[.\-_/]+", " ", s)
    # Tokenise, drop filler, sort
    tokens = [t for t in s.split() if t and t not in _FILLER]
    tokens.sort()
    return "-".join(tokens)


# _________ OVERLAP COMPUTATION __________

def build_key_map(labels: set[str]) -> dict[str, list[str]]:
    """Map canonical key → list of original labels."""
    km: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        km[canonical_key(label)].append(label)
    return dict(km)


def compute_overlaps(
    hf_labels: set[str],
    aa_labels: set[str],
    lm_labels: set[str],
):
    hf_km = build_key_map(hf_labels)
    aa_km = build_key_map(aa_labels)
    lm_km = build_key_map(lm_labels)

    hf_keys = set(hf_km)
    aa_keys = set(aa_km)
    lm_keys = set(lm_km)

    all_three = hf_keys & aa_keys & lm_keys
    hf_aa_only = (hf_keys & aa_keys) - lm_keys
    hf_lm_only = (hf_keys & lm_keys) - aa_keys
    aa_lm_only = (aa_keys & lm_keys) - hf_keys

    hf_only = hf_keys - aa_keys - lm_keys
    aa_only = aa_keys - hf_keys - lm_keys
    lm_only = lm_keys - hf_keys - aa_keys

    # ── Summary ──────────────────────────────────────
    print("=" * 70)
    print("PROVISIONED-MODEL OVERLAP (canonical-key matching)")
    print("=" * 70)
    print(f"  HF provisioned models:  {len(hf_labels):>5}")
    print(f"  AA provisioned models:  {len(aa_labels):>5}")
    print(f"  LM provisioned models:  {len(lm_labels):>5}")
    print()
    print(f"  Unique canonical keys — HF: {len(hf_keys)}  AA: {len(aa_keys)}  LM: {len(lm_keys)}")
    print()
    print("  Overlap counts:")
    print(f"    All three (HF ∩ AA ∩ LM):  {len(all_three):>4}")
    print(f"    HF ∩ AA only:              {len(hf_aa_only):>4}")
    print(f"    HF ∩ LM only:              {len(hf_lm_only):>4}")
    print(f"    AA ∩ LM only:              {len(aa_lm_only):>4}")
    print(f"    HF only:                   {len(hf_only):>4}")
    print(f"    AA only:                   {len(aa_only):>4}")
    print(f"    LM only:                   {len(lm_only):>4}")
    print()

    # ── Detail: all three ────────────────────────────
    if all_three:
        print("-" * 70)
        print(f"ALL THREE ({len(all_three)} models)")
        print("-" * 70)
        for key in sorted(all_three):
            print(f"  key: {key}")
            print(f"    HF: {hf_km[key]}")
            print(f"    AA: {aa_km[key]}")
            print(f"    LM: {lm_km[key]}")
        print()

    # ── Detail: pairwise ─────────────────────────────
    for label, keys, km_a, km_b, name_a, name_b in [
        ("HF ∩ AA only", hf_aa_only, hf_km, aa_km, "HF", "AA"),
        ("HF ∩ LM only", hf_lm_only, hf_km, lm_km, "HF", "LM"),
        ("AA ∩ LM only", aa_lm_only, aa_km, lm_km, "AA", "LM"),
    ]:
        if not keys:
            continue
        print("-" * 70)
        print(f"{label} ({len(keys)} models)")
        print("-" * 70)
        for key in sorted(keys):
            print(f"  key: {key}")
            print(f"    {name_a}: {km_a[key]}")
            print(f"    {name_b}: {km_b[key]}")
        print()


# _________ MAIN __________

def main():
    print("Loading TTL files …")
    hf_labels = provisioned_labels(HF_TTL)
    aa_labels = provisioned_labels(AA_TTL)
    lm_labels = provisioned_labels(LM_TTL)
    print(f"  HF: {len(hf_labels)}  AA: {len(aa_labels)}  LM: {len(lm_labels)}")
    print()

    compute_overlaps(hf_labels, aa_labels, lm_labels)


if __name__ == "__main__":
    main()
