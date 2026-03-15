# Evaluation Data Engineering

Compiles open-source LLM evaluation data from multiple sources into Turtle (`.ttl`) RDF files for the [Contextuel.ai](https://contextuel.ai/) knowledge graph, using the **Enki ontology**.

## Data Sources

| Directory | Source | Pipeline |
|---|---|---|
| `hf_leaderboard/` | [HuggingFace Open LLM Leaderboard v2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) | CSV → TTL |
| `lmarena/` | [LM Arena (Chatbot Arena)](https://lmarena.ai/) | CSV → TTL (trimmed + augmented) |
| `artificial_analysis_models/` | [Artificial Analysis](https://artificialanalysis.ai/) | API → CSV → TTL |

## Provider Model Catalogues

| Directory | Provider | Output |
|---|---|---|
| `Mistral/` | [Mistral AI](https://mistral.ai/) | API + docs scrape → JSON |
| `OpenAI/` | [OpenAI](https://openai.com/) | API + docs scrape → JSON |
| `Fireworks_AI/` | [Fireworks AI](https://fireworks.ai/) | API → JSON |
| `Together_AI/` | [Together AI](https://www.together.ai/) | API + docs scrape → JSON |

## Supporting Files

| File / Directory | Purpose |
|---|---|
| `data_compare.py` | Computes provisioned-model overlap across HF, AA, and LMArena TTL files |
| `proposed_capabilities.ttl` | Proposed new `enki:Capability` classes with subsumption hierarchy |
| `enki/` | Enki ontology backend & frontend (submodule) |
| `turtlelint/` | Turtle verification tool — validates TTL against the ontology |

## Namespaces

```turtle
@base <https://www.contextuel.ai/spec/Contextuel/> .
@prefix enki:       <https://www.contextuel.ai/spec/Enki/> .
@prefix contextuel: <https://www.contextuel.ai/spec/Contextuel/> .
@prefix dct:        <http://purl.org/dc/terms/> .
```

## Technology Stack

- **Python 3.12+** (scripts target 3.12 / 3.13)
- **rdflib** — RDF graph construction
- **pandas** — CSV handling (HF pipeline)
- **requests / BeautifulSoup** — API and web scraping (provider catalogues)
- **turtlelint** — ontology-conformance validation
