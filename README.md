# semantic-cache-key

[![PyPI](https://img.shields.io/pypi/v/semantic-cache-key.svg)](https://pypi.org/project/semantic-cache-key/)
[![Python](https://img.shields.io/pypi/pyversions/semantic-cache-key.svg)](https://pypi.org/project/semantic-cache-key/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Stable cache keys for LLM requests.** Invariant to harmless variation (whitespace, casing, dict key order); sensitive to material changes (model swap, tool list, retrieval context). Pure stdlib, zero runtime dependencies.

Python port of [@mukundakatta/semantic-cache-key](https://github.com/MukundaKatta/semantic-cache-key) -- the canonical-form algorithm matches exactly so a JS frontend and a Python backend can share the same cache.

## Install

```bash
pip install semantic-cache-key
```

## Usage

```python
from semantic_cache_key import key

cache_key = key(
    prompt="Summarize this article in three bullet points.",
    model="claude-sonnet-4-5",
    tools=[{"name": "fetch_url"}, {"name": "read_pdf"}],
    retrieval_context=["doc-id-42", "doc-id-7"],
    system="You are a careful research assistant.",
    version="v1",          # bump to invalidate every cache entry
)
# -> '8d3f7b91c2a04e6f' (16-char hex by default; pass length=64 for full sha256)
```

Two prompts that differ only in whitespace / casing produce the **same** key:

```python
key("Hello,   World!", "gpt-5") == key("hello, world!", "gpt-5")  # True
```

Swap the model, the tools, or the retrieval context and the key changes:

```python
key("p", "gpt-5") != key("p", "claude-sonnet-4-5")              # True
```

## API

| Function | Purpose |
|---|---|
| `key(prompt, model, *, tools=, retrieval_context=, system=, version="v1", length=16)` | The Pythonic short-key form. |
| `semantic_cache_key(request: dict)` | JS-compatible -- takes a single dict, returns the full 64-char hash. |
| `normalize_text(s)` | Lowercase + collapse whitespace + trim (the canonical-form primitive). |
| `stable_value(v)` | Recursively sort dict keys for deterministic JSON. |

See the JS sibling's [README](https://github.com/MukundaKatta/semantic-cache-key) for the design notes and examples in cross-runtime caches.
