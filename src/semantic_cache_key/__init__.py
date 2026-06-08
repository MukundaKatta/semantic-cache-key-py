"""semantic-cache-key -- stable cache keys for LLM requests.

Public surface:

    from semantic_cache_key import key, normalize_text, stable_value

* ``key(prompt, model, ...)`` -> 16-char hex by default; pass ``length=64``
  for the full sha256 digest (matches the JS sibling / ``semantic_cache_key``)
* ``normalize_text(s)``     -> lowercase + collapse whitespace + trim
* ``stable_value(v)``       -> deterministically-ordered dict / list

The canonical-form recipe matches the JS sibling so JS and Python clients
share a cache. See module docstring of :mod:`semantic_cache_key.key` for
the exact algorithm.
"""

from .key import key, normalize_text, semantic_cache_key, stable_value

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "key",
    "normalize_text",
    "semantic_cache_key",
    "stable_value",
]
