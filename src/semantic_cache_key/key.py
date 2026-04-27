"""Compute a deterministic, semantic cache key for an LLM request.

Algorithm (matches the JS sibling exactly so cache buckets are shared):

1. Build a normalized record::

       {
         "model":   request.model,                 # raw, no normalization
         "system":  normalize_text(system),        # case-fold + collapse ws
         "prompt":  normalize_text(prompt),        # case-fold + collapse ws
         "tools":   stable_value(tools list),      # sort tool names + nested
         "context": stable_value([normalized...]), # retrieval context
         "version": options.version or "v1",
       }

2. Recursively sort all object keys so JSON serialization is canonical.
3. ``json.dumps(record, separators=(",", ":"), sort_keys=True)``.
4. ``hashlib.sha256(json.encode("utf-8")).hexdigest()``.
5. Optionally truncate (``length=N``) for short keys, default 16.

Properties:

* Invariant to: whitespace differences, casing, tool object key order,
  retrieval-context entry order *within* a list (the entries themselves
  are sorted), JSON key insertion order.
* Sensitive to: model identifier, the actual tool *names* / arities,
  retrieval-context contents, system prompt content, ``version`` bump.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Optional, Sequence, Union


_WS = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Lowercase, collapse whitespace runs to single spaces, trim ends.

    ``None`` becomes ``""``. Non-strings are stringified via ``str(...)``.
    Mirrors the JS sibling's ``normalizeText`` exactly so the same input
    produces the same canonical form across both runtimes.
    """
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    return _WS.sub(" ", s.lower()).strip()


def stable_value(value: Any) -> Any:
    """Return ``value`` with all dict keys recursively sorted.

    * Lists / tuples: each element is recursed; *order is preserved* (callers
      decide whether ordering is significant -- e.g. tool lists are sorted
      explicitly in :func:`key`, retrieval entries usually aren't).
    * Mappings: returned as a new dict with keys sorted lexicographically;
      values recursed.
    * Everything else: returned as-is.

    Matches the JS sibling's ``stable`` helper.
    """
    if isinstance(value, Mapping):
        return {k: stable_value(value[k]) for k in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [stable_value(v) for v in value]
    return value


def _normalize_tools(tools: Optional[Sequence[Any]]) -> list:
    """Canonicalize a tool list.

    * Each tool is :func:`stable_value`'d so internal key order doesn't matter.
    * The *list* is sorted by tool ``name`` (or by JSON form when a tool has
      no name) so callers can pass tools in any order.
    """
    if not tools:
        return []
    items = [stable_value(t) for t in tools]

    def _sort_key(t: Any) -> str:
        if isinstance(t, Mapping):
            for n in ("name", "function", "id"):
                v = t.get(n)
                if isinstance(v, str):
                    return v
        return json.dumps(t, sort_keys=True, ensure_ascii=False)

    items.sort(key=_sort_key)
    return items


def _normalize_context(items: Optional[Iterable[Any]]) -> list:
    """Canonicalize a retrieval-context list.

    * Strings are :func:`normalize_text`'d.
    * Mappings are :func:`stable_value`'d (and any ``id`` is left as-is so
      callers can sort by it externally).
    * Order is preserved; callers who want order-invariance can sort first.
    """
    if not items:
        return []
    out: list = []
    for x in items:
        if isinstance(x, str):
            out.append(normalize_text(x))
        else:
            out.append(stable_value(x))
    return out


def semantic_cache_key(
    request: Mapping[str, Any],
    *,
    version: str = "v1",
) -> str:
    """JS-compatible signature: take a single ``request`` dict.

    Returns the full 64-character hex digest; matches the JS sibling exactly.
    """
    return key(
        prompt=request.get("prompt", request.get("input", "")),
        model=request.get("model", ""),
        system=request.get("system"),
        tools=request.get("tools"),
        retrieval_context=request.get("context") or request.get("retrieval_context"),
        version=version,
        length=64,
    )


def key(
    prompt: Union[str, Any] = "",
    model: str = "",
    *,
    tools: Optional[Sequence[Any]] = None,
    retrieval_context: Optional[Iterable[Any]] = None,
    system: Optional[str] = None,
    version: str = "v1",
    length: int = 16,
) -> str:
    """Compute a stable cache key for an LLM request.

    Args:
        prompt: User prompt / input. Whitespace + casing normalized.
        model: Model identifier. **Not** normalized -- swapping models
            should change the key.
        tools: Iterable of tool definitions. Order-insensitive.
        retrieval_context: Iterable of retrieved chunks (strings or dicts).
            Strings are normalized; order is preserved (sort externally if
            you want order-invariance across calls).
        system: Optional system prompt; normalized like ``prompt``.
        version: Version tag baked into the key. Bump to invalidate caches.
        length: Hex chars to keep from the SHA-256 digest. ``64`` returns
            the full hash (matches the JS sibling). Default 16 is the
            "short key" form recommended for in-memory caches.

    Returns:
        Lowercase hex string of length ``length``.
    """
    if length < 4 or length > 64:
        raise ValueError("length must be between 4 and 64")

    record = {
        "model": model,
        "system": normalize_text(system),
        "prompt": normalize_text(prompt),
        "tools": _normalize_tools(tools),
        "context": _normalize_context(retrieval_context),
        "version": version,
    }
    canonical = json.dumps(
        stable_value(record),
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]
