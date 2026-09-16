"""
Hosted LLM backend for the category / sentiment / emotion layers.

This module is opt-in. The pipeline default is still the local
HF / fallback engines (privacy-preserving, offline-capable). When
the user explicitly selects a per-layer LLM backend, this module
handles:

  1. Prompt assembly (system prompt + few-shot examples + N rows).
  2. Batched calls with retry on 429 / 500 / timeout.
  3. JSON parsing with strict error reporting on malformed responses.
  4. PII redaction by default (regex), bypass via send_raw_text=True.

Provider selection order:
  - Azure OpenAI if AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY set.
  - Plain OpenAI if OPENAI_API_KEY set.
  - Anthropic if ANTHROPIC_API_KEY set.

We use the openai SDK (>= 1.30) for both Azure and OpenAI
(OpenAI's Python SDK is also Azure's), and the stdlib `urllib` for
Anthropic to avoid pulling the anthropic SDK. Both Azure and OpenAI
paths support `response_format={"type": "json_object"}` for guaranteed
JSON; Anthropic uses system-instruction + parsing.

Auto-scrub:
  Every comment is run through `employee_voice.scrub()` (regex backend)
  before being sent, unless `send_raw_text=True`. The CLI prints a
  loud warning if `send_raw_text=True` is used with a real provider.
"""
from __future__ import annotations
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    """Resolved LLM provider + client config."""
    name: str  # "azure", "openai", "anthropic"
    deployment: str  # model name (Azure uses deployment, OpenAI uses model id)
    api_version: Optional[str] = None  # Azure only


def detect_provider() -> Optional[Provider]:
    """Return the active provider based on env vars, or None.

    Order:
      1. Azure OpenAI: AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY
      2. OpenAI: OPENAI_API_KEY
      3. Anthropic: ANTHROPIC_API_KEY
    """
    az_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    az_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if az_endpoint and az_key:
        return Provider(
            name="azure",
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini").strip(),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview").strip(),
        )

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        return Provider(name="openai", deployment="gpt-4o-mini")

    anth_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anth_key:
        return Provider(name="anthropic", deployment="claude-haiku-3-5-20241022")

    return None


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def load_prompt(name: str, version: str = "v1") -> str:
    """Load a versioned prompt file from prompts/<name>_<version>.txt.

    Falls back to <name>_v1.txt if version not found. Raises FileNotFoundError
    if neither exists.

    Prompt files may use these placeholders, which are substituted from
    `employee_voice.config`:

      {category_list}    rendered as a bullet list of the current HR
                         taxonomy (loaded from CATEGORIES). Used by
                         prompts/category_v1.txt so the prompt can't
                         drift from the runtime taxonomy.
    """
    fname = f"{name}_{version}.txt"
    path = os.path.join(_PROMPTS_DIR, fname)
    if not os.path.exists(path):
        # Try v1 as a default fallback
        default = os.path.join(_PROMPTS_DIR, f"{name}_v1.txt")
        if os.path.exists(default):
            log.warning("prompt %s not found, falling back to %s_v1.txt", fname, name)
            return _render(open(default).read())
        raise FileNotFoundError(f"prompt file not found: {path}")
    return _render(open(path).read())


def _render(template: str) -> str:
    """Substitute documented placeholders in a prompt template."""
    if "{category_list}" in template:
        from .config import CATEGORIES
        bullets = "\n".join(f"  - {c}" for c in CATEGORIES) + "\n  - Other"
        template = template.replace("{category_list}", bullets)
    return template


# ---------------------------------------------------------------------------
# PII scrubbing
# ---------------------------------------------------------------------------

def _scrub_text(text: str) -> str:
    """Apply regex PII scrubber. Lazy import to keep cold-start fast."""
    from .api import scrub as _scrub
    out = _scrub(text, backend="regex")
    # `scrub` returns Union[str, Series]; we always pass a str.
    assert isinstance(out, str)
    return out


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def chunk(items: Sequence[str], size: int) -> Iterable[List[str]]:
    """Yield successive chunks of `size` from `items`."""
    if size <= 0:
        raise ValueError(f"chunk size must be > 0, got {size}")
    for i in range(0, len(items), size):
        yield list(items[i:i + size])


# ---------------------------------------------------------------------------
# Batched calls
# ---------------------------------------------------------------------------

def _build_user_prompt(system: str, rows: Sequence[str]) -> str:
    """Build a single user-role prompt containing the system instructions
    (instructed as 'rules') plus a JSON array of comments to classify.

    Some hosted models (notably older Azure deployments) reject the
    'developer' role or treat 'system' differently. To stay portable
    we put the system instructions inline as 'Rules' at the top of
    the user message. The model still gets the role separation via
    the explicit system message in the API call.
    """
    payload = {"comments": list(rows)}
    return (
        "Apply the rules below to each entry in the JSON array.\n\n"
        f"RULES:\n{system}\n\n"
        "INPUT:\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```\n\n"
        "OUTPUT:\nReturn a JSON object with a single key 'results' whose value is\n"
        "a JSON array (same length as INPUT.comments) of objects, one per row,\n"
        "in the same order. Each object must conform to the schema described in\n"
        "the rules (label/score, or category/secondary, etc)."
    )


def _call_openai_compatible(
    provider: Provider,
    system: str,
    rows: Sequence[str],
    max_retries: int = 4,
    timeout_s: float = 60.0,
) -> str:
    """Call Azure OpenAI or OpenAI. Returns the assistant message text.

    Retries on 429 / 5xx with exponential backoff. Raises RuntimeError on
    persistent failure.
    """
    import openai  # lazy import (this dep is optional)

    user = _build_user_prompt(system, rows)

    if provider.name == "azure":
        client = openai.AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=provider.api_version,
        )
        model_arg = provider.deployment  # Azure uses the deployment name
    else:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model_arg = provider.deployment  # model id

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_arg,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=timeout_s,
            )
            return resp.choices[0].message.content or ""
        except openai.RateLimitError as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning("rate limit (attempt %d), sleeping %ds", attempt + 1, wait)
            time.sleep(wait)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning("API transient (attempt %d): %s", attempt + 1, e)
            time.sleep(wait)
        except openai.APIStatusError as e:
            last_err = e
            # 5xx -> retry. 4xx (other than 429) -> give up.
            if 500 <= e.status_code < 600:
                wait = min(2 ** attempt, 30)
                log.warning("API %d (attempt %d), sleeping %ds", e.status_code, attempt + 1, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(f"OpenAI API {e.status_code}: {e}") from e
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_err}")


def _call_anthropic(
    provider: Provider,
    system: str,
    rows: Sequence[str],
    max_retries: int = 4,
    timeout_s: float = 60.0,
) -> str:
    """Call Anthropic Messages API. Uses urllib to avoid the SDK dep.

    Returns the assistant text content (first block).
    """
    user = _build_user_prompt(system, rows)
    body = json.dumps({
        "model": provider.deployment,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.0,
    }).encode("utf-8")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read())
            content = payload.get("content", [])
            if not content:
                raise RuntimeError(f"Anthropic returned empty content: {payload}")
            return content[0]["text"]
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"Anthropic HTTP {e.code}: {body_text}")
            if e.code == 429 or 500 <= e.code < 600:
                wait = min(2 ** attempt, 30)
                log.warning("Anthropic %d (attempt %d), sleeping %ds", e.code, attempt + 1, wait)
                time.sleep(wait)
                continue
            raise last_err
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            log.warning("Anthropic transient (attempt %d): %s", attempt + 1, e)
            time.sleep(wait)
    raise RuntimeError(f"Anthropic call failed after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str, expected_n: int, layer: str) -> List[dict]:
    """Parse the model's JSON response. Expects {"results": [...]}.

    Robust to:
      - Code-fenced JSON (```json ... ```)
      - Top-level array vs object
      - Comments before/after the JSON
    Returns a list of dicts of length `expected_n`. Raises ValueError if
    the response is unusable.
    """
    s = text.strip()
    # Strip code fences
    if s.startswith("```"):
        lines = s.splitlines()
        s = "\n".join(lines[1:])
        if s.endswith("```"):
            s = s[: s.rfind("```")]
        s = s.strip()
    # Find the first { or [
    obj_idx = s.find("{")
    arr_idx = s.find("[")
    if obj_idx == -1 and arr_idx == -1:
        raise ValueError(f"{layer} response contained no JSON: {s[:200]!r}")
    start = obj_idx if arr_idx == -1 else (arr_idx if obj_idx == -1 else min(obj_idx, arr_idx))
    end = max(s.rfind("}"), s.rfind("]"))
    if end == -1:
        raise ValueError(f"{layer} response unterminated JSON: {s[:200]!r}")
    blob = s[start : end + 1]
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as e:
        raise ValueError(f"{layer} response JSON parse error: {e}; body: {blob[:200]!r}")
    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, dict):
        # Prefer 'results', fall back to any list-typed value
        if "results" in parsed and isinstance(parsed["results"], list):
            rows = parsed["results"]
        else:
            list_keys = [k for k, v in parsed.items() if isinstance(v, list)]
            if list_keys:
                rows = parsed[list_keys[0]]
            else:
                raise ValueError(f"{layer} response dict has no list-typed value: {list(parsed.keys())}")
    else:
        raise ValueError(f"{layer} response is not a dict or list: {type(parsed).__name__}")
    if len(rows) != expected_n:
        raise ValueError(
            f"{layer} response has {len(rows)} rows, expected {expected_n}"
        )
    return rows


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

def batched_classify(
    rows: Sequence[str],
    task: str,
    *,
    provider: Optional[Provider] = None,
    batch_size: int = 50,
    max_concurrent: int = 1,
    send_raw_text: bool = False,
    model: Optional[str] = None,
    prompt_version: str = "v1",
) -> List[dict]:
    """Run `task` against all rows via the LLM backend. Returns a list
    of dicts in the same order as input. Each dict conforms to the
    task's schema:

      task="category"  -> {category: str, secondary?: str, score?: float}
      task="sentiment" -> {label: str, score: float}
      task="emotion"   -> {label: str, score: float}

    Args:
      rows: input comments.
      task: "category" | "sentiment" | "emotion".
      provider: explicit provider, or None to autodetect from env.
      batch_size: rows per LLM call.
      max_concurrent: parallel HTTP requests (1 = serial).
      send_raw_text: if True, skip PII scrubbing. Default False.
      model: override the provider's default deployment.
      prompt_version: which prompt file to load.

    Raises:
      RuntimeError if no provider is configured.
      ValueError if the model returns malformed JSON.
    """
    if not rows:
        return []
    if task not in ("category", "sentiment", "emotion"):
        raise ValueError(f"unknown task: {task!r}")

    provider = provider or detect_provider()
    if provider is None:
        raise RuntimeError(
            "no LLM provider configured: set AZURE_OPENAI_ENDPOINT + "
            "AZURE_OPENAI_API_KEY, or OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        )

    if model:
        provider = Provider(name=provider.name, deployment=model, api_version=provider.api_version)

    system = load_prompt(task, version=prompt_version)

    # Scrub once up front so all downstream calls see the same data.
    if not send_raw_text:
        rows = [_scrub_text(r) for r in rows]

    results: List[Optional[dict]] = [None] * len(rows)

    if max_concurrent <= 1:
        # Serial path (no thread pool needed).
        offset = 0
        for batch in chunk(rows, batch_size):
            raw = (
                _call_anthropic(provider, system, batch)
                if provider.name == "anthropic"
                else _call_openai_compatible(provider, system, batch)
            )
            parsed = _parse_response(raw, expected_n=len(batch), layer=task)
            for j, item in enumerate(parsed):
                results[offset + j] = item
            offset += len(batch)
    else:
        # Parallel path: submit N batches at once with a thread pool.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        batches = list(chunk(rows, batch_size))
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {}
            for i, batch in enumerate(batches):
                fn = _call_anthropic if provider.name == "anthropic" else _call_openai_compatible
                futures[pool.submit(fn, provider, system, batch)] = i
            # Map result index -> first global index in `results`
            batch_starts = [i * batch_size for i in range(len(batches))]
            for fut in as_completed(futures):
                bi = futures[fut]
                raw = fut.result()
                parsed = _parse_response(raw, expected_n=len(batches[bi]), layer=task)
                for j, item in enumerate(parsed):
                    results[batch_starts[bi] + j] = item

    return results  # type: ignore[return-value]


__all__ = [
    "Provider",
    "detect_provider",
    "load_prompt",
    "batched_classify",
    "chunk",
]
