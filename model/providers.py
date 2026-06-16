"""
providers.py — Models to test (one per architecture) and the API calls.

Gathers in a SINGLE place the model/provider-specific part; all the rest of the
pipeline (loaders, prompting, executor, metrics, report, export) is shared and
model-agnostic — so the LLM vs MoE vs SLM comparison is fair (same prompt, same
execution, same metrics for all).

  MODELS / ALL_MODELS            the registered models (one ModelSpec per model)
  generate(spec, prompt)         the API call, dispatched by provider
  require_key / models_by_keys   helpers for keys and model selection

Providers: Anthropic (its own SDK) and providers with an OpenAI-compatible API
(DeepSeek, Mistral) through the single `openai` client (only base_url and key
change).
"""

import os
from functools import lru_cache

from .config import (
    ModelSpec, SYSTEM_PROMPT, MAX_TOKENS, CLAUDE_EFFORT, CLAUDE_THINKING,
    TEMPERATURE, PROVIDER_ENV_KEYS, MissingKeyError,
)

# Modelli confrontati, UNO per architettura, per la tabella "errori per
# architettura" del report. Verifica/aggiorna i model_id sulle console dei
# provider prima di lanciare (deprecano/rinominano spesso).
MODELS = [
    ModelSpec("claude-opus-4.7", "claude-opus-4-7", "anthropic", "LLM",
              "Claude Opus 4.7 (dense, frontier)"),
    ModelSpec("deepseek-v4-pro", "deepseek-v4-pro", "deepseek", "MoE",
              "DeepSeek V4 Pro (MoE 1.6T/49B attivi, frontier, non-thinking)"),
    ModelSpec("ministral-8b", "ministral-8b-latest", "mistral", "SLM",
              "Mistral Ministral 3 8B (edge/efficiente, dic 2025)"),
]
ALL_MODELS = MODELS

# Provider con API OpenAI-compatibile (stesso client `openai`, solo base_url e
# chiave diversi). Anthropic ha invece il suo SDK (vedi _generate_anthropic).
OPENAI_COMPATIBLE_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "mistral": "https://api.mistral.ai/v1",
}


def require_key(provider: str) -> str:
    """Read the provider's API key from the environment or raise MissingKeyError."""
    env_name = PROVIDER_ENV_KEYS[provider]
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise MissingKeyError(f"Manca {env_name} (provider '{provider}')")
    return key


def models_by_keys(keys: list[str] | None) -> list[ModelSpec]:
    """Filter the models by the requested keys (None = all)."""
    if not keys:
        return list(ALL_MODELS)
    wanted = set(keys)
    return [m for m in ALL_MODELS if m.key in wanted]


@lru_cache(maxsize=None)
def get_balance(provider: str) -> dict | None:
    """Remaining balance of the provider, if the API exposes it, as
    `{"available": bool, "amount": float, "currency": str}`; `None` if the
    provider has no balance endpoint (Anthropic and Mistral: only viewable from
    the web console) or if the call fails. Per-process cache: a single network
    request per provider per run."""
    if provider != "deepseek":
        return None  # solo DeepSeek espone GET /user/balance via API
    import json
    import urllib.request
    try:
        key = require_key(provider)
        req = urllib.request.Request(
            "https://api.deepseek.com/user/balance",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return None
    infos = data.get("balance_infos") or [{}]
    info = infos[0]
    return {
        "available": bool(data.get("is_available")),
        "amount": float(info.get("total_balance") or 0),
        "currency": info.get("currency", "USD"),
    }


@lru_cache(maxsize=1)
def _anthropic_client():
    """Lazily build and cache the Anthropic SDK client (one per process), so the
    `anthropic` import and client setup happen only when Claude is actually used."""
    import anthropic
    return anthropic.Anthropic(api_key=require_key("anthropic"))


@lru_cache(maxsize=None)
def _openai_client(provider: str):
    """OpenAI-compatible client for the non-Anthropic providers (DeepSeek,
    Mistral): same `openai` SDK, only base_url and key change. Cached per
    provider so each client is built once."""
    from openai import OpenAI
    return OpenAI(api_key=require_key(provider),
                  base_url=OPENAI_COMPATIBLE_BASE_URLS[provider])


def generate(spec: ModelSpec, prompt: str) -> tuple[str, dict]:
    """Return (generated_code, usage) where usage = token counts.
    Dispatches to the right provider: Anthropic has its own SDK, the others use
    the OpenAI-compatible API. This is the single entry point the pipeline calls
    to query any model."""
    if spec.provider == "anthropic":
        return _generate_anthropic(spec, prompt)
    return _generate_openai_compatible(spec, prompt)


def _generate_anthropic(spec: ModelSpec, prompt: str) -> tuple[str, dict]:
    """Generate with Claude via the Anthropic SDK and return (text, usage).
    Behavior is controlled by `thinking` + `output_config.effort` (see config.py),
    NOT by temperature, and the token counts (including cache fields) are
    normalized into the shared `usage` dict used by the report."""
    # ⚠️ NIENTE temperature/top_p/top_k: Opus 4.7/4.8 li rifiuta con 400.
    # Il determinismo/economia si controlla con thinking + effort (vedi config.py).
    resp = _anthropic_client().messages.create(
        model=spec.model_id,
        system=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
        thinking={"type": CLAUDE_THINKING},
        output_config={"effort": CLAUDE_EFFORT},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    usage = {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
    return text, usage


def _generate_openai_compatible(spec: ModelSpec, prompt: str) -> tuple[str, dict]:
    """Generate via the OpenAI-compatible API (DeepSeek, Mistral) and return
    (text, usage). Unlike Claude, these support `temperature` (set to 0.0 for
    determinism) and the system prompt is a normal `system` role message. For
    DeepSeek we force non-thinking mode for fairness with Claude's regime."""
    extra = {}
    if spec.provider == "deepseek":
        # Equità col regime di Opus (CLAUDE_THINKING="disabled", effort "low"):
        # DeepSeek V4 di default RAGIONA → forziamo la modalità non-thinking, così
        # nessun modello ha un budget di reasoning che gli altri non hanno.
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = _openai_client(spec.provider).chat.completions.create(
        model=spec.model_id,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        **extra,
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    return text, usage
