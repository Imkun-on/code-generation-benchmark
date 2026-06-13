"""
providers.py — Modelli da testare (uno per architettura) e chiamata alle API.

Raccoglie in un SOLO posto la parte specifica del modello/provider; tutto il
resto della pipeline (loader, prompting, executor, metriche, report, export) è
condiviso e agnostico al modello — così il confronto LLM vs MoE vs SLM è equo
(stesso prompt, stessa esecuzione, stesse metriche per tutti).

  MODELS / ALL_MODELS            i modelli registrati (un ModelSpec per modello)
  generate(spec, prompt)         la chiamata API, smistata per provider
  require_key / models_by_keys   utilità su chiavi e selezione dei modelli

Provider: Anthropic (SDK proprio) e provider con API OpenAI-compatibile
(DeepSeek, Mistral) tramite l'unico client `openai` (cambiano solo base_url e
chiave).
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
    """Legge la API key del provider dall'ambiente o solleva MissingKeyError."""
    env_name = PROVIDER_ENV_KEYS[provider]
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise MissingKeyError(f"Manca {env_name} (provider '{provider}')")
    return key


def models_by_keys(keys: list[str] | None) -> list[ModelSpec]:
    """Filtra i modelli per le chiavi richieste (None = tutti)."""
    if not keys:
        return list(ALL_MODELS)
    wanted = set(keys)
    return [m for m in ALL_MODELS if m.key in wanted]


@lru_cache(maxsize=None)
def get_balance(provider: str) -> dict | None:
    """Saldo residuo del provider, se l'API lo espone, come
    `{"available": bool, "amount": float, "currency": str}`; `None` se il provider
    non ha un endpoint saldo (Anthropic e Mistral: consultabile solo dalla console
    web) o se la chiamata fallisce. Cache per-processo: una sola richiesta di rete
    per provider per esecuzione."""
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
    import anthropic
    return anthropic.Anthropic(api_key=require_key("anthropic"))


@lru_cache(maxsize=None)
def _openai_client(provider: str):
    """Client OpenAI-compatibile per i provider non-Anthropic (DeepSeek, Mistral):
    stesso SDK `openai`, cambiano solo base_url e chiave."""
    from openai import OpenAI
    return OpenAI(api_key=require_key(provider),
                  base_url=OPENAI_COMPATIBLE_BASE_URLS[provider])


def generate(spec: ModelSpec, prompt: str) -> tuple[str, dict]:
    """Restituisce (codice_generato, usage) dove usage = conteggio token.
    Smista sul provider giusto: Anthropic ha il suo SDK, gli altri usano l'API
    OpenAI-compatibile."""
    if spec.provider == "anthropic":
        return _generate_anthropic(spec, prompt)
    return _generate_openai_compatible(spec, prompt)


def _generate_anthropic(spec: ModelSpec, prompt: str) -> tuple[str, dict]:
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
    """Generazione via API OpenAI-compatibile (DeepSeek, Mistral). A differenza di
    Claude, questi supportano `temperature` (impostata a 0.0 per determinismo) e il
    system prompt è un normale messaggio di ruolo `system`."""
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
