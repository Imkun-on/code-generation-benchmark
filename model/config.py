"""
config.py — Shared definitions: architectures, the structure of a model,
generation constants and the providers' endpoints.

The LIST of models lives in model/claude.py (the only remaining provider).
"""

from dataclasses import dataclass


# ------------------------------------------------------------------
# Architetture confrontate (sottoinsieme testabile per la generazione
# di codice). SAM e MLM sono esclusi: non generano codice.
# ------------------------------------------------------------------
ARCHITECTURES = {
    "LLM": "Large Language Model (grande, dense)",
    "SLM": "Small Language Model (piccolo / efficiente)",
    "MoE": "Mixture of Experts (esperti attivati selettivamente)",
    "VLM": "Vision-Language Model (multimodale: testo + immagini)",
}


@dataclass(frozen=True)
class ModelSpec:
    """Description of a model to be tested.

    Frozen dataclass that holds everything the pipeline needs to identify a
    model and route it to the right provider: the short `key` used in output
    files, the exact `model_id` required by the provider's API, the `provider`
    (which selects the API client and credentials) and the `architecture`
    (LLM/MoE/SLM/VLM) used to group results in the report. `note` is a free-text
    description (size, parameters…)."""
    key: str            # nome breve usato nei file di output
    model_id: str       # id esatto richiesto dall'API del provider
    provider: str       # anthropic | openai | google | deepseek | mistral
    architecture: str   # una delle chiavi di ARCHITECTURES
    note: str = ""      # nota descrittiva (taglia, parametri...)


# ------------------------------------------------------------------
# Costanti di generazione condivise da tutti i provider
# ------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Sei un programmatore Python esperto. "
    "Rispondi ESCLUSIVAMENTE con il codice Python della soluzione, pronto da eseguire: "
    "includi la firma completa della funzione e tutti gli import necessari. "
    "NON scrivere spiegazioni, NON scrivere testo introduttivo, "
    "NON racchiudere il codice in blocchi markdown o backtick (```). "
    "La tua intera risposta deve essere solo codice."
)
MAX_TOKENS = 2048  # margine per firma + import + corpo funzione senza troncare

# TEMPERATURE è usato SOLO dai provider che lo supportano (OpenAI, Gemini,
# DeepSeek, Mistral). ⚠️ Claude Opus 4.7/4.8 ha RIMOSSO temperature/top_p/top_k:
# inviarli dà errore 400. Per Claude il determinismo si guida col prompt + effort.
TEMPERATURE = 0.0  # deterministico: pass@1 stabile (non-Claude)

# --- Parametri specifici di Claude (Opus 4.7+) ---
# effort governa quanto il modello "ragiona/spende": low|medium|high|xhigh|max.
# Per un benchmark pass@1 economico su HumanEval teniamo gli stimoli bassi:
# thinking disattivato + effort basso = meno token = meno spreco di risorse.
# Alza a "high" se vuoi misurare la capacità massima del modello.
CLAUDE_EFFORT = "low"
CLAUDE_THINKING = "disabled"  # "disabled" | "adaptive"

# Variabile d'ambiente che contiene la chiave di ciascun provider.
PROVIDER_ENV_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

# Prezzi API in USD per 1.000.000 di token (input, output). Chiave = model_id
# esatto. Aggiorna se un provider cambia il listino (verifica sulle console:
# anthropic.com, platform.deepseek.com, mistral.ai/pricing).
PRICING = {
    "claude-opus-4-7": (5.0, 25.0),     # Anthropic — LLM dense (frontier)
    "deepseek-v4-pro": (0.435, 0.87),     # DeepSeek — MoE (1.6T/49B attivi, frontier, non-thinking)
    "ministral-8b-latest": (0.15, 0.15),  # Mistral — SLM (Ministral 3 8B, dic 2025; -latest → ministral-8b-2512)
}


class MissingKeyError(RuntimeError):
    """Raised when the provider's API key is not set in the environment.

    Used to fail fast (with a clear message) when a model is requested but its
    credentials are missing, instead of letting an opaque API error surface."""
