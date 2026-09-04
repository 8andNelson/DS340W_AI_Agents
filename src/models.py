# Model registry — all available OpenRouter models for this project.
# Use the "call" value as the model ID in API requests.
# Update this file when new models are added to the OpenRouter account.

MODELS = {
    # --- Free tier ---
    "dots3-note-free": {
        "call": "dots-studio/dots-3-note-preview:free",
        "provider": "Dots Studio",
        "name": "Dots3-Note Preview (free)",
        "context_k": 512,
        "cost": "free",
        "strengths": ["reasoning", "coding", "multimodal", "long-context", "multi-step agents"],
        "notes": "16B active / 280B total MoE. Free tier. Good default for development.",
    },
    "cohere-north-mini-free": {
        "call": "cohere/north-mini-code:free",
        "provider": "Cohere",
        "name": "North Mini Code (free)",
        "context_k": None,
        "cost": "free",
        "strengths": ["coding"],
        "notes": "Free coding model.",
    },

    # --- Anthropic Claude ---
    "claude-haiku-4.5": {
        "call": "anthropic/claude-haiku-4.5",
        "provider": "Anthropic",
        "name": "Claude Haiku 4.5",
        "context_k": 200,
        "cost": "low",
        "strengths": ["fast", "lightweight tasks", "structured output"],
        "notes": "Primary fast/cheap model for intake, formatting, and simple classification.",
    },
    "claude-sonnet-4.5": {
        "call": "anthropic/claude-sonnet-4.5",
        "provider": "Anthropic",
        "name": "Claude Sonnet 4.5",
        "context_k": 200,
        "cost": "medium",
        "strengths": ["reasoning", "research", "writing", "code"],
        "notes": "Primary capable model for research, validation, and report generation.",
    },
    "claude-sonnet-4.6": {
        "call": "anthropic/claude-sonnet-4.6",
        "provider": "Anthropic",
        "name": "Claude Sonnet 4.6",
        "context_k": 200,
        "cost": "medium",
        "strengths": ["reasoning", "research", "writing", "code"],
        "notes": "Upgraded Sonnet. Use when 4.5 is insufficient.",
    },
    "claude-opus-4.5": {
        "call": "anthropic/claude-opus-4.5",
        "provider": "Anthropic",
        "name": "Claude Opus 4.5",
        "context_k": 200,
        "cost": "high",
        "strengths": ["complex reasoning", "long documents", "master agent decisions"],
        "notes": "Most capable Anthropic model. Reserve for Master Agent critical decisions.",
    },
    "claude-fable-5": {
        "call": "anthropic/claude-fable-5",
        "provider": "Anthropic",
        "name": "Claude Fable 5",
        "context_k": None,
        "cost": "high",
        "strengths": ["unknown — new model"],
        "notes": "Latest Anthropic model. Evaluate before using in production.",
    },

    # --- Google Gemini ---
    "gemini-2.5-flash-lite": {
        "call": "google/gemini-2.5-flash-lite",
        "provider": "Google",
        "name": "Gemini 2.5 Flash Lite",
        "context_k": None,
        "cost": "low",
        "strengths": ["fast", "multimodal"],
        "notes": "Lightest Gemini option. Alternative to Claude Haiku for fast tasks.",
    },
    "gemini-2.5-flash": {
        "call": "google/gemini-2.5-flash",
        "provider": "Google",
        "name": "Gemini 2.5 Flash",
        "context_k": None,
        "cost": "low-medium",
        "strengths": ["fast", "multimodal", "long-context"],
        "notes": "Good balance of speed and capability.",
    },
    "gemini-2.5-pro": {
        "call": "google/gemini-2.5-pro",
        "provider": "Google",
        "name": "Gemini 2.5 Pro",
        "context_k": None,
        "cost": "high",
        "strengths": ["complex reasoning", "long-context", "multimodal"],
        "notes": "Google's top model. Useful for very long document tasks.",
    },

    # --- DeepSeek ---
    "deepseek-chat": {
        "call": "deepseek/deepseek-chat",
        "provider": "DeepSeek",
        "name": "DeepSeek Chat",
        "context_k": None,
        "cost": "low",
        "strengths": ["coding", "reasoning", "cost-effective"],
        "notes": "Strong coding and reasoning at very low cost.",
    },
    "deepseek-r1": {
        "call": "deepseek/deepseek-r1",
        "provider": "DeepSeek",
        "name": "DeepSeek R1",
        "context_k": None,
        "cost": "medium",
        "strengths": ["reasoning", "math", "code"],
        "notes": "Reasoning-focused model. Comparable to o1-class models.",
    },

    # --- Amazon Nova ---
    "nova-micro": {
        "call": "amazon/nova-micro-v1",
        "provider": "Amazon",
        "name": "Nova Micro v1",
        "context_k": None,
        "cost": "very-low",
        "strengths": ["fast", "cheap"],
        "notes": "Extremely cheap. Suitable for bulk classification or filtering tasks.",
    },
    "nova-lite": {
        "call": "amazon/nova-lite-v1",
        "provider": "Amazon",
        "name": "Nova Lite v1",
        "context_k": None,
        "cost": "low",
        "strengths": ["fast", "multimodal"],
        "notes": "Lightweight multimodal model.",
    },
    "nova-pro": {
        "call": "amazon/nova-pro-v1",
        "provider": "Amazon",
        "name": "Nova Pro v1",
        "context_k": None,
        "cost": "medium",
        "strengths": ["reasoning", "multimodal"],
        "notes": "Amazon's capable tier.",
    },
}


def get_model(key: str) -> str:
    """Return the OpenRouter call string for a model key."""
    if key not in MODELS:
        raise KeyError(f"Unknown model key '{key}'. Available: {list(MODELS.keys())}")
    return MODELS[key]["call"]


def list_models(cost_tier: str = None) -> list:
    """Return all model keys, optionally filtered by cost tier."""
    if cost_tier:
        return [k for k, v in MODELS.items() if v["cost"] == cost_tier]
    return list(MODELS.keys())
