import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Web search (src/search/web_search.py): Brave first, then SerpApi (Google).
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SERPENT_API_KEY = os.getenv("SERPENT_API_KEY", "")  # SerpApi (serpapi.com)
# Free local browser fallback for blocked public pages; no API key or proxy.
CAMOUFOX_ENABLED = os.getenv("CAMOUFOX_ENABLED", "true").lower() in ("true", "1", "yes")
CAMOUFOX_TIMEOUT_SECONDS = max(30, min(300, int(os.getenv("CAMOUFOX_TIMEOUT_SECONDS", "180"))))
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")  # main status/activity channel
SLACK_ERRORS_CHANNEL_ID = os.getenv("SLACK_ERRORS_CHANNEL_ID", "")  # Master Agent corrections/failures
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free model — no credits required, use for development and testing
FREE_MODEL = "dots-studio/dots-3-note-preview:free"
# Fast/cheap model for lightweight structuring tasks
FAST_MODEL = "anthropic/claude-haiku-4.5"
# More capable model for complex reasoning tasks
CAPABLE_MODEL = "anthropic/claude-sonnet-4.5"

# Active model used by agents — swap to FAST_MODEL or CAPABLE_MODEL once credits are loaded
DEFAULT_MODEL = FREE_MODEL

# --- LLM provider ---
# Which provider every agent's LLM call goes through (src/llm.py):
#   "openrouter" -- OpenRouter's chat-completions API with DEFAULT_MODEL above (default)
#   "freellmapi" -- a local FreeLLMAPI router (OpenAI-compatible) that pools many
#                   providers' free tiers and fails over between them on 429/5xx
#   "anthropic"  -- Claude via the official Anthropic SDK. Implemented and
#                   unit-tested, but not yet run live: the project has no
#                   Claude API access yet. Set LLM_PROVIDER=anthropic plus
#                   ANTHROPIC_API_KEY in .env to switch once it does.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
# Claude Opus 5.5's own default effort is "medium"; set explicitly so the
# level is a visible project setting. Code generation/repair calls ask for
# "high" (see the effort= arguments in replication/experiment agents).
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "medium")

# FreeLLMAPI (LLM_PROVIDER=freellmapi): self-hosted router, default port 3001.
# "auto" lets the router pick any healthy model; code generation/repair calls
# (effort="high" or above) ask for FREELLMAPI_SMART_MODEL instead.
FREELLMAPI_BASE_URL = os.getenv("FREELLMAPI_BASE_URL", "http://localhost:3001/v1").rstrip("/")
FREELLMAPI_API_KEY = os.getenv("FREELLMAPI_API_KEY", "")
FREELLMAPI_MODEL = os.getenv("FREELLMAPI_MODEL", "auto")
FREELLMAPI_SMART_MODEL = os.getenv("FREELLMAPI_SMART_MODEL", "auto:smart")

# Per-agent model routing on FreeLLMAPI: (primary, backup) router model ids.
# A requested model id is strict on FreeLLMAPI -- verified live, a timed-out
# gemini-3.8-flash returned 502 instead of moving to another model -- so
# src/llm.py tries the backup itself, then FREELLMAPI_MODEL/SMART_MODEL
# ("auto") as a last resort so one provider outage cannot halt the pipeline.
# An agent not listed here uses auto routing directly.
FREELLMAPI_AGENT_MODELS = {
    "intake_agent":         ("gemini-3.5-flash-lite", "ministral-14b"),      # speed
    "research_agent":       ("gemini-3.6-flash",      "dots3-note-preview"), # reasoning
    "validation_agent":     ("dots3-note-preview",    "qwen3.8-27b"),        # accuracy
    "data_agent":           ("gemini-3.5-flash-lite", "gemini-3.6-flash"),   # efficiency
    "cleaning_agent":       ("qwen3.8-27b",           "gemini-3.7-flash"),   # coding (incl. column mapping)
    "code_discovery_agent": ("gemini-3.6-flash",      "qwen3.8-27b"),        # code analysis
    "replication_agent":    ("gemini-3.6-flash",      "qwen3.8-27b"),        # coding
    "experiment_agent":     ("qwen3.8-27b",           "gemini-3.7-flash"),   # technical reasoning
    "evaluation_agent":     ("dots3-note-preview",    "gemini-3.6-flash"),   # critical reasoning
    "report_agent":         ("gemini-3.6-flash",      "dots3-note-preview"), # synthesis
}
