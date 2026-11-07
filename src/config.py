import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SERPENT_API_KEY = os.getenv("SERPENT_API_KEY", "")
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
