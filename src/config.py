import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SERPENT_API_KEY = os.getenv("SERPENT_API_KEY", "")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID", "")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free model — no credits required, use for development and testing
FREE_MODEL = "dots-studio/dots-3-note-preview:free"
# Fast/cheap model for lightweight structuring tasks
FAST_MODEL = "anthropic/claude-haiku-4.5"
# More capable model for complex reasoning tasks
CAPABLE_MODEL = "anthropic/claude-sonnet-4.5"

# Active model used by agents — swap to FAST_MODEL or CAPABLE_MODEL once credits are loaded
DEFAULT_MODEL = FREE_MODEL
