"""
Thin wrapper around slack_sdk's WebClient for posting to, and reading from,
the project's two configured Slack channels (src/config.py):

  * SLACK_CHANNEL_ID        -- main channel: routine agent status
                               (started/finished) and Master Agent
                               Approvals. Also where fetch_new_messages
                               reads from, for the agent-request relay.
  * SLACK_ERRORS_CHANNEL_ID -- errors_corrections channel: Master Agent
                               Corrections and agent exceptions, so there
                               is one place to check for anything that
                               actually needs attention. Falls back to the
                               main channel when unset, so a correction is
                               never silently dropped just because a
                               second channel hasn't been configured yet.

Slack is off unless SLACK_ENABLED=true (src/config.py): the dashboard
shows the same progress, so the project does not need Slack. When it is
off, nothing is posted or read, and slack_sdk does not need to be
installed (it is imported only when a client is first needed).

When it is on, Slack is an observability layer, not a hard dependency -- every function here fails soft. A missing token, a
network error, or a bad channel ID prints a warning and returns None/[]
rather than raising, so an agent's actual work (research, downloading
data, etc.) never fails because Slack is unreachable or unconfigured, and
the project still runs exactly as it did before Slack was wired in when
SLACK_BOT_TOKEN/SLACK_CHANNEL_ID aren't set (same optionality pattern as
KAGGLE_USERNAME/KAGGLE_KEY -- see CLAUDE.md's Secrets section).
"""
from src.config import SLACK_BOT_TOKEN, SLACK_CHANNEL_ID, SLACK_ENABLED, SLACK_ERRORS_CHANNEL_ID

_client = None


def _get_client():
    global _client
    if _client is None and SLACK_BOT_TOKEN:
        try:
            from slack_sdk import WebClient
        except ImportError:
            print("[Slack] SLACK_ENABLED is true, but slack_sdk is not installed (pip install slack_sdk).")
            return None
        _client = WebClient(token=SLACK_BOT_TOKEN)
    return _client


def is_configured() -> bool:
    return bool(SLACK_ENABLED and SLACK_BOT_TOKEN and SLACK_CHANNEL_ID)


def is_errors_configured() -> bool:
    return bool(SLACK_ENABLED and SLACK_BOT_TOKEN and SLACK_ERRORS_CHANNEL_ID)


def _post(channel: str, text: str) -> str | None:
    if not SLACK_ENABLED or not SLACK_BOT_TOKEN or not channel:
        return None
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.chat_postMessage(channel=channel, text=text)
        return resp.get("ts")
    except Exception as e:
        print(f"[Slack] Could not post message: {e}")
        return None


def post_message(text: str) -> str | None:
    """
    Post text to the main channel. Returns the message's Slack timestamp
    (its id -- also usable as an ordering cursor) on success, None on any
    failure or when Slack isn't configured.
    """
    return _post(SLACK_CHANNEL_ID, text)


def post_error_message(text: str) -> str | None:
    """
    Post text to the errors_corrections channel -- for Master Agent
    Corrections and agent exceptions specifically, so there's one place to
    check for anything that needs attention. Falls back to the main
    channel if SLACK_ERRORS_CHANNEL_ID isn't set, per this project's
    "never silently drop" policy: a correction still gets posted
    somewhere rather than vanishing just because the second channel
    hasn't been configured yet.
    """
    if SLACK_ERRORS_CHANNEL_ID:
        return _post(SLACK_ERRORS_CHANNEL_ID, text)
    return _post(SLACK_CHANNEL_ID, text)


def fetch_new_messages(oldest_ts: str = None, limit: int = 50) -> list:
    """
    Read messages posted to the main channel after oldest_ts, oldest-first.
    oldest_ts is exclusive (a message already at that exact timestamp is
    not re-returned), so a caller can safely pass back the ts of the last
    message it already processed. Returns [] on any failure or when
    Slack isn't configured -- this is Master Agent's read side of the
    channel, used to notice another agent's request without running a
    separate long-lived listener process.
    """
    if not is_configured():
        return []
    client = _get_client()
    if client is None:
        return []
    try:
        kwargs = {"channel": SLACK_CHANNEL_ID, "limit": limit}
        if oldest_ts:
            kwargs["oldest"] = oldest_ts
            kwargs["inclusive"] = False
        resp = client.conversations_history(**kwargs)
        messages = resp.get("messages", [])
    except Exception as e:
        print(f"[Slack] Could not fetch messages: {e}")
        return []

    messages = sorted(messages, key=lambda m: float(m.get("ts", 0) or 0))
    return [
        {"ts": m.get("ts", ""), "text": m.get("text", ""), "user": m.get("user", m.get("bot_id", ""))}
        for m in messages
    ]
