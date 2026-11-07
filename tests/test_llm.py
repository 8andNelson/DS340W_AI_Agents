from unittest.mock import MagicMock, patch

import pytest
import requests

from src import llm


def response(status, content="ok"):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(f"{status} error")
    return r


@patch("src.llm.OPENROUTER_API_KEY", "key")
@patch("src.llm.time.sleep")
def test_retries_429_then_succeeds(sleep):
    with patch("src.llm.requests.post", side_effect=[response(429), response(200, "hello")]) as post:
        assert llm.chat("hi") == "hello"
    assert post.call_count == 2
    sleep.assert_called_once_with(llm.RATE_LIMIT_BACKOFF_SECONDS)


@patch("src.llm.OPENROUTER_API_KEY", "key")
@patch("src.llm.time.sleep")
def test_gives_up_after_retries(sleep):
    with patch("src.llm.requests.post", return_value=response(429)) as post:
        with pytest.raises(requests.HTTPError):
            llm.chat("hi")
    assert post.call_count == llm.RATE_LIMIT_RETRIES + 1


@patch("src.llm.OPENROUTER_API_KEY", "key")
@patch("src.llm.time.sleep")
def test_other_errors_are_not_retried(sleep):
    with patch("src.llm.requests.post", return_value=response(500)) as post:
        with pytest.raises(requests.HTTPError):
            llm.chat("hi")
    assert post.call_count == 1
    sleep.assert_not_called()


@patch("src.llm.OPENROUTER_API_KEY", "key")
def test_system_prompt_is_sent_first():
    with patch("src.llm.requests.post", return_value=response(200)) as post:
        llm.chat("hi", system="sys")
    messages = post.call_args.kwargs["json"]["messages"]
    assert messages == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


class TestParseCode:
    def test_python_block(self):
        content = 'Here:\n```json\n{"a": 1}\n```\n```python\nprint("hi")\n```'
        assert llm.parse_code(content) == 'print("hi")\n'

    def test_longest_block_wins(self):
        content = "```python\nx = 1\n```\ntext\n```python\nimport os\nprint(os.name)\n```"
        assert llm.parse_code(content) == "import os\nprint(os.name)\n"

    def test_json_code_fallback(self):
        assert llm.parse_code('{"code": "print(1)"}') == "print(1)"

    def test_nothing_found(self):
        assert llm.parse_code("no code here") == ""


class TestParseJson:
    def test_json_block_next_to_python_block(self):
        content = '```json\n{"inferred_assumptions": ["a"]}\n```\n```python\nd = {"x": 1}\n```'
        assert llm.parse_json(content) == {"inferred_assumptions": ["a"]}

    def test_default_when_unparseable(self):
        assert llm.parse_json("{not json", default={}) == {}


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


def _claude_message(text="hello", stop_reason="end_turn", stop_details=None):
    thinking = MagicMock(type="thinking")
    text_block = MagicMock(type="text", text=text)
    return MagicMock(content=[thinking, text_block], stop_reason=stop_reason, stop_details=stop_details)


def _fake_client(message):
    client = MagicMock()
    client.with_options.return_value = client
    client.beta.messages.stream.side_effect = lambda **kwargs: _FakeStream(message)
    return client


class TestClaudeProvider:
    def _chat(self, message, **kwargs):
        client = _fake_client(message)
        with patch.object(llm, "LLM_PROVIDER", "anthropic"), patch.object(llm, "_client", return_value=client):
            return llm.chat("hi", **kwargs), client

    def test_returns_text_blocks_only(self):
        text, _ = self._chat(_claude_message("answer"))
        assert text == "answer"

    def test_request_shape(self):
        _, client = self._chat(_claude_message(), system="sys", effort="high")
        kwargs = client.beta.messages.stream.call_args.kwargs
        assert kwargs["model"] == llm.CLAUDE_MODEL
        assert kwargs["system"] == "sys"
        assert kwargs["output_config"] == {"effort": "high"}
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["fallbacks"] == "default"
        assert kwargs["betas"] == [llm.CLAUDE_FALLBACK_BETA]
        assert "thinking" not in kwargs  # Opus 5.5: thinking is always on; effort is the control

    def test_default_effort_from_config(self):
        _, client = self._chat(_claude_message())
        assert client.beta.messages.stream.call_args.kwargs["output_config"] == {"effort": llm.CLAUDE_EFFORT}

    def test_no_fallback_for_models_that_do_not_accept_it(self):
        _, client = self._chat(_claude_message(), model="claude-haiku-5-5")
        kwargs = client.beta.messages.stream.call_args.kwargs
        assert "fallbacks" not in kwargs and "betas" not in kwargs

    def test_refusal_raises(self):
        with pytest.raises(RuntimeError, match="declined"):
            self._chat(_claude_message("", stop_reason="refusal", stop_details=MagicMock(category="cyber")))

    def test_openrouter_still_selectable(self):
        with patch.object(llm, "LLM_PROVIDER", "openrouter"), \
             patch.object(llm, "OPENROUTER_API_KEY", "key"), \
             patch("src.llm.requests.post", return_value=response(200, "from openrouter")):
            assert llm.chat("hi") == "from openrouter"

    def test_unknown_provider_is_an_error(self):
        with patch.object(llm, "LLM_PROVIDER", "other"), pytest.raises(ValueError):
            llm.chat("hi")


@patch("src.llm.OPENROUTER_API_KEY", "key")
@patch("src.llm.time.sleep")
def test_daily_quota_fails_fast_without_retries(sleep):
    quota = response(429)
    quota.json.return_value = {"error": {"message": "Rate limit exceeded: free-models-per-day", "metadata": {
        "limit_source": "openrouter_free_tier_daily",
        "headers": {"X-RateLimit-Limit": "50", "X-RateLimit-Reset": "1794096000000"}}}}
    with patch("src.llm.requests.post", return_value=quota) as post:
        with pytest.raises(llm.DailyQuotaExceeded, match="50 requests/day"):
            llm.chat("hi")
    assert post.call_count == 1
    sleep.assert_not_called()
