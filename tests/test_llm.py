import json
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


class TestFreeLLMAPIProvider:
    def _chat(self, routed="groq/llama-3.3-70b-versatile", **kwargs):
        reply = response(200, "routed answer")
        reply.headers = {"X-Routed-Via": routed} if routed else {}
        with patch.object(llm, "LLM_PROVIDER", "freellmapi"), \
             patch.object(llm, "FREELLMAPI_API_KEY", "freellmapi-key"), \
             patch("src.llm.requests.post", return_value=reply) as post:
            return llm.chat("hi", **kwargs), post

    def test_posts_to_local_router_with_auto_model(self):
        text, post = self._chat()
        assert text == "routed answer"
        assert post.call_args.args[0] == f"{llm.FREELLMAPI_BASE_URL}/chat/completions"
        assert post.call_args.kwargs["json"]["model"] == llm.FREELLMAPI_MODEL
        assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer freellmapi-key"

    def test_high_effort_requests_the_smart_model(self):
        _, post = self._chat(effort="high")
        assert post.call_args.kwargs["json"]["model"] == llm.FREELLMAPI_SMART_MODEL

    def test_serving_model_is_recorded_and_logged(self):
        self._chat()
        assert llm.last_served_by() == "groq/llama-3.3-70b-versatile"
        entry = json.loads(llm.CALL_LOG.read_text().splitlines()[-1])
        assert entry["provider"] == "freellmapi"
        assert entry["requested_model"] == llm.FREELLMAPI_MODEL
        assert entry["served_by"] == "groq/llama-3.3-70b-versatile"

    def test_falls_back_to_body_model_without_header(self):
        reply = response(200, "x")
        reply.headers = {}
        reply.json.return_value = {"model": "mistral/mistral-small", "choices": [{"message": {"content": "x"}}]}
        with patch.object(llm, "LLM_PROVIDER", "freellmapi"), patch.object(llm, "FREELLMAPI_API_KEY", "k"), \
             patch("src.llm.requests.post", return_value=reply):
            llm.chat("hi")
        assert llm.last_served_by() == "mistral/mistral-small"

    def test_missing_key_is_a_clear_error(self):
        with patch.object(llm, "LLM_PROVIDER", "freellmapi"), patch.object(llm, "FREELLMAPI_API_KEY", ""), \
             pytest.raises(EnvironmentError, match="FREELLMAPI_API_KEY"):
            llm.chat("hi")


class TestFreeLLMAPIAgentRouting:
    MODELS = {"research_agent": ("primary-model", "backup-model")}

    def _chat(self, replies, **kwargs):
        def reply(r):
            if isinstance(r, Exception):
                return r
            r.headers = {"X-Routed-Via": "p/served"}
            return r
        with patch.object(llm, "LLM_PROVIDER", "freellmapi"), \
             patch.object(llm, "FREELLMAPI_API_KEY", "k"), \
             patch.object(llm, "FREELLMAPI_AGENT_MODELS", self.MODELS), \
             patch("src.llm.time.sleep") as sleep, \
             patch("src.llm.requests.post", side_effect=[reply(r) for r in replies]) as post:
            text = llm.chat("hi", **kwargs)
        return text, [c.kwargs["json"]["model"] for c in post.call_args_list], sleep

    def test_agent_uses_its_primary_model(self):
        text, models, _ = self._chat([response(200, "a")], agent="research_agent")
        assert text == "a"
        assert models == ["primary-model"]

    def test_primary_failure_moves_to_backup(self):
        text, models, _ = self._chat([response(502), response(200, "b")], agent="research_agent")
        assert text == "b"
        assert models == ["primary-model", "backup-model"]

    def test_timeout_and_rate_limit_move_on_without_backoff(self):
        text, models, sleep = self._chat(
            [requests.Timeout("slow"), response(429), response(200, "c")], agent="research_agent")
        assert text == "c"
        assert models == ["primary-model", "backup-model", llm.FREELLMAPI_MODEL]
        sleep.assert_not_called()

    def test_empty_answer_moves_to_backup(self):
        text, models, _ = self._chat([response(200, "  "), response(200, "d")], agent="research_agent")
        assert text == "d"
        assert models == ["primary-model", "backup-model"]

    def test_last_resort_auto_error_is_raised(self):
        with pytest.raises(requests.HTTPError):
            self._chat([response(502), response(502), response(500)], agent="research_agent")

    def test_high_effort_last_resort_is_the_smart_model(self):
        _, models, _ = self._chat([response(502), response(502), response(200, "e")],
                                  agent="research_agent", effort="high")
        assert models[-1] == llm.FREELLMAPI_SMART_MODEL

    def test_unlisted_agent_uses_auto_routing(self):
        _, models, _ = self._chat([response(200, "f")], agent="unknown_agent")
        assert models == [llm.FREELLMAPI_MODEL]

    def test_explicit_model_overrides_agent_routing(self):
        _, models, _ = self._chat([response(200, "g")], agent="research_agent", model="pinned")
        assert models == ["pinned"]

    def test_failed_attempt_and_agent_are_logged(self):
        self._chat([response(502), response(200, "h")], agent="research_agent")
        entries = [json.loads(l) for l in llm.CALL_LOG.read_text().splitlines()]
        assert entries[-2]["requested_model"] == "primary-model" and "error" in entries[-2]
        assert entries[-1]["requested_model"] == "backup-model" and entries[-1]["agent"] == "research_agent"
        assert llm.last_served_by() == "p/served"

    def test_every_configured_agent_has_two_models(self):
        from src.config import FREELLMAPI_AGENT_MODELS
        for agent, models in FREELLMAPI_AGENT_MODELS.items():
            assert len(models) == 2 and all(models), agent


@patch("src.llm.OPENROUTER_API_KEY", "key")
def test_master_agent_notes_are_added_to_the_system_prompt_only_while_active():
    with patch("src.llm.requests.post", return_value=response(200)) as post:
        with llm.supervisor_notes(["Attempt 1 failed: only 3 papers. Do this next time: broaden the queries."]):
            llm.chat("hi", system="You are the Research Agent.")
        llm.chat("hi", system="You are the Research Agent.")
    with_notes = post.call_args_list[0].kwargs["json"]["messages"][0]["content"]
    without = post.call_args_list[1].kwargs["json"]["messages"][0]["content"]
    assert with_notes.startswith("You are the Research Agent.")
    assert "MASTER AGENT NOTES" in with_notes and "broaden the queries" in with_notes
    assert without == "You are the Research Agent."


@patch("src.llm.OPENROUTER_API_KEY", "key")
def test_notes_become_the_system_prompt_when_the_call_has_none():
    with patch("src.llm.requests.post", return_value=response(200)) as post:
        with llm.supervisor_notes(["fix it"]):
            llm.chat("hi")
    messages = post.call_args.kwargs["json"]["messages"]
    assert messages[0]["role"] == "system" and "- fix it" in messages[0]["content"]
