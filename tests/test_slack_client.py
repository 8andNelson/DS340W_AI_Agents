import unittest
from unittest.mock import MagicMock, patch

from src.slack import slack_client


class TestIsConfigured(unittest.TestCase):
    def test_false_when_token_missing(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", ""), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C123"):
            self.assertFalse(slack_client.is_configured())

    def test_false_when_channel_missing(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", ""):
            self.assertFalse(slack_client.is_configured())

    def test_true_when_both_set(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C123"):
            self.assertTrue(slack_client.is_configured())


class TestIsErrorsConfigured(unittest.TestCase):
    def test_false_when_errors_channel_missing(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_ERRORS_CHANNEL_ID", ""):
            self.assertFalse(slack_client.is_errors_configured())

    def test_true_when_errors_channel_set(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_ERRORS_CHANNEL_ID", "C_ERR"):
            self.assertTrue(slack_client.is_errors_configured())


class TestPostErrorMessage(unittest.TestCase):
    def test_posts_to_errors_channel_when_configured(self):
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True, "ts": "1.0"}
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C_MAIN"), \
             patch.object(slack_client, "SLACK_ERRORS_CHANNEL_ID", "C_ERR"), \
             patch.object(slack_client, "_get_client", return_value=mock_client):
            slack_client.post_error_message("[Master Agent]\nACTION: Correction")

        mock_client.chat_postMessage.assert_called_once_with(
            channel="C_ERR", text="[Master Agent]\nACTION: Correction",
        )

    def test_falls_back_to_main_channel_when_errors_channel_not_set(self):
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True, "ts": "1.0"}
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C_MAIN"), \
             patch.object(slack_client, "SLACK_ERRORS_CHANNEL_ID", ""), \
             patch.object(slack_client, "_get_client", return_value=mock_client):
            slack_client.post_error_message("a correction that must not be dropped")

        mock_client.chat_postMessage.assert_called_once_with(
            channel="C_MAIN", text="a correction that must not be dropped",
        )

    def test_returns_none_when_no_token_at_all(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", ""):
            self.assertIsNone(slack_client.post_error_message("hello"))


class TestPostMessage(unittest.TestCase):
    def test_returns_none_when_not_configured(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", ""):
            self.assertIsNone(slack_client.post_message("[Data Agent]\nSTATUS: Working"))

    def test_posts_and_returns_ts_on_success(self):
        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True, "ts": "1700000000.000100"}
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C123"), \
             patch.object(slack_client, "_get_client", return_value=mock_client):
            ts = slack_client.post_message("[Data Agent]\nSTATUS: Working")

        self.assertEqual(ts, "1700000000.000100")
        mock_client.chat_postMessage.assert_called_once_with(
            channel="C123", text="[Data Agent]\nSTATUS: Working",
        )

    def test_returns_none_and_does_not_raise_on_api_failure(self):
        mock_client = MagicMock()
        mock_client.chat_postMessage.side_effect = RuntimeError("network down")
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C123"), \
             patch.object(slack_client, "_get_client", return_value=mock_client):
            ts = slack_client.post_message("hello")

        self.assertIsNone(ts)


class TestFetchNewMessages(unittest.TestCase):
    def test_returns_empty_list_when_not_configured(self):
        with patch.object(slack_client, "SLACK_BOT_TOKEN", ""):
            self.assertEqual(slack_client.fetch_new_messages(), [])

    def test_returns_messages_oldest_first(self):
        mock_client = MagicMock()
        mock_client.conversations_history.return_value = {
            "messages": [
                {"ts": "200.0", "text": "second", "user": "U2"},
                {"ts": "100.0", "text": "first", "user": "U1"},
            ]
        }
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C123"), \
             patch.object(slack_client, "_get_client", return_value=mock_client):
            messages = slack_client.fetch_new_messages(oldest_ts="50.0")

        self.assertEqual([m["text"] for m in messages], ["first", "second"])
        mock_client.conversations_history.assert_called_once_with(
            channel="C123", limit=50, oldest="50.0", inclusive=False,
        )

    def test_returns_empty_list_on_api_failure(self):
        mock_client = MagicMock()
        mock_client.conversations_history.side_effect = RuntimeError("network down")
        with patch.object(slack_client, "SLACK_BOT_TOKEN", "xoxb-fake"), \
             patch.object(slack_client, "SLACK_CHANNEL_ID", "C123"), \
             patch.object(slack_client, "_get_client", return_value=mock_client):
            self.assertEqual(slack_client.fetch_new_messages(), [])


if __name__ == "__main__":
    unittest.main()
