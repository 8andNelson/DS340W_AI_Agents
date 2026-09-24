import unittest

from src.slack import message_formatter


class TestFormatting(unittest.TestCase):
    def test_format_started_begins_with_agent_name(self):
        text = message_formatter.format_started("Data Agent", "Building dataset pool")
        self.assertTrue(text.startswith("[Data Agent]"))
        self.assertIn("STATUS: Working", text)
        self.assertIn("TASK: Building dataset pool", text)

    def test_format_finished_includes_status_and_summary(self):
        text = message_formatter.format_finished("Cleaning Agent", "Completed", "Merged 3 datasets.")
        self.assertTrue(text.startswith("[Cleaning Agent]"))
        self.assertIn("STATUS: Completed", text)
        self.assertIn("SUMMARY: Merged 3 datasets.", text)
        self.assertNotIn("NEXT:", text)

    def test_format_finished_includes_next_when_given(self):
        text = message_formatter.format_finished("Research Agent", "Completed", "Found 6 papers.", "Validate next.")
        self.assertIn("NEXT: Validate next.", text)

    def test_format_master_entry_matches_claude_md_shape(self):
        entry = {
            "action": "Correction", "target": "Research Agent",
            "issue": "Paper #3 was published in 2021.", "decision": "Rejected",
            "next": "Replace with a 2022+ paper.",
        }
        text = message_formatter.format_master_entry(entry)
        self.assertEqual(text, (
            "[Master Agent]\n"
            "ACTION: Correction\n"
            "TARGET: Research Agent\n"
            "ISSUE: Paper #3 was published in 2021.\n"
            "DECISION: Rejected\n"
            "NEXT: Replace with a 2022+ paper."
        ))

    def test_format_run_started_begins_with_master_agent_and_includes_topic(self):
        text = message_formatter.format_run_started("credit card fraud detection")
        self.assertTrue(text.startswith("[Master Agent]"))
        self.assertIn("New pipeline run started", text)
        self.assertIn("TOPIC: credit card fraud detection", text)


class TestRequestRoundTrip(unittest.TestCase):
    def test_parse_request_reads_back_a_formatted_request(self):
        text = message_formatter.format_request(
            "Modeling Agent", "Data Agent", "Model needs more training examples.", "target=20000",
        )
        parsed = message_formatter.parse_request(text)
        self.assertEqual(parsed, {
            "sender": "Modeling Agent",
            "target": "Data Agent",
            "reason": "Model needs more training examples.",
            "detail": "target=20000",
        })

    def test_parse_request_without_detail(self):
        text = message_formatter.format_request("Modeling Agent", "Data Agent", "Need more rows.")
        parsed = message_formatter.parse_request(text)
        self.assertEqual(parsed["detail"], "")

    def test_parse_request_returns_none_for_ordinary_status_update(self):
        text = message_formatter.format_started("Data Agent", "Building dataset pool")
        self.assertIsNone(message_formatter.parse_request(text))

    def test_parse_request_returns_none_for_plain_text(self):
        self.assertIsNone(message_formatter.parse_request("hey, how's it going?"))

    def test_parse_request_returns_none_for_empty_text(self):
        self.assertIsNone(message_formatter.parse_request(""))
        self.assertIsNone(message_formatter.parse_request(None))

    def test_parse_request_requires_bracketed_sender_first_line(self):
        self.assertIsNone(message_formatter.parse_request("REQUEST: Data Agent\nREASON: no sender"))

    def test_parse_request_requires_reason(self):
        self.assertIsNone(message_formatter.parse_request("[Modeling Agent]\nREQUEST: Data Agent"))


class TestParseTargetFromDetail(unittest.TestCase):
    def test_extracts_explicit_target(self):
        self.assertEqual(message_formatter.parse_target_from_detail("target=20000"), 20000)

    def test_extracts_target_with_spaces_and_case_insensitive(self):
        self.assertEqual(message_formatter.parse_target_from_detail("Target = 15000, urgent"), 15000)

    def test_returns_none_when_no_target_stated(self):
        self.assertIsNone(message_formatter.parse_target_from_detail("please hurry"))

    def test_returns_none_for_empty_detail(self):
        self.assertIsNone(message_formatter.parse_target_from_detail(""))


if __name__ == "__main__":
    unittest.main()
