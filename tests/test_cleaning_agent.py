import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.agents import cleaning_agent


def make_candidate(**overrides) -> dict:
    candidate = {
        "name": "https://example.com/dataset",
        "display_name": "Example Dataset",
        "local_csv_path": "",
    }
    candidate.update(overrides)
    return candidate


class CleaningAgentTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = Path(tempfile.mkdtemp())
        self._processed_patch = patch.object(cleaning_agent, "PROCESSED_DATA_DIR", self._tmpdir / "processed")
        self._conflicts_patch = patch.object(cleaning_agent, "CONFLICTS_LOG", self._tmpdir / "cleaning_conflicts.json")
        self._processed_patch.start()
        self._conflicts_patch.start()

    def tearDown(self):
        self._processed_patch.stop()
        self._conflicts_patch.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_csv(self, name: str, df: pd.DataFrame) -> str:
        path = self._tmpdir / name
        df.to_csv(path, index=False)
        return str(path)


class TestSuccessfulMerge(CleaningAgentTestCase):
    def test_merges_datasets_with_overlapping_columns(self):
        path_a = self._write_csv("a.csv", pd.DataFrame({"Amount": [1, 2], "Label": ["x", "y"]}))
        path_b = self._write_csv("b.csv", pd.DataFrame({"amount": [3], "label": ["z"], "extra": [9]}))

        candidates = [
            make_candidate(name="https://example.com/a", display_name="A", local_csv_path=path_a),
            make_candidate(name="https://example.com/b", display_name="B", local_csv_path=path_b),
        ]

        result = cleaning_agent.run(candidates)

        # DEGRADED, not OK -- no conflicts, but only 3 rows total, well
        # under the 10,000-row target (see TestTargetRows for OK vs
        # DEGRADED specifically).
        self.assertEqual(result["status"], "DEGRADED")
        self.assertFalse(result["target_met"])
        self.assertEqual(result["datasets_merged"], 2)
        self.assertEqual(result["conflicts"], [])
        self.assertTrue(Path(result["master_csv_path"]).exists())

        master = pd.read_csv(result["master_csv_path"])
        # "Amount"/"amount" and "Label"/"label" normalize to the same
        # column, "extra" only exists in B, plus provenance.
        self.assertIn("amount", master.columns)
        self.assertIn("label", master.columns)
        self.assertIn("extra", master.columns)
        self.assertIn(cleaning_agent.SOURCE_COLUMN, master.columns)
        self.assertEqual(len(master), 3)

    def test_provenance_column_records_source_link(self):
        path_a = self._write_csv("a.csv", pd.DataFrame({"x": [1]}))
        candidates = [make_candidate(name="https://example.com/a", local_csv_path=path_a)]

        result = cleaning_agent.run(candidates)
        master = pd.read_csv(result["master_csv_path"])
        self.assertEqual(master[cleaning_agent.SOURCE_COLUMN].iloc[0], "https://example.com/a")

    def test_cross_dataset_duplicate_rows_are_removed(self):
        path_a = self._write_csv("a.csv", pd.DataFrame({"x": [1, 2]}))
        path_b = self._write_csv("b.csv", pd.DataFrame({"x": [1, 3]}))
        candidates = [
            make_candidate(name="https://example.com/a", local_csv_path=path_a),
            make_candidate(name="https://example.com/b", local_csv_path=path_b),
        ]

        result = cleaning_agent.run(candidates)
        # Row (x=1) from A and (x=1) from B differ by _source_dataset, so
        # they are NOT exact duplicates and both survive -- 4 total rows.
        self.assertEqual(result["rows_total"], 4)


class TestTargetRows(CleaningAgentTestCase):
    """The project's >=10,000-entry target is enforced here, on the merged
    master CSV's actual row count -- not per individual dataset (Data
    Agent no longer gates on that, so Kaggle's row-count-free search API
    can be used for discovery)."""

    def test_target_rows_constant_is_10000(self):
        self.assertEqual(cleaning_agent.TARGET_ROWS, 10000)

    def test_status_ok_when_merged_total_meets_target(self):
        big = pd.DataFrame({"amount": range(10000), "label": [0] * 10000})
        path = self._write_csv("big.csv", big)
        result = cleaning_agent.run([make_candidate(name="https://example.com/big", local_csv_path=path)])

        self.assertEqual(result["rows_total"], 10000)
        self.assertTrue(result["target_met"])
        self.assertEqual(result["status"], "OK")

    def test_status_degraded_when_merged_total_is_under_target(self):
        small = pd.DataFrame({"amount": [1, 2, 3]})
        path = self._write_csv("small.csv", small)
        result = cleaning_agent.run([make_candidate(name="https://example.com/small", local_csv_path=path)])

        self.assertFalse(result["target_met"])
        self.assertEqual(result["status"], "DEGRADED")
        self.assertIn("below the 10000-row target", result["notes"])

    def test_status_degraded_when_target_met_but_conflicts_exist(self):
        """Hitting the row target doesn't excuse an unmerged dataset from
        being visible -- conflicts still make the outcome DEGRADED even
        when volume is fine."""
        big = pd.DataFrame({"amount": range(10000)})
        good_path = self._write_csv("big.csv", big)
        candidates = [
            make_candidate(name="https://example.com/big", local_csv_path=good_path),
            make_candidate(name="https://example.com/missing", local_csv_path=""),
        ]
        result = cleaning_agent.run(candidates)

        self.assertTrue(result["target_met"])
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["status"], "DEGRADED")

    def test_not_found_reports_target_not_met(self):
        result = cleaning_agent.run([])
        self.assertFalse(result["target_met"])
        self.assertEqual(result["target_rows"], 10000)


class TestConflicts(CleaningAgentTestCase):
    def test_missing_local_csv_path_is_a_conflict(self):
        candidates = [make_candidate(local_csv_path="")]
        result = cleaning_agent.run(candidates)

        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertIn("no local CSV path", result["conflicts"][0]["reason"])

    def test_unreadable_csv_is_a_conflict_not_a_crash(self):
        bad_path = self._tmpdir / "corrupt.csv"
        bad_path.write_bytes(b"\x00\x01\x02\x03 not a real csv \xff\xfe")
        candidates = [make_candidate(local_csv_path=str(bad_path))]

        result = cleaning_agent.run(candidates)

        # Should not raise; either parses garbage into something empty (a
        # conflict) or fails to parse at all (also a conflict) -- either
        # way nothing gets merged.
        self.assertEqual(result["datasets_merged"], 0)
        self.assertGreaterEqual(len(result["conflicts"]), 1)

    def test_nonexistent_file_is_a_conflict(self):
        candidates = [make_candidate(local_csv_path=str(self._tmpdir / "does_not_exist.csv"))]
        result = cleaning_agent.run(candidates)

        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertIn("could not read CSV", result["conflicts"][0]["reason"])

    def test_zero_column_overlap_is_logged_and_excluded(self):
        path_a = self._write_csv("a.csv", pd.DataFrame({"transaction_amount": [1, 2], "is_fraud": [0, 1]}))
        path_b = self._write_csv("b.csv", pd.DataFrame({"patient_age": [40, 55], "diagnosis": ["a", "b"]}))
        candidates = [
            make_candidate(name="https://example.com/a", display_name="Fraud Data", local_csv_path=path_a),
            make_candidate(name="https://example.com/b", display_name="Medical Data", local_csv_path=path_b),
        ]

        result = cleaning_agent.run(candidates)

        self.assertEqual(result["status"], "DEGRADED")
        self.assertEqual(result["datasets_merged"], 1)
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertEqual(result["conflicts"][0]["dataset"], "Medical Data")
        self.assertIn("schema incompatible", result["conflicts"][0]["reason"])

    def test_conflicts_are_written_to_the_log_file(self):
        candidates = [make_candidate(local_csv_path="")]
        cleaning_agent.run(candidates)

        self.assertTrue(cleaning_agent.CONFLICTS_LOG.exists())
        logged = json.loads(cleaning_agent.CONFLICTS_LOG.read_text())
        self.assertEqual(len(logged), 1)
        self.assertIn("reason", logged[0])

    def test_empty_input_returns_not_found_without_crashing(self):
        result = cleaning_agent.run([])
        self.assertEqual(result["status"], "NOT_FOUND")
        self.assertEqual(result["datasets_merged"], 0)
        self.assertEqual(result["master_csv_path"], "")


class TestPerDatasetCleaning(CleaningAgentTestCase):
    def test_fully_empty_rows_and_columns_are_dropped(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
        cleaned = cleaning_agent._clean_dataframe(df, "link")
        self.assertNotIn("b", cleaned.columns)
        self.assertEqual(len(cleaned), 2)

    def test_within_dataset_duplicate_rows_are_dropped(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        cleaned = cleaning_agent._clean_dataframe(df, "link")
        # Both rows share the same _source_dataset too, so the true
        # duplicate (a=1) collapses to one row.
        self.assertEqual(len(cleaned), 2)

    def test_column_name_normalization_is_deterministic_not_fuzzy(self):
        self.assertEqual(cleaning_agent._normalize_column_name("Transaction Amount"), "transaction_amount")
        self.assertEqual(cleaning_agent._normalize_column_name(" amount-usd "), "amount_usd")
        # Different concepts must NOT be coerced together.
        self.assertNotEqual(
            cleaning_agent._normalize_column_name("Amount"),
            cleaning_agent._normalize_column_name("TransactionAmt"),
        )

    def test_colliding_normalized_names_are_disambiguated(self):
        df = pd.DataFrame([[1, 2]], columns=["Amount", "amount "])
        normalized = cleaning_agent._normalize_columns(df)
        self.assertEqual(len(set(normalized.columns)), 2)


if __name__ == "__main__":
    unittest.main()
