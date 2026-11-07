import io

import numpy as np
import pandas as pd
import pytest

from src.orchestration import dataset_budget as budget


def fraud_frame(n=10_000, fraud_every=50):
    return pd.DataFrame({
        "amount": np.arange(n, dtype=float),
        "is_fraud": [1 if i % fraud_every == 0 else 0 for i in range(n)],
    })


class TestFindLabelColumn:
    def test_matches_target_term_inside_column_name(self):
        df = pd.DataFrame({"Is Fraud?": ["No", "Yes"], "amount": [1, 2]})
        assert budget.find_label_column(df, ["fraud"]) == "Is Fraud?"

    def test_requires_two_values(self):
        df = pd.DataFrame({"fraud_score": [0.1, 0.5, 0.9]})
        assert budget.find_label_column(df, ["fraud"]) is None

    def test_no_terms_no_label(self):
        assert budget.find_label_column(pd.DataFrame({"is_fraud": [0, 1]}), []) is None


class TestBinaryLabelMap:
    def test_yes_no(self):
        assert budget.binary_label_map(["No", "Yes"]) == {"No": 0, "Yes": 1}

    def test_zero_one(self):
        assert budget.binary_label_map([0, 1]) == {0: 0, 1: 1}

    def test_unrecognized_values_are_not_guessed(self):
        assert budget.binary_label_map(["A", "B"]) is None


class TestCheckTargetFit:
    def test_rejects_loan_default_dataset_for_fraud_project(self, tmp_path):
        path = tmp_path / "default.csv"
        pd.DataFrame({"limit_bal": [1, 2], "default.payment.next.month": [0, 1]}).to_csv(path, index=False)
        fit, why = budget.check_target_fit(str(path), ["fraud"], "Default of Credit Card Clients Dataset")
        assert not fit
        assert "predicts something else" in why

    def test_accepts_label_column(self, tmp_path):
        path = tmp_path / "fraud.csv"
        fraud_frame(100).to_csv(path, index=False)
        assert budget.check_target_fit(str(path), ["fraud"])[0]

    def test_accepts_generic_label_when_name_matches(self, tmp_path):
        path = tmp_path / "ulb.csv"
        pd.DataFrame({"amount": [1, 2], "class": [0, 1]}).to_csv(path, index=False)
        assert budget.check_target_fit(str(path), ["fraud"], "Credit Card Fraud Detection")[0]

    def test_not_checked_without_terms(self, tmp_path):
        assert budget.check_target_fit(str(tmp_path / "missing.csv"), [])[0]


class TestStratifiedCap:
    def test_keeps_class_ratio_and_size(self):
        df = fraud_frame(10_000, fraud_every=50)  # 2% fraud
        sampled, info = budget.stratified_cap(df, 1_000, "is_fraud")
        assert len(sampled) == 1_000
        assert sampled["is_fraud"].sum() == 20
        assert info["sampled"] and info["original_rows"] == 10_000

    def test_rare_class_is_never_trimmed_away(self):
        df = fraud_frame(10_000, fraud_every=5_000)  # 2 fraud rows
        sampled, _ = budget.stratified_cap(df, 10, "is_fraud")
        assert len(sampled) == 10
        assert sampled["is_fraud"].sum() >= 1

    def test_under_budget_is_unchanged(self):
        df = fraud_frame(100)
        sampled, info = budget.stratified_cap(df, 1_000, "is_fraud")
        assert sampled is df and not info["sampled"]

    def test_deterministic(self):
        df = fraud_frame(5_000)
        a, _ = budget.stratified_cap(df, 500, "is_fraud")
        b, _ = budget.stratified_cap(df, 500, "is_fraud")
        pd.testing.assert_frame_equal(a, b)


class TestReadCsvCapped:
    def test_chunked_read_is_capped_stratified_and_ordered(self, monkeypatch):
        monkeypatch.setattr(budget, "CHUNK_ROWS", 700)  # force many chunks
        buffer = io.BytesIO(fraud_frame(10_000, fraud_every=50).to_csv(index=False).encode())
        sampled, info = budget.read_csv_capped(buffer, 1_000, ["fraud"])
        assert len(sampled) == 1_000
        assert sampled["is_fraud"].sum() == 20
        assert info["label_column"] == "is_fraud" and info["original_rows"] == 10_000
        assert sampled["amount"].is_monotonic_increasing  # original file order kept

    def test_small_file_read_whole(self):
        buffer = io.BytesIO(fraud_frame(50).to_csv(index=False).encode())
        df, info = budget.read_csv_capped(buffer, 1_000, ["fraud"])
        assert len(df) == 50 and not info["sampled"]
        assert list(df.columns) == ["amount", "is_fraud"]

    def test_python_engine_with_sniffed_separator(self):
        df, _ = budget.read_csv_capped(io.BytesIO(b"a;b\n1;2\n3;4\n"), 10, None, sep=None, engine="python")
        assert list(df.columns) == ["a", "b"] and len(df) == 2
