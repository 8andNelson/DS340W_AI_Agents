import json
from unittest.mock import patch

import pandas as pd

from src.orchestration import column_mapping as cm


class TestToNumeric:
    def test_parses_currency_strings(self):
        assert list(cm.to_numeric(pd.Series(["$1,234.50", "$2.00"]))) == [1234.5, 2.0]

    def test_text_is_not_numeric(self):
        assert cm.to_numeric(pd.Series(["food", "travel"])) is None


class TestVerify:
    def test_same_numeric_within_ratio(self):
        ok, evidence, _ = cm.verify(pd.Series([10.0, 20, 30]), pd.Series([12.0, 18, 33]), cm.RELATION_SAME, None)
        assert ok and "ratio" in evidence

    def test_monthly_vs_annual_income_rejected_as_same(self):
        monthly, annual = pd.Series([3000.0, 4000, 5000]), pd.Series([36000.0, 48000, 60000])
        assert not cm.verify(monthly, annual, cm.RELATION_SAME, None)[0]

    def test_monthly_vs_annual_income_accepted_with_factor(self):
        monthly, annual = pd.Series([3000.0, 4000, 5000]), pd.Series([36000.0, 48000, 60000])
        ok, _, converted = cm.verify(monthly, annual, cm.RELATION_UNIT, 12)
        assert ok and list(converted) == [36000.0, 48000.0, 60000.0]

    def test_unit_conversion_needs_a_factor(self):
        assert not cm.verify(pd.Series([1.0]), pd.Series([12.0]), cm.RELATION_UNIT, None)[0]

    def test_categorical_overlap(self):
        assert cm.verify(pd.Series(["NC", "CA", "TX"]), pd.Series(["CA", "NY", "TX"]), cm.RELATION_SAME, None)[0]

    def test_categorical_disjoint_rejected(self):
        assert not cm.verify(pd.Series(["misc_net", "grocery_pos"]), pd.Series(["Swipe", "Chip"]),
                             cm.RELATION_SAME, None)[0]

    def test_numeric_vs_text_rejected(self):
        assert not cm.verify(pd.Series([5411, 5812]), pd.Series(["food", "travel"]), cm.RELATION_SAME, None)[0]


class TestAlign:
    def anchor(self):
        return pd.DataFrame({"amt": [10.0, 20, 30], "state": ["NC", "CA", "TX"], "zip": [1, 2, 3],
                             "is_fraud": [0, 1, 0]})

    def incoming(self):
        return pd.DataFrame({"amount": ["$11.00", "$19.00", "$31.00"], "merchant_state": ["CA", "TX", "NY"],
                             "zip": ["a", "b", "c"], "is_fraud": [0, 0, 1]})

    def test_verified_mappings_rename_and_convert(self):
        proposals = [{"source": "amount", "target": "amt", "relation": "same", "factor": None},
                     {"source": "merchant_state", "target": "state", "relation": "same", "factor": None}]
        with patch.object(cm, "propose_mappings", return_value=proposals):
            aligned, records = cm.align(self.anchor(), "A", self.incoming(), "B", {"is_fraud"}, "ds2")
        assert list(aligned["amt"]) == [11.0, 19.0, 31.0]
        assert "state" in aligned.columns
        # same name but numeric vs text: not merged, kept apart with a suffix
        assert "zip__ds2" in aligned.columns and "zip" not in aligned.columns
        zip_record = next(r for r in records if r["source"] == "zip")
        assert zip_record["origin"] == "same_name" and not zip_record["accepted"]

    def test_proposal_for_unknown_or_label_column_is_ignored(self):
        proposals = [{"source": "nope", "target": "amt", "relation": "same"},
                     {"source": "is_fraud", "target": "amt", "relation": "same"}]
        with patch.object(cm, "propose_mappings", return_value=proposals):
            _, records = cm.align(self.anchor(), "A", self.incoming(), "B", {"is_fraud"}, "ds2")
        assert {r["source"] for r in records} == {"zip"}

    def test_an_anchor_column_is_mapped_once(self):
        incoming = pd.DataFrame({"amount": [11.0, 19.0], "amt": [10.0, 21.0]})
        proposals = [{"source": "amount", "target": "amt", "relation": "same"}]
        with patch.object(cm, "propose_mappings", return_value=proposals):
            aligned, records = cm.align(self.anchor().drop(columns=["is_fraud"]), "A", incoming, "B", set(), "ds2")
        assert list(aligned.columns).count("amt") == 1
        assert sum(r["accepted"] for r in records) == 1


class TestProposalCache:
    def test_cached_per_pair_and_reused(self):
        content = json.dumps([{"source": "amount", "target": "amt", "relation": "same", "factor": None}])
        anchor, incoming = pd.DataFrame({"amt": [1.0]}), pd.DataFrame({"amount": [1.0]})
        with patch("src.llm.chat", return_value=content) as chat:
            first = cm.propose_mappings(anchor, "A", incoming, "B", set())
            second = cm.propose_mappings(anchor, "A", incoming, "B", set())
        assert first == second and chat.call_count == 1

    def test_failed_call_returns_nothing_and_is_not_cached(self):
        anchor, incoming = pd.DataFrame({"amt": [1.0]}), pd.DataFrame({"amount": [1.0]})
        with patch("src.llm.chat", side_effect=RuntimeError("429")):
            assert cm.propose_mappings(anchor, "A", incoming, "B", set()) == []
        assert not cm.MAPPING_CACHE.exists()
