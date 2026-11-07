import unittest
from unittest.mock import patch

import pytest

from src.agents import validation_agent


def make_paper(**overrides) -> dict:
    """Build a synthetic validated paper dict with sane defaults."""
    paper = {
        "title": "Untitled Paper",
        "authors": [],
        "year": 2023,
        "venue": "Some Conference",
        "peer_reviewed": True,
        "doi": "",
        "paper_url": "https://doi.org/10.1000/example",
        "abstract": "",
        "dataset": "",
        "dataset_source": "",
        "methodology": "",
        "results_summary": "",
        "has_tables": False,
        "has_figures": False,
        "has_graphs": False,
        "has_code": False,
        "code_url": "",
        "citation_count": 0,
        "parent_paper_score": 5,
        "notes": "",
        "source": "brave_search",
        "verified": False,
        "validation_status": "VERIFIED",
        "rejection_reason": "",
        "url_reachable": True,
        "validated_peer_reviewed": True,
        "venue_credible": True,
        "year_valid": True,
        "validated_has_dataset": False,
        "validated_has_results": False,
        "validated_reproducibility": 3,
        "validation_notes": "",
    }
    paper.update(overrides)
    return paper


def validation_results_from(verified=None, unverified=None, rejected=None, all_papers=None) -> dict:
    verified = verified or []
    unverified = unverified or []
    rejected = rejected or []
    return {
        "verified": verified,
        "unverified": unverified,
        "rejected": rejected,
        "all": all_papers if all_papers is not None else verified + unverified + rejected,
    }


class TestDeterministicRanking(unittest.TestCase):
    def test_prefers_higher_reproducibility_and_clarity(self):
        strong = make_paper(
            title="Strong Candidate",
            dataset="CIFAR-10",
            dataset_source="Official benchmark",
            methodology=(
                "We describe the model architecture, hyperparameter search, training "
                "procedure, preprocessing steps, feature engineering, baseline comparisons, "
                "and evaluation metrics in detail."
            ),
            results_summary="Achieved 94.2% accuracy, F1 0.91, ablation study included.",
            has_tables=True,
            has_figures=True,
            has_graphs=True,
            validated_reproducibility=5,
            parent_paper_score=9,
        )
        weak = make_paper(
            title="Weak Candidate",
            methodology="We used a model.",
            validated_reproducibility=1,
            parent_paper_score=2,
        )

        results = validation_results_from(verified=[weak, strong])
        rec = validation_agent.rank_and_recommend(results)

        self.assertEqual(rec["status"], "OK")
        self.assertEqual(rec["recommended_parent_paper"]["title"], "Strong Candidate")

    def test_code_availability_audit_flags_more_than_one_and_does_not_dominate(self):
        no_code_strong = make_paper(
            title="No-Code Strong",
            dataset="ImageNet",
            dataset_source="Official",
            methodology="Full architecture, hyperparameters, training procedure, baselines described.",
            results_summary="Top-1 accuracy 82%.",
            has_tables=True,
            has_figures=True,
            validated_reproducibility=5,
            has_code=False,
        )
        code_weak_1 = make_paper(title="Code Weak 1", methodology="Brief.", validated_reproducibility=1, has_code=True)
        code_weak_2 = make_paper(title="Code Weak 2", methodology="Brief.", validated_reproducibility=1, has_code=True)
        code_weak_3 = make_paper(title="Code Weak 3", methodology="Brief.", validated_reproducibility=1, has_code=True)

        results = validation_results_from(verified=[code_weak_1, code_weak_2, code_weak_3, no_code_strong])
        rec = validation_agent.rank_and_recommend(results)

        self.assertTrue(rec["code_availability_audit"]["policy_violation"])
        self.assertEqual(rec["code_availability_audit"]["count_with_code"], 3)
        # The strong no-code paper should still win — code's flat bonus must not dominate.
        self.assertEqual(rec["recommended_parent_paper"]["title"], "No-Code Strong")
        self.assertFalse(rec["code_availability_deciding_factor"])


class TestFallbackBehavior(unittest.TestCase):
    def test_fallback_to_unverified_when_verified_empty(self):
        papers = [make_paper(title=f"Paper {i}", validation_status="UNVERIFIED") for i in range(5)]
        results = validation_results_from(unverified=papers)

        rec = validation_agent.rank_and_recommend(results)

        self.assertEqual(rec["candidate_pool_source"], "unverified")
        self.assertTrue(rec["degraded"])
        self.assertIsNotNone(rec["recommended_parent_paper"])

    def test_fallback_to_all_non_rejected_when_stuck_pending(self):
        # Reproduces the real observed validation_agent failure mode: every
        # paper stuck at validation_status "PENDING", all buckets empty.
        pending_papers = [
            make_paper(
                title=f"Pending Paper {i}",
                validation_status="PENDING",
                validated_reproducibility=0,
                validation_notes="LLM assessment unavailable",
            )
            for i in range(15)
        ]
        results = validation_results_from(all_papers=pending_papers)

        rec = validation_agent.rank_and_recommend(results)

        self.assertEqual(rec["candidate_pool_source"], "all_non_rejected")
        self.assertTrue(rec["degraded"])
        self.assertIsNotNone(rec["recommended_parent_paper"])
        self.assertEqual(rec["status"], "DEGRADED")

    def test_no_candidates_when_all_rejected(self):
        rejected = [make_paper(title=f"Rejected {i}", validation_status="REJECTED") for i in range(3)]
        results = validation_results_from(rejected=rejected)

        rec = validation_agent.rank_and_recommend(results)

        self.assertEqual(rec["status"], "NO_CANDIDATES")
        self.assertIsNone(rec["recommended_parent_paper"])


class TestNoHallucination(unittest.TestCase):
    def test_reproducibility_estimate_not_confused_with_validated_score(self):
        # validated_reproducibility=0 means Layer 3 never actually scored it.
        paper = make_paper(
            title="Unscored Paper",
            validated_reproducibility=0,
            dataset="Some Dataset",
            methodology="Some methodology.",
            results_summary="Some results.",
            has_code=True,
        )
        results = validation_results_from(verified=[paper])

        rec = validation_agent.rank_and_recommend(results)
        row = rec["comparison_table"][0]

        self.assertTrue(row["reproducibility_estimated"])
        self.assertIn("estimated", row["reproducibility_difficulty"].lower())
        self.assertIn("estimate", rec["justification"].lower())


class TestDatasetNaming(unittest.TestCase):
    """A Parent Paper must clearly identify its dataset -- a generic name
    that only restates the topic doesn't count."""

    def test_generic_names_are_not_specific(self):
        cases = [
            ("real-world data", "AI-based credit card fraud detection on real-world data"),
            ("online credit card data", "Anomaly detection in online credit card data using GNNs"),
            ("", "Any Paper"),
            ("public transaction dataset", "Fraud detection"),
        ]
        for dataset, title in cases:
            self.assertFalse(validation_agent._is_specific_dataset_name(dataset, title), dataset)

    def test_real_names_are_specific(self):
        cases = [
            ("European cardholders dataset", "Credit card fraud detection"),
            ("IEEE-CIS Fraud Detection", "Credit card fraud detection"),
            ("PaySim", "Mobile money fraud"),
        ]
        for dataset, title in cases:
            self.assertTrue(validation_agent._is_specific_dataset_name(dataset, title), dataset)

    def test_generic_name_alone_is_not_specified(self):
        paper = make_paper(title="Fraud detection on real-world data", dataset="real-world data")
        self.assertEqual(validation_agent._dataset_availability(paper), "Not specified")

    def test_named_dataset_outranks_otherwise_identical_generic_one(self):
        generic = make_paper(title="Fraud detection on real-world data", dataset="real-world data")
        named = make_paper(title="Fraud detection study", dataset="European cardholders dataset")
        rec = validation_agent.rank_and_recommend(validation_results_from(verified=[generic, named]))
        self.assertEqual(rec["recommended_parent_paper"]["title"], "Fraud detection study")
        self.assertEqual(rec["dataset_warning"], "")

    def test_warns_when_no_candidate_names_a_dataset(self):
        papers = [make_paper(title=f"Paper {i}", dataset="") for i in range(3)]
        rec = validation_agent.rank_and_recommend(validation_results_from(verified=papers))
        self.assertIsNotNone(rec["recommended_parent_paper"])
        self.assertIn("No candidate paper names a specific dataset", rec["dataset_warning"])


if __name__ == "__main__":
    unittest.main()


class TestValidationRunLayers:
    """validation_agent.run()'s three validation layers (concurrent URL
    checks and LLM batches, retraction rejection)."""

    @pytest.fixture(autouse=True)
    def no_openalex(self):
        with patch("src.search.openalex.work_by_doi", return_value=None),              patch("src.search.openalex.match_title", return_value=None):
            yield

    def _paper(self, title, url):
        return {"title": title, "paper_url": url, "abstract": "", "year": 2023,
                "peer_reviewed": True, "dataset": "", "results_summary": ""}

    def test_rejects_retracted_paper_in_layer_1(self):
        papers = [self._paper("Retracted: Fraud detection", "https://ieeexplore.ieee.org/document/1")]
        with patch("src.agents.validation_agent._check_url_reachable") as reach, \
             patch("src.agents.validation_agent._llm_validate_batch") as batch:
            result = validation_agent.run(papers)
        assert result["rejected"][0]["rejection_reason"] == "Retracted paper"
        reach.assert_not_called()
        batch.assert_not_called()

    def test_every_paper_is_checked_and_none_stays_pending(self):
        papers = [self._paper(f"Paper {i}", f"https://doi.org/10.1/{i}") for i in range(6)]

        def assess(batch):
            # Mark only the first paper of each batch; the rest are "omitted".
            batch[0]["validation_status"] = "VERIFIED"
            return batch

        with patch("src.agents.validation_agent._check_url_reachable", return_value=(True, 200)) as reach, \
             patch("src.agents.validation_agent._llm_validate_batch", side_effect=assess):
            result = validation_agent.run(papers)

        assert reach.call_count == 6
        assert all(p["url_reachable"] for p in papers)
        assert len(result["verified"]) == 2  # 6 papers -> batches of 4 and 2
        assert len(result["unverified"]) == 4
        assert not [p for p in papers if p["validation_status"] == "PENDING"]

    def test_failed_llm_batch_falls_back_to_unverified(self):
        papers = [self._paper("Paper", "https://doi.org/10.1/x")]
        with patch("src.agents.validation_agent._check_url_reachable", return_value=(True, 200)), \
             patch("src.agents.validation_agent._llm_validate_batch", side_effect=RuntimeError("429")):
            result = validation_agent.run(papers)
        assert result["unverified"][0]["validation_notes"] == "LLM assessment unavailable"


class TestAttachOfficialRecord:
    def test_attaches_abstract_on_title_match(self):
        paper = {"title": "Fraud Model", "doi": "10.1/x", "abstract": "Their approach yielded 99%."}
        record = {"title": "Fraud model", "abstract": "We propose a model.", "is_retracted": False, "doi": "10.1/x"}
        with patch("src.search.openalex.work_by_doi", return_value=record):
            validation_agent._attach_official_record(paper)
        assert paper["official_abstract"] == "We propose a model."
        assert paper["abstract_source"] == "openalex"
        assert paper["openalex_retracted"] is False

    def test_rejects_a_different_paper(self):
        paper = {"title": "Fraud Model", "doi": ""}
        with patch("src.search.openalex.work_by_doi", return_value=None),              patch("src.search.openalex.match_title", return_value={"title": "Other", "abstract": "x",
                                                                    "is_retracted": False}):
            validation_agent._attach_official_record(paper)
        assert "official_abstract" not in paper
        assert paper["abstract_source"] == "search_snippet"
