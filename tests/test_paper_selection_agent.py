import unittest

from src.agents import paper_selection_agent


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
        rec = paper_selection_agent.run(results)

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
        rec = paper_selection_agent.run(results)

        self.assertTrue(rec["code_availability_audit"]["policy_violation"])
        self.assertEqual(rec["code_availability_audit"]["count_with_code"], 3)
        # The strong no-code paper should still win — code's flat bonus must not dominate.
        self.assertEqual(rec["recommended_parent_paper"]["title"], "No-Code Strong")
        self.assertFalse(rec["code_availability_deciding_factor"])


class TestFallbackBehavior(unittest.TestCase):
    def test_fallback_to_unverified_when_verified_empty(self):
        papers = [make_paper(title=f"Paper {i}", validation_status="UNVERIFIED") for i in range(5)]
        results = validation_results_from(unverified=papers)

        rec = paper_selection_agent.run(results)

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

        rec = paper_selection_agent.run(results)

        self.assertEqual(rec["candidate_pool_source"], "all_non_rejected")
        self.assertTrue(rec["degraded"])
        self.assertIsNotNone(rec["recommended_parent_paper"])
        self.assertEqual(rec["status"], "DEGRADED")

    def test_no_candidates_when_all_rejected(self):
        rejected = [make_paper(title=f"Rejected {i}", validation_status="REJECTED") for i in range(3)]
        results = validation_results_from(rejected=rejected)

        rec = paper_selection_agent.run(results)

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

        rec = paper_selection_agent.run(results)
        row = rec["comparison_table"][0]

        self.assertTrue(row["reproducibility_estimated"])
        self.assertIn("estimated", row["reproducibility_difficulty"].lower())
        self.assertIn("estimate", rec["justification"].lower())


if __name__ == "__main__":
    unittest.main()
