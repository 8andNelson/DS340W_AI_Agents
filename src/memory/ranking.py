"""
Base quality of a memory (docs/CLAUDE_BACKEND_PLAN.md, section 9.1):
deterministic, from recorded evidence, with every component and its reason
stored in quality_reasons so a score can always be explained.

Good and neutral memories:
    quality = 0.35 outcome + 0.20 approval + 0.20 validity + 0.15 downstream + 0.10 trust

  outcome     what the agent's part achieved (per agent; see OUTCOME_RULES
              in retain.py). The gap to the paper counts only for a valid
              baseline with a measured gap; the Evaluation Agent is never
              scored by the gap -- real findings whose fix was kept are
              success even when the gap grows (validity is success).
  approval    1 on the first attempt; 0.6 after one failure; 0.3 after more.
  validity    0 when the evaluated baseline had verified leakage (a leaky
              score is not a real score); else 1.
  downstream  1 when the run reached a kept evaluation; 0.5 when it went
              past this agent and stopped later (or crashed: unknown);
              0 when it stopped at this agent.
  trust       1 normal; 0.7 seed; 0.5 low_trust.

Failures (warnings) are not ranked by low quality: a warning is useful
when it has a clear cause and the next attempt passed with the fix.
    usefulness = 0.4 cause + 0.3 fixed + 0.2 specific + 0.1 trust

Use-based learning (section 9.3) is milestone M6; this is the base score.
"""
WEIGHTS = {"outcome": 0.35, "approval": 0.20, "validity": 0.20, "downstream": 0.15, "trust": 0.10}
WARNING_WEIGHTS = {"cause": 0.4, "fixed": 0.3, "specific": 0.2, "trust": 0.1}
TRUST_SCORE = {"normal": 1.0, "seed": 0.7, "low_trust": 0.5}
MAX_GAP_POINTS = 20.0


def approval_score(failures_before: int) -> float:
    return 1.0 if failures_before <= 0 else 0.6 if failures_before == 1 else 0.3


def gap_outcome(gap: float | None) -> float | None:
    """1 at no gap, 0 at MAX_GAP_POINTS or more; None when not measured."""
    return None if gap is None else 1.0 - min(abs(gap), MAX_GAP_POINTS) / MAX_GAP_POINTS


def base_quality(outcome: tuple, approval: tuple, validity: tuple, downstream: tuple, trust: str) -> tuple:
    """Each component is (value 0..1, reason). Returns (quality, reasons)."""
    parts = {"outcome": outcome, "approval": approval, "validity": validity, "downstream": downstream,
             "trust": (TRUST_SCORE[trust], f"trust: {trust}")}
    quality = sum(WEIGHTS[k] * float(v) for k, (v, _) in parts.items())
    reasons = {"formula": "0.35 outcome + 0.20 approval + 0.20 validity + 0.15 downstream + 0.10 trust",
               **{k: {"value": round(float(v), 3), "reason": r} for k, (v, r) in parts.items()}}
    return round(quality, 4), reasons


def warning_usefulness(cause: str | None, fixed: bool | None, signature: str | None, trust: str) -> tuple:
    has_cause = bool(cause) and not cause.lower().startswith(("no cause", "master agent could not diagnose"))
    parts = {
        "cause": (1.0 if has_cause else 0.0, "the failure has a diagnosed cause" if has_cause else "no cause recorded"),
        "fixed": (1.0 if fixed else 0.0, "the next attempt passed" if fixed else
                  "the next attempt also failed" if fixed is False else "no later attempt"),
        "specific": (1.0 if signature else 0.0, f"failure signature: {signature}" if signature else "no signature"),
        "trust": (TRUST_SCORE[trust], f"trust: {trust}"),
    }
    quality = sum(WARNING_WEIGHTS[k] * v for k, (v, _) in parts.items())
    reasons = {"formula": "warning usefulness = 0.4 cause + 0.3 fixed + 0.2 specific + 0.1 trust",
               **{k: {"value": v, "reason": r} for k, (v, r) in parts.items()}}
    return round(quality, 4), reasons
