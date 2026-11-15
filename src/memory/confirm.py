"""
Insight memories, step 4 (docs/CLAUDE_BACKEND_PLAN.md, section 18.5):
confidence from repeats.

A lesson from one run is "seen once". When a later run makes a memory about
the same thing, in a compatible scope, it either agrees or disagrees:

  * setting-change insights (step 1) -- each change has a key
    (change_keys: area, step, setting and the direction of the change, e.g.
    "model / XGBClassifier / n_estimators / increase"). Two insights that
    share a key agree when their effect went the same way (better / worse)
    and disagree when it went the other way. An insight whose result came
    with verified leakage is not compared: its effect is not a valid one.
  * reflection lessons (step 3) -- the same agent, a compatible scope, the
    same polarity and mostly the same specific words in the claim
    (CLAIM_OVERLAP) agree; the opposite polarity disagrees. A candidate
    lesson that another run agrees with becomes active.

Scopes are compatible when every trait known on both sides matches: task
type, model families that overlap, and minority share in the same or the
next bucket. Unknown on one side is no objection.

Each memory keeps its record in quality_reasons["confidence"]:
{"agree": [run ids], "disagree": [run ids], "value": agree/(agree+disagree),
"base_quality": ..., "reason": ...}, its own run counted as agreeing. Its
quality is the base quality + AGREE_BONUS per other agreeing run -
DISAGREE_PENALTY per disagreeing run (within 0..1). A disagreement is also
linked (contradicted_by), so recall shows the two together. Nothing is ever
hidden for disagreeing. Writes use fixed op ids, so a retried retain does
not count a run twice. Fail-soft.
"""
import re
import uuid

from src.memory import recall

AGREE_BONUS = 0.1
DISAGREE_PENALTY = 0.2
CLAIM_OVERLAP = 0.5      # word overlap (Jaccard, specific words only) for two lessons to be about the same thing
MINORITY_ORDER = ("<1%", "1-5%", "5-20%", "20-40%", ">=40%")


# --- What two memories are about ---

def _direction(before, after) -> str:
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (before, after))
    if numeric:
        return "increase" if after > before else "decrease"
    if before == "absent":
        return f"set to {after}"
    if after == "absent":
        return "back to default"
    return f"{before} -> {after}"


def change_keys(changes: list) -> set:
    """The keys of a setting-change insight's changes (insights.settings_diff
    items): what was changed and in which direction, without the run's own
    values -- 100 -> 500 trees and 300 -> 3010 trees are the same key."""
    keys = set()
    for c in changes or []:
        if not isinstance(c, dict):
            continue
        call = re.sub(r"#\d+$", "", str(c.get("call") or ""))
        if call == "order":
            before, after = c.get("before") or [], c.get("after") or []
            i = next((i for i, (b, a) in enumerate(zip(before, after)) if b != a), None)
            if i is not None:
                keys.add(("pipeline_order", f"{after[i]} before {before[i]}"))
        elif c.get("param") is None:
            keys.add((c.get("area"), call, "added" if c.get("before") == "absent" else "removed"))
        else:
            keys.add((c.get("area"), call, c["param"], _direction(c.get("before"), c.get("after"))))
    return keys


def compatible(a: dict, b: dict) -> bool:
    """Do two memories' scopes (task features) allow comparing them?"""
    ta, tb = a.get("task") or {}, b.get("task") or {}
    if ta.get("task_type") and tb.get("task_type") and ta["task_type"] != tb["task_type"]:
        return False
    fa, fb = set(ta.get("model_family") or []), set(tb.get("model_family") or [])
    if fa and fb and not fa & fb:
        return False
    ma, mb = ((t.get("dataset_traits") or {}).get("minority_share") for t in (ta, tb))
    if ma in MINORITY_ORDER and mb in MINORITY_ORDER and abs(MINORITY_ORDER.index(ma) - MINORITY_ORDER.index(mb)) > 1:
        return False
    return True


def _content(m: dict) -> dict:
    import json
    try:
        value = json.loads(m.get("content") or "{}")
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


def _rule(m: dict) -> str:
    return (m.get("evidence") or {}).get("rule") or ""


def relation(new: dict, old: dict) -> str | None:
    """"agree", "disagree" or None (not about the same thing) for two
    memories of the same agent from different runs."""
    if new.get("agent") != old.get("agent") or new.get("run_id") == old.get("run_id") or _rule(new) != _rule(old):
        return None
    if not compatible(new, old):
        return None
    if _rule(new) == "setting_change":
        effects = [(m.get("outcome") or {}) for m in (new, old)]
        if any(e.get("valid") is False or e.get("direction") not in ("better", "worse") for e in effects):
            return None
        if not change_keys(_content(new).get("changes")) & change_keys(_content(old).get("changes")):
            return None
        return "agree" if effects[0]["direction"] == effects[1]["direction"] else "disagree"
    if _rule(new) == "reflection":
        claims = [recall._words(_content(m).get("claim") or m.get("title") or "") for m in (new, old)]
        if recall._jaccard(*claims) < CLAIM_OVERLAP:
            return None
        return "agree" if new.get("polarity") == old.get("polarity") else "disagree"
    return None


# --- Recording it ---

def _op(*parts) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "confirm|" + "|".join(str(p) for p in parts)))


def _record(m: dict, other: dict, kind: str) -> tuple[float, dict]:
    """m's quality and reasons after another run (other) agrees or disagrees."""
    reasons = dict(m.get("quality_reasons") or {})
    conf = dict(reasons.get("confidence") or {})
    base = float(conf.get("base_quality", m.get("quality") or 0))
    agree, disagree = set(conf.get("agree") or [m.get("run_id")]), set(conf.get("disagree") or [])
    (agree if kind == "agree" else disagree).add(other.get("run_id"))
    value = len(agree) / (len(agree) + len(disagree))
    quality = round(min(1.0, max(0.0, base + AGREE_BONUS * (len(agree) - 1) - DISAGREE_PENALTY * len(disagree))), 4)
    reasons["confidence"] = {
        "agree": sorted(r for r in agree if r), "disagree": sorted(r for r in disagree if r),
        "value": round(value, 3), "base_quality": base,
        "reason": f"seen in {len(agree) + len(disagree)} run(s): {len(agree)} agree, {len(disagree)} disagree"}
    if kind == "disagree":
        reasons["contradicted_by"] = sorted(set(reasons.get("contradicted_by") or []) | {other["id"]})
    return quality, reasons


def confirm_run(run_id: str, new_memories: list) -> dict:
    """Compare this run's insights and reflection lessons with earlier runs'.
    Returns {"agree": [[new id, old id]], "disagree": [...], "activated": [ids]}."""
    from src import memory
    report = {"agree": [], "disagree": [], "activated": []}
    try:
        new = [m for m in new_memories if _rule(m) in ("setting_change", "reflection") and m.get("run_id") == run_id]
        if not new:
            return report
        earlier = {}
        for agent in sorted({m["agent"] for m in new}):
            earlier[agent] = [m for m in memory.find_memories(agent=agent, kind="lesson", states=("active", "candidate"),
                                                              limit=10_000) if m.get("run_id") != run_id]
        for m in new:
            for old in earlier.get(m["agent"], []):
                kind = relation(m, old)
                if kind is None:
                    continue
                for a, b in ((m, old), (old, m)):
                    quality, reasons = _record(a, b, kind)
                    if memory.set_quality(a["id"], quality, reasons, op_id=_op(run_id, a["id"], b["id"])):
                        a["quality"], a["quality_reasons"] = quality, reasons  # later comparisons in this run build on it
                    memory.increment(a["id"], op_id=_op(run_id, "seen", a["id"], b["id"]), seen_count=1)
                report[kind].append([m["id"], old["id"]])
                if kind == "agree":
                    for a, other in ((m, old), (old, m)):
                        if a.get("state") == "candidate" and memory.set_state(
                                a["id"], "active", f"confirmed by another run ({other.get('run_id')}): "
                                                   f"{a.get('title', '')[:120]}", actor="confirm"):
                            a["state"] = "active"
                            report["activated"].append(a["id"])
        if report["agree"] or report["disagree"]:
            print(f"[Memory] Confidence from repeats: {len(report['agree'])} agreement(s), "
                  f"{len(report['disagree'])} disagreement(s), {len(report['activated'])} lesson(s) activated.")
    except Exception as e:
        print(f"[Memory] WARNING: could not compare run {run_id}'s lessons with earlier runs: {e}")
    return report
