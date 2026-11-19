import json
import re
from src import llm
from src.orchestration.handoff import request as handoff_request


SYSTEM_PROMPT = """You are a research intake assistant for an academic AI research system.
Your job is to analyze a user-provided research topic and produce structured search parameters
that will be used to find peer-reviewed academic papers."""

INTAKE_PROMPT = """Analyze this research topic and return a JSON object with the following fields:

Topic: "{topic}"

Return a JSON object with:
- "domain": the research domain or field (e.g. "financial fraud detection")
- "ml_task": the specific ML or data science task type (e.g. "binary classification", "anomaly detection")
- "keywords": a list of 5-7 academic search keywords
- "search_queries": a list of 3-4 specific search queries suitable for finding peer-reviewed papers on Google Scholar or Semantic Scholar
- "target_terms": 1-3 short lower-case keywords naming the outcome being predicted, as it would appear in a dataset's label column (e.g. ["fraud"] for credit card fraud detection, ["churn"] for customer churn). Name the outcome itself, not the domain: for credit card fraud this is "fraud", not "credit card".

Return ONLY valid JSON with no explanation or markdown."""


PAPER_SYSTEM_PROMPT = """You are a research intake assistant. You read the first part of an academic paper and
report its title and its prediction target exactly as the paper gives them. Never invent or reword a title."""

PAPER_PROMPT = """The user supplied this academic paper as the Parent Paper of the project "{topic}".

START OF THE PAPER (as extracted from the PDF; line breaks and spacing may be imperfect):
\"\"\"
{head}
\"\"\"

PASSAGES ABOUT THE DATA AND THE LABEL:
\"\"\"
{data}
\"\"\"

Return a JSON object with:
- "title": the paper's title, copied exactly as it is printed at the start of the paper (not a section heading,
  journal name or running header). Empty string if you cannot find it.
- "target_terms": 1-3 short lower-case keywords naming the outcome the paper's models predict, as it would appear in
  the dataset's label column (e.g. ["fraud"] for credit card fraud detection, ["churn"] for customer churn). Name the
  outcome itself, not the domain. Use words the paper itself uses.
- "domain": the research domain (e.g. "financial fraud detection").
- "ml_task": the ML task type (e.g. "binary classification").

Return ONLY valid JSON with no explanation or markdown."""

PAPER_HEAD_CHARS = 4000
PAPER_DATA_CHARS = 4000


def term_in_text(term: str, text: str) -> bool:
    """Does the paper's text use this target term? A word start matches, so
    "fraud" is found in "fraudulent"; the words of a multi-word term may be
    separated by spaces, hyphens or underscores."""
    words = [re.escape(w) for w in re.split(r"[\s_-]+", term.strip()) if w]
    return bool(words) and re.search(r"\b" + r"[\s_-]+".join(words), text or "", re.IGNORECASE) is not None


def read_paper(text: str, topic: str = "") -> dict:
    """Read a supplied Parent Paper's title and prediction target from its
    own text. One LLM call proposes them; the code then checks the answer:

      - the title is kept only if it is printed at the start of the paper
        (paper_input.title_in_text, the same check as a typed title);
      - a target term is kept only if the paper's text uses it (term_in_text).

    Returns {"title", "target_terms", "domain", "ml_task", "notes",
    "read_by"}; an empty title or target list means it could not be read
    (Master Agent's check then rejects the attempt)."""
    from src.paper_input import title_in_text
    from src.search.paper_fulltext import DATA_KEYWORDS, excerpt

    prompt = PAPER_PROMPT.format(
        topic=topic, head=(text or "")[:PAPER_HEAD_CHARS],
        data=excerpt(text or "", DATA_KEYWORDS + ("label", "target", "predict", "class"),
                     max_chars=PAPER_DATA_CHARS, head_chars=0))
    answer = llm.parse_json(llm.chat(prompt, system=PAPER_SYSTEM_PROMPT, timeout=120, agent="intake_agent"),
                            default={})
    answer = answer if isinstance(answer, dict) else {}
    notes = []
    title = " ".join(str(answer.get("title") or "").split())
    if title and not title_in_text(title, text):
        notes.append(f"The proposed title '{title}' is not printed at the start of the PDF, so it was not used.")
        title = ""
    terms = answer.get("target_terms") if isinstance(answer.get("target_terms"), list) else []
    terms = list(dict.fromkeys(str(t).strip().lower() for t in terms if str(t).strip()))[:3]
    kept = [t for t in terms if term_in_text(t, text)]
    if len(kept) < len(terms):
        notes.append("Not used, because the paper's text does not use them: "
                     + ", ".join(t for t in terms if t not in kept) + ".")
    return {"title": title, "target_terms": kept, "domain": str(answer.get("domain") or ""),
            "ml_task": str(answer.get("ml_task") or ""), "notes": notes, "read_by": llm.last_served_by()}


def _call_openrouter(prompt: str, topic: str = "") -> str:
    # Past cases (src/memory/recall.py): how earlier runs framed similar topics.
    from src.memory import recall
    return llm.chat(prompt, system=SYSTEM_PROMPT, timeout=120, agent="intake_agent",
                    memory=recall.for_agent("Intake Agent", text=topic, topic=topic) if topic else None)


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON from model response:\n{text}")


def run(topic: str) -> dict:
    """Process a raw user topic into a structured research request."""
    prompt = INTAKE_PROMPT.format(topic=topic)
    raw = _call_openrouter(prompt, topic)
    structured = _extract_json(raw)

    # Guarantee expected keys are present
    structured.setdefault("domain", "")
    structured.setdefault("ml_task", "")
    structured.setdefault("keywords", [])
    structured.setdefault("search_queries", [])
    structured.setdefault("target_terms", [])
    terms = structured["target_terms"] if isinstance(structured["target_terms"], list) else []
    structured["target_terms"] = [str(t).strip().lower() for t in terms if str(t).strip()][:3]

    return structured


def handoff(result: dict, context: dict) -> list:
    """The Intake Agent's request to Master Agent when it finishes: run the
    Research Agent with the structured research request."""
    queries = len((result or {}).get("search_queries") or [])
    return [handoff_request("intake_agent", "Research Agent", {"structured": result},
                            f"Topic analyzed (domain: {(result or {}).get('domain', '')}); {queries} search "
                            "queries ready for the literature search.")]
