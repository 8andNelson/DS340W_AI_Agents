import json
import re
from src import llm


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


def _call_openrouter(prompt: str) -> str:
    return llm.chat(prompt, system=SYSTEM_PROMPT, timeout=120, agent="intake_agent")


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
    raw = _call_openrouter(prompt)
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
