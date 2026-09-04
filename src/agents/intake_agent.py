import json
import re
import requests
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, DEFAULT_MODEL


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

Return ONLY valid JSON with no explanation or markdown."""


def _call_openrouter(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in .env")

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


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

    return structured
