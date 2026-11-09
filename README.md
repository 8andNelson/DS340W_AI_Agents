 How to Use This GitHub Repo

  1. Get the code

  git clone https://github.com/8andNelson/DS340W_AI_Agents.git

  2. Navigate into the cloned folder and set up a virtual environment

  The .venv folder is intentionally left out of the repo (it's in .gitignore), so it has to be created locally after
  cloning:

  cd DS340W_AI_Agents
  python -m venv .venv

  3. Activate the virtual environment

  Windows (PowerShell):
  .venv\Scripts\Activate.ps1

  macOS/Linux (bash/zsh):
  source .venv/bin/activate

  4. Install dependencies

  pip install -r requirements.txt

  5. Configure secrets

  Windows:
  Copy-Item .env.example .env
  notepad .env

  macOS/Linux:
  cp .env.example .env
  nano .env

  Fill in:
  OPENROUTER_API_KEY=
  SERPENT_API_KEY=
  BRAVE_API_KEY=
  SLACK_BOT_TOKEN=
  SLACK_CHANNEL_ID=

  6. Run the project

  python run.py

  With a topic passed directly:
  python run.py "machine learning for detecting credit card fraud"

  Reset saved project state and start fresh:
  python run.py --reset "new topic"

  7. Run tests

  pytest

  Run a single test:
  pytest tests/test_file.py::test_function_name

  8. When done, deactivate the virtual environment

  deactivate

## Search and paper access

Agents search through Brave first (Brave's own index), then SerpAPI
(Google results) if Brave is unavailable. Set BRAVE_API_KEY and
SERPENT_API_KEY in .env.

Result pages are fetched directly. CAPTCHA, login, and rate-limit pages are
detected even when the server returns HTTP 200. For blocked public pages,
Camoufox runs locally as a fallback, with up to two security-checkbox
interactions. Login pages, rate limits, and authenticated requests stay direct.
Browser content is checked again before it is accepted. If access still fails,
paper validation can seek a verified alternate source through SerpAPI.
Fetch attempts and their providers are recorded in logs/page_fetch_calls.jsonl.

Install the browser once after installing requirements:

```powershell
python -m camoufox fetch
```

No browser API key, subscription, or paid proxy is required. Set
CAMOUFOX_ENABLED=false to disable the fallback, or CAMOUFOX_TIMEOUT_SECONDS
to adjust its default 180-second overall deadline (including browser launch
and downloads). At most two browsers run concurrently; each worker runs in
its own process and its process tree is terminated if the deadline expires.
Browser responses/downloads are limited to 50 MB. The tested versions are
Camoufox Python 0.5.9 and browser 158.0.1-beta.41. Installation needs space
for the browser package; it is not downloaded on an agent request.

Paper validation fetches content rather than trusting HEAD or HTTP status.
It checks the retrieved title or DOI and records reachability evidence,
attempts, the source URL, and LANDING_PAGE versus FULL_TEXT. If the original
cannot be verified, it fetches up to three SerpAPI alternatives and applies
the same checks and prohibited-source rules. A search snippet never proves
paper access. Inaccessible papers remain UNVERIFIED even if the LLM
accepts their metadata. A reachable landing page does not prove full-text
availability or peer review; those are separate checks.

Full-text retrieval follows individual View PDF links and citation PDF metadata
from accessible article HTML. It excludes full-issue/bulk downloads and checks
ScienceDirect links against the requested article's PII. This uses HTTP link
following; when a page is blocked, Camoufox preserves the session while following
the individual article's PDF link. Full-text extraction still verifies the paper
identity, and persistent challenges remain failures. The separately run dashboard
PDF downloader is unchanged.
