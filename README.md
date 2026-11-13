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

## Start modeling from a fixed Parent Paper

On the dashboard Overview page, enter the topic and select **Start from my
Parent Paper PDF**. Choose a text-based PDF (up to 50 MB), enter its printed
title, and give the prediction-target keywords, such as `fraud`. Start the
run normally. This skips Intake, Research, and paper selection and begins
Data and Code Discovery in parallel. The supplied paper remains the Parent
Paper throughout the run; Data, Replication, and Evaluation read its exact
uploaded bytes. A missing or changed upload stops the stage instead of
searching for a replacement paper. Scanned PDFs need OCR before upload.

Reuse the same PDF, topic, title, and target for memory-on/off comparisons.
The dashboard records a fixed input ID and PDF SHA-256. The PDF and input
manifest live under `dashboard_files/parent_papers/` and are archived with
each run. Papers and Documents can open the run's archived copy. The
comparison tool groups supplied-paper runs by their exact input ID,
including runs that crash before producing a final result. Dataset choices,
splits, code, models, and resource limits still need to be controlled when
testing modeling improvements. Supplying a paper selects it; it does not
certify peer review or scientific validity.

The saved input can also be reused from the command line:

```powershell
python run.py --parent-paper-input "dashboard_files/parent_papers/<input-id>.json" "machine learning for detecting credit card fraud"
```

## Agent handoffs and parallel work

Agents return a list of requests from `handoff(result, context)`. Master
Agent reviews each request, supervises retries, and schedules independent
first-pass stages concurrently (up to three agents). After Parent Paper
approval, Data and Code Discovery start together. Cleaning starts as soon
as Data is approved; Replication waits for both Cleaning and Code Discovery.
Matching requests from joining branches run once. Missing prerequisites
wait for other requested work, and unresolved or conflicting requests halt
the pipeline. Reruns execute serially to preserve the baseline's files.

Retry notes are scoped to each invocation and explicitly inherited by
Research and Validation workers. Stage reports, run records, and Slack
start/finish messages still identify each agent independently. Project-state
updates happen in the coordinator; Slack data-expansion requests wait until
active branches finish before changing shared data files.

## Agent execution timing

Each agent attempt records its own invocation ID, UTC start/finish times,
and elapsed seconds from a monotonic clock. Execution timing covers the
agent call, including its internal workers, and stops when it returns or
raises. Supervisor checks, reporting, artifact capture, and retry diagnosis
are outside that duration; total run time includes this overhead.

The dashboard shows every concurrent invocation, using the database even
without console output. Finished execution is shown as Checking until the
supervisor records its decision. An unfinished invocation in a stopped or
crashed run is Interrupted, with an unknown final duration. Agent output
details show the selected attempt's measured execution time.

Metrics version 2 sums measured execution durations across each agent's
attempts and reruns. Overlapping durations may sum to more than the whole
run's elapsed time. Old metric versions remain unchanged. Historical
completion-gap estimates are labeled and stored as `agent_seconds_estimate`;
their `agent_seconds` is unknown. Missing finish records also stay unknown,
so they cannot silently enter comparisons as zero or a peer's completion gap.

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
