# AI Agents — Multi-Agent Research Pipeline

A multi-agent system that takes a machine learning / data-related research topic and runs it through literature search, paper validation, Parent Paper selection, replication, experimentation, reporting, and presentation generation — coordinated by a Master Agent and observable via Slack.

See `CLAUDE.md` for the full project spec, architecture, and academic requirements.

## Requirements

- Python 3.10+
- API keys for: OpenRouter, SerpentAPI or Brave Search, and (later) Slack

## Setup

### Windows (PowerShell)

```powershell
git clone <repo-url>
cd AI_Agents
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

### macOS / Linux (bash/zsh)

```bash
git clone <repo-url>
cd AI_Agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
```

Fill in `.env` with your own keys:
```
OPENROUTER_API_KEY=
SERPENT_API_KEY=
BRAVE_API_KEY=
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=
```
Slack integration is not wired up yet, so `SLACK_BOT_TOKEN` and `SLACK_CHANNEL_ID` can be left blank for now. `OPENROUTER_API_KEY` and at least one of `SERPENT_API_KEY` / `BRAVE_API_KEY` are required for the agents to run.

## Running

Make sure the virtual environment is activated first (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
# Interactive prompt for a topic
python run.py

# Pass a topic directly
python run.py "machine learning for detecting credit card fraud"

# Reset saved project state and start fresh
python run.py --reset "new topic"
```

## Tests

```bash
pytest

# Run a single test
pytest tests/test_file.py::test_function_name
```

## Deactivating the virtual environment

```bash
deactivate
```

## Project Status

See the "Current Priority" and "Development Philosophy" sections of `CLAUDE.md` for milestone progress. As of now, milestones 1–5 (intake, research, validation, Parent Paper selection, and Master Agent supervision) are complete; Slack status reporting is next.
