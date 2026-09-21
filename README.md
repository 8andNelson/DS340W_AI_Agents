How to Use This GitHub Repo

  1. Get the code

  git clone https://github.com/8andNelson/DS340W_AI_Agents.git
  cd DS340W_AI_Agents

  If you already have it cloned and just need the latest changes:

  git pull origin main

  2. Create a virtual environment

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
