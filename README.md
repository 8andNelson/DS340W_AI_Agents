# DS340W AI Agents

A multi-agent research pipeline. You give it a machine learning or data
topic. A **Master Agent** then supervises specialized agents that:

1. find at least 5 peer-reviewed papers (2022 to now) and pick one **Parent Paper**,
2. download and merge the Parent Paper's dataset,
3. find the authors' code (or rebuild it from the paper),
4. run the baseline and compare it with the paper's published results,
5. request reruns to close the gap,
6. write a report (`reports/`) and slides (`presentations/`).

A local dashboard shows every agent's progress, the Master Agent's
decisions, and every past run. Slack is not used and needs no setup.

This guide takes you from an empty machine to a working copy. Do the steps in order.

---

## 0. What you need

| Tool | Why | Get it |
|---|---|---|
| **Git** | clone this repo and FreeLLMAPI | https://git-scm.com/downloads |
| **Python 3.11 or newer** (tested on 3.14) | runs the pipeline | https://www.python.org/downloads/ (Windows: select **"Add python.exe to PATH"**) |
| **Docker Desktop** | runs FreeLLMAPI and the sandbox for cloned repositories | https://www.docker.com/products/docker-desktop/ |
| VS Code (optional) | the dashboard's "Open in VS Code" button | https://code.visualstudio.com/ |
| NVIDIA GPU driver (optional) | lets replication scripts train on an NVIDIA GPU | https://www.nvidia.com/Download/index.aspx |

About 5 GB of free disk space is enough for Docker images, the Camoufox
browser, and datasets. With an NVIDIA GPU, allow about 6 GB more for
PyTorch's CUDA build (see "GPU (optional)" below).

### GPU (optional)

**No NVIDIA GPU? Do nothing.** Everything runs on the CPU, and you do not
need CUDA. Do not install CUDA or a CUDA build of PyTorch yourself.

PyTorch is not in `requirements.txt` on purpose. pip cannot choose a build by
hardware, so the Replication Agent installs PyTorch itself, inside each run's
own environment, only when a script needs it:

| This computer | What the Replication Agent installs |
|---|---|
| No NVIDIA GPU | PyTorch's normal CPU build (from PyPI) |
| NVIDIA GPU with a recent driver | PyTorch's CUDA build from download.pytorch.org (2–3 GB the first time; pip caches it), chosen to match the driver |
| CUDA install fails | falls back to the CPU build, and says so |

- With an NVIDIA GPU you only need the **driver**. The PyTorch CUDA build
  includes the CUDA libraries, so do not install the CUDA Toolkit.
- To skip the CUDA download even with a GPU, set `TORCH_CUDA=off` in `.env`.
- Scripts choose the device when they run (`cuda` if available, else
  `cpu`), so a script made on a GPU computer also runs on one without.
- XGBoost and CatBoost use an NVIDIA GPU without extra installs. TensorFlow
  cannot use a GPU on native Windows, so neural networks are written in
  PyTorch where the paper allows it.
- Replications have no time limit. Before running, the Replication Agent
  checks this computer's RAM, GPU and disk, and stops early, with the
  reason, if the paper's method needs more than this computer has. A run
  that stops working (no CPU, disk, GPU or output for 15 minutes) is
  stopped as hung.

---

## 1. Install Docker Desktop

1. Download and run the installer from https://www.docker.com/products/docker-desktop/.
   - Windows: keep **"Use WSL 2 instead of Hyper-V"** selected. If Windows asks
     for WSL, run `wsl --install` in an administrator PowerShell, then restart.
     (Guide: https://docs.docker.com/desktop/setup/install/windows-install/)
   - macOS: https://docs.docker.com/desktop/setup/install/mac-install/
   - Linux: https://docs.docker.com/desktop/setup/install/linux/
2. Start Docker Desktop and wait until it shows **"Engine running"**.
3. Check it in a new terminal:

   ```powershell
   docker --version
   docker run --rm hello-world
   ```

   You should see "Hello from Docker!".

Docker is used two ways:

- **FreeLLMAPI** (step 3) runs as a Docker container.
- **Repository sandbox:** when the pipeline finds the authors' code, it
  installs and runs it only inside a locked-down Docker container
  (`python:<version>-slim`, no network at run time, no secrets). Without
  Docker, the pipeline never runs a cloned repository; it rebuilds the
  baseline from the paper instead.

---

## 2. Clone this repo and set up Python

```powershell
git clone https://github.com/8andNelson/DS340W_AI_Agents.git
cd DS340W_AI_Agents
python -m venv .venv
```

Windows: a short folder is best, for example `C:\dev\DS340W_AI_Agents` or
your Desktop. Windows cannot load files from paths longer than 259
characters. When the project's folder is deep, the Replication Agent puts
its venvs in a short folder under `%LOCALAPPDATA%\ds340w_venvs` by itself, so you
do not have to do anything.

Activate the virtual environment (`.venv` is gitignored, so each person makes their own):

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

If PowerShell refuses to run the script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Install the packages and the Camoufox browser (a free local browser for
pages that block plain requests; it is downloaded once):

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m camoufox fetch
```

---

## 3. Set up FreeLLMAPI (free LLM router)

[FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) is a self-hosted,
OpenAI-compatible router. It pools the free tiers of many LLM providers and
switches to another provider on rate limits. This is the recommended
provider for full runs, because one run makes many LLM calls.

Clone it **outside** this repo (for example next to it), then make its
`.env` with an encryption key and start it.

**Windows (PowerShell):**

```powershell
cd ..
git clone https://github.com/tashfeenahmed/freellmapi.git
cd freellmapi
$Bytes = New-Object Byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($Bytes)
$KEY = -join ($Bytes | ForEach-Object { "{0:x2}" -f $_ })
[IO.File]::WriteAllText("$PWD\.env", "ENCRYPTION_KEY=$KEY`nPORT=3001`n")
docker compose up -d
```

(`WriteAllText` writes the file without a byte-order mark, so the first
variable is read correctly.)

**macOS / Linux:**

```bash
cd ..
git clone https://github.com/tashfeenahmed/freellmapi.git
cd freellmapi
printf "ENCRYPTION_KEY=%s\nPORT=3001\n" "$(openssl rand -hex 32)" > .env
docker compose up -d
```

Then:

1. Open http://localhost:3001 in a browser.
2. On the **Keys** page, add free API keys for the providers it lists.
   Each provider's page links to where you sign up. More providers means
   fewer rate-limit stops.
3. Copy the **unified API key** shown in the header of the Keys page. It
   goes in this repo's `.env` as `FREELLMAPI_API_KEY` (step 5).
4. Note the full path of the `freellmapi` folder (for example
   `C:\Users\you\Desktop\freellmapi`). It goes in `.env` as
   `FREELLMAPI_COMPOSE_DIR`, so the dashboard can start and stop the router for you.

Useful commands (run in the `freellmapi` folder):

```powershell
docker compose ps                   # is it running?
docker compose logs -f freellmapi   # server log
docker compose down                 # stop it (keeps your keys)
```

Do **not** run `docker compose down -v` unless you want to delete all its saved keys.

Install guide: https://github.com/tashfeenahmed/freellmapi/blob/main/docs/en/install/01-install.md

Go back to this repo when you are done: `cd ..\DS340W_AI_Agents`

---

## 4. Get the API keys

Only the LLM provider and one search key are required. The rest are optional.

| Variable | Required? | What it is | Where to get it |
|---|---|---|---|
| `FREELLMAPI_API_KEY` | **Yes** (if `LLM_PROVIDER=freellmapi`) | unified key from FreeLLMAPI | Keys page at http://localhost:3001 (step 3) |
| `OPENROUTER_API_KEY` | **Yes** (if `LLM_PROVIDER=openrouter`) | OpenRouter key | https://openrouter.ai/settings/keys |
| `ANTHROPIC_API_KEY` | only for `LLM_PROVIDER=anthropic` | Claude API key | https://console.anthropic.com/settings/keys |
| `BRAVE_API_KEY` | **Yes** (or SerpApi) | Brave Search API, first search provider | https://brave.com/search/api/ → keys at https://api-dashboard.search.brave.com/app/keys |
| `SERPENT_API_KEY` | Recommended | SerpApi (Google results), used when Brave fails or hits its limit | https://serpapi.com/manage-api-key |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Optional | lets the Data Agent search and download Kaggle datasets | https://www.kaggle.com/settings → **API** → **Create New Token** (the downloaded `kaggle.json` has both values) |

Free-tier limits to know:

- **OpenRouter** without credits: 50 free-model requests per day. One full
  run makes about 30–60 requests, so the free tier may not finish a run.
  This is why FreeLLMAPI is recommended.
- **Brave Search:** 1,000 requests per month, reset on the 1st.

Slack needs no keys and no setup: it is off, and the dashboard shows all
agent progress.

---

## 5. Make your `.env` file

Copy the template. `.env` is gitignored, so your keys are never committed.

```powershell
# Windows
Copy-Item .env.example .env
notepad .env
```

```bash
# macOS / Linux
cp .env.example .env
nano .env
```

Minimum working `.env` with FreeLLMAPI (recommended):

```ini
LLM_PROVIDER=freellmapi
FREELLMAPI_API_KEY=<unified key from the FreeLLMAPI Keys page>
FREELLMAPI_BASE_URL=http://localhost:3001/v1
FREELLMAPI_COMPOSE_DIR=C:\Users\you\Desktop\freellmapi
BRAVE_API_KEY=<your Brave key>
SERPENT_API_KEY=<your SerpApi key>
```

Or with OpenRouter instead:

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=<your OpenRouter key>
BRAVE_API_KEY=<your Brave key>
SERPENT_API_KEY=<your SerpApi key>
```

Leave the other values at their defaults. `.env.example` explains each one.
Useful optional settings:

- `MEMORY_CONTRIBUTOR=<your short name>`: recorded with each run's metrics.
- `MEMORY_BACKEND=sqlite` (default, local only), `postgres` (the team's
  hosted store; needs `MEMORY_DATABASE_URL` from the team), or `off`.
- `AGENTS_OFF=experiment,report,presentation`: skip optional agents for a faster test run.
- `CAMOUFOX_ENABLED=false`: turn off the browser fallback.

Never put real keys in `.env.example`, code, or commits.

---

## 6. Check the install

Check that the packages and the pipeline's modules load. This makes no
network or LLM calls:

```powershell
python -c "import src.main, src.dashboard.server, camoufox; print('install OK')"
```

Check that FreeLLMAPI answers (if you use it):

```powershell
curl.exe http://localhost:3001/api/ping
```

---

## 7. Start the dashboard

```powershell
python -m src.dashboard
```

Open http://127.0.0.1:8050. With `LLM_PROVIDER=freellmapi`, the dashboard
starts Docker Desktop and the FreeLLMAPI router for you (from
`FREELLMAPI_COMPOSE_DIR`) and stops the router when you press Ctrl+C,
unless a run is still active.

Options:

```powershell
python -m src.dashboard --port 8060          # other port
python -m src.dashboard --no-freellmapi      # do not start/stop the router
python -m src.dashboard --keep-freellmapi    # do not stop the router on exit
```

The dashboard lets you enter a topic and start a run, watch each agent live,
and open the papers, datasets, scripts, report, and slides of every past run.
It listens only on 127.0.0.1.

---

## 8. Run the pipeline

From the dashboard (Overview page), or from the command line:

```powershell
python run.py                                                      # asks for a topic
python run.py "machine learning for detecting credit card fraud"   # topic as argument
python run.py --reset "new topic"                                  # clear saved state first
python run.py --agents-off report,presentation "topic"             # skip optional agents
```

A full run can take a long time (paper search, dataset downloads,
training, reruns). It runs unattended. Only one run at a time: two runs
overwrite each other's files.

Outputs:

| Where | What |
|---|---|
| `reports/<slug>/report.md` | final report, with `figures/*.svg` |
| `presentations/<slug>/slides.pptx` | slide deck |
| `data/processed/master_dataset.csv` | merged dataset and train/test/validation splits |
| `replication/<slug>/` | baseline code, `ENVIRONMENT.md`, `pip_freeze.txt`, run output (the baseline's venv is deleted when the run ends) |
| `logs/` | project state, Master Agent log, per-attempt agent reports, LLM and search call logs |
| `database/project_runs.db` | permanent record of every run |

All of these are gitignored.

Inspect past runs from the command line:

```powershell
python -m src.orchestration.run_store list
python -m src.orchestration.run_store show <run_id>
```

---

## 9. Your working copy: final checklist

You have a working copy when all of these are true:

- [ ] `docker run --rm hello-world` prints "Hello from Docker!"
- [ ] `.venv` is active and `pip install -r requirements.txt` finished without errors
- [ ] `python -m camoufox fetch` finished
- [ ] http://localhost:3001 opens, and the Keys page has at least one provider key (FreeLLMAPI users)
- [ ] `.env` has `LLM_PROVIDER`, its key, and `BRAVE_API_KEY` and/or `SERPENT_API_KEY`
- [ ] the install check in step 6 prints `install OK`
- [ ] `python -m src.dashboard` opens at http://127.0.0.1:8050
- [ ] a test run, for example
      `python run.py --reset --agents-off experiment,report,presentation "machine learning for detecting credit card fraud"`,
      gets past Research and Validation and approves a Parent Paper

When you are done working:

```powershell
deactivate
```

To update later: `git pull`, then `pip install -r requirements.txt` again.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `docker: command not found` / engine not running | start Docker Desktop and wait for "Engine running"; open a new terminal |
| dashboard says the router is down | check `FREELLMAPI_COMPOSE_DIR`, then run `docker compose up -d` in that folder |
| LLM calls fail with 401 | wrong `FREELLMAPI_API_KEY` / `OPENROUTER_API_KEY`; copy it again |
| `DailyQuotaExceeded` | OpenRouter free limit reached; wait for the reset or switch to FreeLLMAPI |
| every search fails | check `BRAVE_API_KEY` and `SERPENT_API_KEY`; Brave resets monthly |
| Kaggle datasets skipped | set `KAGGLE_USERNAME` and `KAGGLE_KEY` |
| a found repository is not run | Docker is not running; the baseline is rebuilt from the paper instead |

---

## More details

### Start from your own Parent Paper

On the dashboard Overview page, enter the topic and select **Start from my
Parent Paper PDF**. Choose a text-based PDF (up to 50 MB), enter its printed
title, and give the prediction-target keywords, such as `fraud`. This skips
Intake, Research, and paper selection, and starts Data and Code Discovery in
parallel. The supplied paper stays the Parent Paper for the whole run.
Scanned PDFs need OCR before upload. Supplying a paper selects it; it does
not certify peer review or scientific validity.

The saved input (under `dashboard_files/parent_papers/`) can be reused from
the command line, for example for memory-on/off comparisons with the same paper:

```powershell
python run.py --parent-paper-input "dashboard_files/parent_papers/<input-id>.json" "machine learning for detecting credit card fraud"
```

### How the agents work together

Agents never run each other. Each agent returns a request (`handoff`) to
the Master Agent, which checks it, runs the next agent, checks its output,
and on a failure diagnoses it and retries (up to 5 attempts). Independent
stages run in parallel (up to three agents): after the Parent Paper is
approved, Data and Code Discovery start together; Replication waits for
Cleaning and Code Discovery. Reruns requested by the Evaluation and
Experiment Agents run one at a time, so the known-good baseline is
always kept. Each agent attempt records its own start and finish time.

### Search and paper access

Searches go to Brave first, then SerpApi. Result pages are fetched directly;
CAPTCHA, login, and rate-limit pages are detected even with HTTP 200. Blocked
public pages can use the local Camoufox browser (no key, proxy, or
subscription). Fetch attempts are logged in `logs/page_fetch_calls.jsonl`.
A search snippet is never accepted as paper access, and a paper that cannot
be verified stays `UNVERIFIED`.

### Project documentation

- `docs/PROJECT_STATUS.md`: what is built and what is open
- `docs/CLAUDE_BACKEND_PLAN.md`: memory and metrics design
