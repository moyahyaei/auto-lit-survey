# Recent Publications Digest (OpenAlex + OneMine + Gemini)

A lightweight pipeline that:
- Collects recent academic publications from **OpenAlex** (journals/sources)
- Collects recent repository items from **OneMine** (search pages)
- Deduplicates items and exports a **raw CSV**
- Uses **Gemini** to triage items (e.g., **Must read / Good to read / Not sure**)
- Produces an **HTML email digest** and sends it via Gmail SMTP (with CSV attachments)

> This repo is designed to be **personalised per user**. Each user must provide their own **Gemini API key** and **Gmail credentials** (via a local `.env` file or GitHub Secrets).

---

## What this produces

Each run writes outputs to `./output/`:

- `raw_academic_repository_YYYY-MM-DD.csv`  
  Raw items: title, authors, abstract, URL, source, date, etc.

- `triaged_academic_repository_YYYY-MM-DD.csv`  
  Raw + Gemini classification (and brief rationale if enabled)

- `digest_YYYY-MM-DD.html`  
  Rendered digest HTML (also emailed)

---

## Recommended repository layout

```text
.
├─ daily_digest.py          # main script (your filename may differ)
├─ config.yaml              # configuration (YAML supports comments)
├─ sources.csv              # journals + repositories (+ optional RSS feeds)
├─ keywords.csv             # topics/keywords used for relevance triage
├─ requirements.txt
├─ .env.example
├─ .gitignore
└─ output/                  # generated files (should be gitignored)
```

**Tip:** avoid keeping many “versioned” script files in the repo (e.g., `daily_digest_GPTV6_yaml.py`).  
Keep one “current” entrypoint like `daily_digest.py`. Git already stores your history.

---

## Quick start (run locally)

### 1) Create a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Create your `.env`
Copy `.env.example` → `.env` and fill in your secrets:

```bash
# macOS/Linux
cp .env.example .env
```

On Windows, copy the file manually or:
```powershell
Copy-Item .env.example .env
```

### 4) Run
```bash
python daily_digest.py
```

> If your main file is named differently, run that filename instead.

---

## Personalising the pipeline

You will typically edit these files:

- `.env` → credentials (private, never commit)
- `config.yaml` → run behaviour (lookback window, rate limiting, etc.)
- `sources.csv` → which journals/repositories you scan
- `keywords.csv` → what topics/keywords count as “relevant” to you

---

## Secrets and credentials (`.env`)

This project expects the following environment variables.

Create a `.env` file (NOT committed) based on `.env.example`.

### Required
- `GEMINI_API_KEY`  
  Your Google Gemini API key

- `EMAIL_SENDER`  
  Your Gmail address used to send the digest (SMTP)

- `EMAIL_PASSWORD`  
  A Gmail **App Password** (recommended)  
  ⚠️ Do NOT use your normal Gmail password.

- `EMAIL_RECEIVER`  
  Where to send the digest (can be the same as `EMAIL_SENDER`)

### Optional but recommended
- `OPENALEX_MAILTO`  
  Contact email for OpenAlex “polite pool” usage

---

## Configuration (`config.yaml`)

YAML is recommended over JSON because YAML supports comments.

Example:

```yaml
# Configuration for the Recent Publications Digest
lookback_days: 7                   # How many days into the past to search (OpenAlex + RSS)
max_items_per_feed: 5              # Max items per source per run

email_subject: "Recent Publications Digest"

# Gemini API Rate Limiting
gemini_min_interval_seconds: 2.0   # Min delay between Gemini calls
gemini_error_sleep_seconds: 10.0   # Extra delay after an error (backoff)

# Repositories (OneMine)
repository_extra_days: 30          # Repo lookback becomes (lookback_days + repository_extra_days)
```

### Important behaviour: repository window
Repositories are scraped with a longer time window by design:

- Academic sources (OpenAlex, RSS): **lookback_days**
- Repositories (OneMine): **lookback_days + repository_extra_days**

Example:
- `lookback_days: 7`
- `repository_extra_days: 30`
→ OneMine lookback window is **37 days**

This is intentional because repository pages often lag or list fewer items in the last week.

---

## Sources (`sources.csv`)

This file defines **what** you scan.

Suggested columns:
- `name` → label shown in the digest
- `type` → one of: `journal`, `repository`, `rss`
- `identifier` → meaning depends on type

Example:

```csv
name,type,identifier
Minerals Engineering,journal,Minerals Engineering
Hydrometallurgy,journal,Hydrometallurgy
OneMine (global),repository,onemine:
SME Annual Conference,repository,onemine:keywords=SME%20Annual%20Conference&searchfield=all
```

### Notes
- For `journal`, the identifier is the journal/source name used for OpenAlex resolution.
- For OneMine, the identifier supports:
  - `onemine:` for global “most recent”
  - `onemine:keywords=...&searchfield=all` for targeted searches  
    (URL-encode spaces as `%20`)

---

## Keywords (`keywords.csv`)

This file defines your “areas of interest” for triage.

Suggested columns:
- `topic` → label (e.g., Flotation, Comminution, Control)
- `keywords` → semicolon-separated list

Example:

```csv
topic,keywords
Flotation,flotation; froth; collector; depressant; activation
Control,MPC; model predictive control; fault detection; soft sensor; anomaly detection
Decarbonisation,net zero; electrification; hydrogen; emissions; LCA
```

These keywords are included in the prompt context for Gemini triage and are also useful for structuring the digest.

---

## How the script chooses the “latest” Gemini model

The script uses `google-generativeai` to call `genai.list_models()` and filters to models that support **generateContent**.

It then selects the “best available” model using a preference order (heuristic), typically:
1) Prefer “latest” aliases like `models/gemini-pro-latest` or `models/gemini-flash-latest`
2) Otherwise prefer higher capability families (`2.5-pro` > `2.5-flash` > `2.0-flash` etc.)
3) Avoid non-text models (image-only, TTS-only, robotics, etc.) unless explicitly requested

This makes the code resilient when Google:
- renames preview IDs,
- retires models,
- adds newer “latest” aliases.

---

## Avoiding Gemini rate limits (HTTP 429)

If you see:
- `429 Resource exhausted`

Do one or more of:
- Increase `gemini_min_interval_seconds` to **3–5 seconds**
- Increase `gemini_error_sleep_seconds` to **20–60 seconds**
- Reduce batch size (if your script supports `ai_batch_size` or similar)

Rate limits vary by model tier and can change over time.

---

## Running on GitHub Actions (scheduled)

You can run this daily/weekly on GitHub.

### 1) Add GitHub Secrets
In your repo: **Settings → Secrets and variables → Actions → New repository secret**

Add:
- `GEMINI_API_KEY`
- `EMAIL_SENDER`
- `EMAIL_PASSWORD`
- `EMAIL_RECEIVER`
- (optional) `OPENALEX_MAILTO`

### 2) Workflow should:
- check out code
- set up Python
- install deps
- run the script

### 3) Update your workflow entrypoint
If your script filename differs from `daily_digest.py`, change the workflow command accordingly.

> Reality check: Gmail SMTP from GitHub Actions usually works with an App Password, but it can still fail if your Google account security policies block it. That’s an account/security issue, not a code issue.

---

## Security & privacy

- **Never commit `.env`**
- Use Gmail **App Passwords** (requires 2FA) instead of your real password
- Treat exported CSVs as potentially sensitive if you later add private sources

---

## License

Add your preferred license (MIT/Apache-2.0/etc.) if you want others to reuse it professionally.
