# mr-lead-agent

> **AI-powered code review agent for GitLab Merge Requests.**  
> Analyses diffs, retrieves relevant codebase context and produces structured reviews with blockers, risks and questions — in seconds.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Providers](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20Gemini%20%7C%20Groq%20%7C%20OpenRouter-green.svg)](#llm-providers)

---

## Features

- 🔍 **Smart context retrieval** — extracts identifiers from the diff and ripgrep-searches the local repo clone for relevant definitions
- 🛡️ **Secret redaction** — masks API keys, tokens, passwords before sending to any LLM
- 🤖 **Multi-provider LLM support** — plug in DeepSeek, Gemini, OpenRouter (free tier) or Groq
- 📋 **Structured output** — blockers, key risks, summary and questions to the author in JSON + Rich terminal output
- 💾 **Run history** — every review is saved to `runs/` as JSON for audit / post-processing
- ⚡ **Fast** — full pipeline (fetch → clone/fetch → retrieve → redact → prompt → LLM → render) typically under 60 s

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/mr-lead-agent
cd mr-lead-agent
poetry install
```

Requires: **Python 3.11+**, **[ripgrep](https://github.com/BurntSushi/ripgrep)** (`rg`), **git**

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set GITLAB_BASE_URL, GITLAB_TOKEN, LLM_PROVIDER + API key
```

### 3. Run

```bash
poetry run review-mr --mr-iid 42
```

---

## LLM Providers

| Provider | Free tier | Quality | Setup |
|----------|-----------|---------|-------|
| **DeepSeek** (`deepseek-chat` = V3) | ✅ ~$0.007/review | ⭐⭐⭐⭐⭐ Best value | `DEEPSEEK_API_KEY=sk-…` |
| **Groq** | ✅ Limited TPM | ⭐⭐⭐⭐ | `GROQ_API_KEY=gsk_…` |
| **OpenRouter** | ✅ Free models | ⭐⭐⭐⭐ | `OPENROUTER_API_KEY=sk-or-…` |
| **Gemini** | Regional limits | ⭐⭐⭐⭐ | `GEMINI_API_KEY=AIza…` |

Switch via `.env`:
```bash
LLM_PROVIDER=deepseek   # gemini | deepseek | openrouter | groq
```

---

## Configuration

All options can be set via `.env` or CLI flags (`poetry run review-mr --help`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GITLAB_BASE_URL` | — | Your GitLab instance URL |
| `GITLAB_TOKEN` | — | Personal Access Token (read_api scope) |
| `REPO_URL` | — | Full HTTPS repo URL |
| `MR_IID` | — | Merge Request IID |
| `LLM_PROVIDER` | `gemini` | `gemini` \| `deepseek` \| `openrouter` \| `groq` |
| `MAX_CONTEXT_FRAGMENTS` | `12` | Context code snippets to include (0 = disable) |
| `NO_VERIFY_SSL` | `false` | Skip SSL verification (self-hosted GitLab) |

---

## How It Works

```
GitLab API → MR diff + metadata
     ↓
Repo clone/fetch → ripgrep context retrieval
     ↓
Secret redaction (API keys, tokens, URLs)
     ↓
Prompt assembly (diff + context + instructions)
     ↓
LLM API call (DeepSeek / Gemini / Groq / OpenRouter)
     ↓
Structured JSON review → Rich terminal output + runs/*.json
```

---

## Output Example

```
── MR Review: Add economic activity endpoints ─────────────────────
URL:    https://gitlab.example.com/org/repo/-/merge_requests/47
Author: developer

┌─ Pipeline Stats ────────────────┐
│ Diff lines          1746        │
│ Context fragments     12        │
│ Secrets redacted       0        │
└─────────────────────────────────┘

── Summary ────────────────────────────────────────────────────────
  • Added new API endpoints for economic activity pie chart data
  • Implemented aggregation logic for profitability / tax burden

── Blockers ───────────────────────────────────────────────────────
  [1] Incorrect conditional logic for industry filtering
      router.py:315  elif → should be if
      Fix: Change 'elif not region_name and ...' to 'if ...'
```

---

## Development

```bash
poetry run pytest            # run tests (59 tests)
poetry run ruff check src/   # lint
```

---

## License

Copyright © 2026 Eugene M.

This project is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).

**TL;DR:** Free for open-source use. If you run this as part of a network service (SaaS), you must publish your source code under AGPL-3.0 — or [contact us](mailto:ev1geniu@gmail.com) for a commercial license.
