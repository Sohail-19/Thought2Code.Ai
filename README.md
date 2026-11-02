# Thought2Code.Ai

**Describe it. Build it.**

Thought2Code.Ai turns a natural-language project description into a planned application and generated project files. It combines a Python generation workflow with a local React interface for submitting ideas, browsing generated files, previewing supported websites, and downloading completed projects.

## How it works

The existing LangGraph workflow keeps its Planner → Architect → Coder structure:

- **Planner** creates a structured project plan.
- **Architect** turns the plan into ordered implementation tasks.
- **Coder** writes the project files through the existing file tools.

The backend uses LangChain and Groq with the configured `openai/gpt-oss-120b` model. The FastAPI generation endpoint runs synchronously. The command-line entry point remains available alongside the API.

## Stack

- Python 3.11+
- LangGraph, LangChain, Groq, and Pydantic
- FastAPI and Uvicorn
- React and Vite
- `uv` for Python dependencies

## Local setup

### Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js and npm
- A Groq API key from the [Groq Console](https://console.groq.com/keys)

### Install

```bash
git clone https://github.com/Sohail-19/Thought2Code.Ai.git
cd Thought2Code.Ai
uv sync
npm install --prefix frontend
```

Create a local `.env` based on `.sample_env` and add the Groq key there. Keep the key on the backend and never commit `.env`.

### Start the API

In one terminal, from the repository root:

```bash
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

For local development, bind the API to `127.0.0.1`. CORS defaults to the local Vite origin. CORS is not authentication and does not prevent direct HTTP clients from calling the API.

Generation limits can be tuned with backend environment variables. Defaults include a 64 KiB HTTP request-body limit (`THOUGHT2CODE_MAX_REQUEST_BODY_BYTES`), a 300-second soft job deadline, a 30-second timeout per model request, one transient retry, 4,000 prompt characters, 6,000 Planner context tokens, 12,000 Architect context tokens, 4,096 Planner output tokens, 8,192 Architect output tokens, 16,000 Coder and repair context tokens, 40 planned implementation steps, 8 Coder model iterations per file, 1 MiB per generated file, 50 MiB total output, 200 files per project, and one concurrent generation. The Coder's LangGraph recursion cap is derived from its model-iteration budget to account for the pre-model hook, model node, and tool node; the cap still bounds the internal graph independently. Relevant settings use the `THOUGHT2CODE_` prefix, including `THOUGHT2CODE_MAX_CODER_AGENT_STEPS`, `THOUGHT2CODE_PLANNER_CONTEXT_TOKENS`, `THOUGHT2CODE_ARCHITECT_CONTEXT_TOKENS`, `THOUGHT2CODE_PLANNER_MAX_OUTPUT_TOKENS`, `THOUGHT2CODE_ARCHITECT_MAX_OUTPUT_TOKENS`, `THOUGHT2CODE_CODER_CONTEXT_TOKENS`, `THOUGHT2CODE_REPAIR_CONTEXT_TOKENS`, `THOUGHT2CODE_MODEL_TIMEOUT_SECONDS`, and `THOUGHT2CODE_MAX_PROMPT_CHARACTERS`; values are clamped to conservative bounds. Context budgets use a deterministic UTF-8 byte-based estimate (bytes rounded up by three), not the provider tokenizer. It is an approximate budget rather than an exact token count; prompts and file contents are never silently truncated.

For local diagnosis only, set `THOUGHT2CODE_ENV=local` and `THOUGHT2CODE_LOCAL_DIAGNOSTICS=1`, and run the API on loopback (`127.0.0.1`). This opt-in mode records allowlisted workflow events (node entry/exit, task index, model iteration count, tool name/outcome, and termination reason) without prompt text, tool arguments, generated source, provider messages, or credentials. Hosted-runtime markers disable it. Unset both variables to turn it off.

Failed or abandoned partial jobs default to a 24-hour retention period. Completed jobs also default to a 24-hour retention period, after which they are eligible for deletion and no longer available for browsing or download. Cleanup runs before generation requests, so a job can remain longer than its threshold during idle periods; on Render the entire ephemeral filesystem can disappear sooner during a restart, redeploy, or spin-down. Each cleanup invocation scans at most 100 job-root entries, removes at most 10 stale jobs, and examines at most 1,001 entries within any one job during preflight. The deletion walk independently enforces its entry budget using descriptor-relative, no-follow operations. If concurrent growth exceeds the budget, cleanup stops and retains the partially cleaned job directory for a later attempt; cleanup work is bounded, but this is not an atomic filesystem snapshot. A bounded scan cursor lets later requests continue through the job root. Configure `THOUGHT2CODE_ORPHAN_JOB_RETENTION_SECONDS`, `THOUGHT2CODE_COMPLETED_JOB_RETENTION_SECONDS`, `THOUGHT2CODE_ORPHAN_CLEANUP_MAX_ENTRIES`, `THOUGHT2CODE_ORPHAN_CLEANUP_MAX_JOBS`, and `THOUGHT2CODE_ORPHAN_CLEANUP_MAX_TREE_ENTRIES`. Jobs above the deletion-work limit are retained for a later attempt. Per-job OS locks protect active jobs from cleanup.

The end-to-end deadline is cooperative: synchronous Python/LangGraph work cannot be safely force-cancelled mid-node. Model calls have their own network timeout, agent and output sizes are bounded, and deadline checks prevent a timed-out run from being marked complete after the timeout is observed. An already-running synchronous node may therefore return after the configured soft deadline.

## Free placement-demo deployment: Vercel + Render

The frontend is a static Vite app. In Vercel, set **Root Directory** to `frontend`, **Framework Preset** to Vite, **Build Command** to `npm run build`, and **Output Directory** to `dist`. Set the production environment variable `VITE_API_BASE_URL` to the HTTPS base URL of the Render API (for example, `https://<your-service>.onrender.com`, without a trailing slash). `frontend/src/api.js` keeps `http://127.0.0.1:8000` as its development fallback. Vite-prefixed values are embedded in public browser assets: never set `VITE_GROQ_API_KEY`, provider keys, passwords, or other secrets. The Vite build rejects sensitive-looking `VITE_*` variable names.

This setup uses only **Vercel Hobby**, a **Render Free Web Service**, and a **Render Free Key Value** instance. It requires no paid Render disk, paid service, or database. It is intended for a placement/demo workload, not production availability or durable storage.

Create a Render **Web Service** from the `main` branch, repository root `.`, and Python 3.11 or newer. Select the **Free** instance type. Use build command `pip install uv && uv sync --frozen --no-dev` and start command `uv run --no-sync uvicorn api.main:app --host 0.0.0.0 --port $PORT --proxy-headers '--forwarded-allow-ips=*'`. Set health check path to `/health`. Render supplies `PORT`; the process must bind to `0.0.0.0`. The lockfile is `uv.lock`; `uv sync --frozen --no-dev` installs the locked runtime dependencies.

Set these Render environment variables:

| Variable | Value |
| --- | --- |
| `GROQ_API_KEY` | Your provider key, entered only in Render's backend environment settings and marked secret. |
| `THOUGHT2CODE_PUBLIC_DEMO` | `true` to require the shared limiter for public generation. |
| `THOUGHT2CODE_GENERATION_ENABLED` | `true` to allow generation; set to `false` immediately to disable generation if abuse or spend is concerning. |
| `THOUGHT2CODE_ALLOWED_ORIGINS` | Comma-separated exact frontend origins, e.g. `https://<your-production-domain>`; no wildcard. |
| `THOUGHT2CODE_REDIS_URL` | The private/internal URL for a Render Key Value (Redis-compatible) instance in the same region. Keep it backend-only and secret. |
| `THOUGHT2CODE_PROJECTS_ROOT` | `/tmp/thought2code/jobs` (ephemeral local filesystem; not durable). |
| `THOUGHT2CODE_MAX_REQUEST_BODY_BYTES` | Optional; defaults to `65536` bytes. |
| `THOUGHT2CODE_CLIENT_RATE_LIMIT` | Optional; defaults to 5 generation requests per client per hour in public-demo mode. |
| `THOUGHT2CODE_CLIENT_RATE_WINDOW_SECONDS` | Optional; defaults to `3600`. |
| `THOUGHT2CODE_MAX_CONCURRENT_GENERATIONS` | Optional; defaults to 1 shared generation at a time. |
| `THOUGHT2CODE_ORPHAN_JOB_RETENTION_SECONDS` | Optional; defaults to `86400` seconds (24 hours) for failed/partial jobs. |
| `THOUGHT2CODE_COMPLETED_JOB_RETENTION_SECONDS` | Optional; defaults to `86400` seconds (24 hours) before completed jobs become eligible for cleanup. |

Create a **Free** Render Key Value instance in the same region as the Web Service. In its settings choose **Maxmemory Policy: `noeviction`** so a full store returns write errors instead of silently evicting rate-limit keys. Use the internal connection URL shown in the Key Value dashboard as `THOUGHT2CODE_REDIS_URL`; keep it only in Render's backend environment. Do not enable an external Key Value connection. Free Key Value is in-memory and non-persistent. If it restarts, the per-client hourly counters and generation leases reset. During an outage, public `/generate` requests fail closed with HTTP 503; the application does not fall back to weaker process-local public limits. After a store restart, protection resumes, but previously accumulated counters cannot be recovered.

Public-demo protections default to five generation requests per client per 3,600-second window, one concurrent generation, and a 64 KiB maximum HTTP request body. Keep `THOUGHT2CODE_PUBLIC_DEMO=true` and `THOUGHT2CODE_GENERATION_ENABLED=true` for the demo; set the latter to `false` to stop new generations. The app requires the shared Key Value store in public mode and returns HTTP 503 if it is unconfigured or unavailable. The limiter uses Render's forwarded client IP, but remains an abuse deterrent rather than authentication: distributed IPs can evade per-client limits, and a free KV restart resets counters. CORS is not authentication. These controls reduce exposure but do not guarantee a maximum Groq bill; monitor provider usage and disable generation if costs or abuse become concerning. Render's free services may suspend or restart, so availability is not guaranteed.

Render's Free Web Service has a monthly instance-hours allowance and included bandwidth/build usage. Check the dashboard's usage and billing controls before launch. To avoid charges, do not create paid resources or add-ons, and do not enable paid usage beyond the free plan; if the workspace has a payment method, configure the workspace spend limit conservatively. Vercel Hobby is free but subject to its plan limits and usage policies. Groq API usage is separate from hosting and can have its own quota or charges; the generation kill switch does not change provider billing for requests already in flight.

Generated projects are stored under `/tmp/thought2code/jobs` on the Render service's **ephemeral filesystem**. Jobs are temporary: Render may erase them when the service spins down after 15 minutes without inbound traffic, or on a restart, redeploy, or instance replacement. A cold start after idle can take about a minute. The API cleans stale failed and completed jobs during generation requests; completed jobs default to 24-hour retention. Idle periods can delay cleanup, but Render's ephemeral filesystem may erase files sooner. Users must download completed ZIPs promptly; job links and files are not durable and can disappear at any time. Free Web Services cannot attach persistent disks. The project does not require or document any paid disk, database, or service. Render Free Web Services run one instance and have monthly usage/resource limitations; the Free Key Value store also has limited memory, connections, and no persistence. The selected plan's current limits are shown in the Render dashboard and may change.

For Vercel, create a project using `frontend` as the **Root Directory**, the Vite preset, `npm run build` as the build command, and `dist` as the output directory. Set `VITE_API_BASE_URL` to the Render HTTPS URL in **Project → Settings → Environment Variables** for Production, then redeploy. This variable is public frontend configuration; Groq credentials and the Key Value URL belong only in the Render backend environment. Use the exact Vercel production domain for `THOUGHT2CODE_ALLOWED_ORIGINS`. Preview deployments have different origins; add only stable origins you explicitly intend to trust, never a wildcard.

In Render Dashboard, create the **Free Web Service** and **Free Key Value** in the same region. Configure the service environment variables above, set **Health Check Path** to `/health`, and set `THOUGHT2CODE_PROJECTS_ROOT=/tmp/thought2code/jobs`. In the Key Value service, set **Maxmemory Policy** to `noeviction`, leave persistence unavailable/off, and copy its internal URL into the Render Web Service's `THOUGHT2CODE_REDIS_URL`. No paid disk, service, or database is needed or should be selected for this placement-demo configuration.

Review the current limits before setup: [Render Free services](https://render.com/docs/free), [Render Key Value](https://render.com/docs/key-value), [Render Web Services](https://render.com/docs/web-services), and [Vercel Hobby](https://vercel.com/docs/plans/hobby).

### Start the frontend

In a second terminal, from the repository root:

```bash
npm run dev --prefix frontend
```

Open the local Vite address printed in the terminal (normally `http://localhost:5173`). The frontend uses the local API at `http://127.0.0.1:8000` by default.

### Use the CLI

The command-line generation flow remains available from the repository root:

```bash
uv run python main.py
```

## Generated projects

Completed jobs provide a project summary, generated file listing and text-file viewer. Supported HTML/CSS/JavaScript projects can be previewed in a restricted sandboxed iframe, and completed projects can be downloaded as ZIP archives. Generated code is untrusted; previewing it does not grant access to the Thought2Code.Ai parent interface or local files.

The preview embeds supported same-job styles, classic JavaScript, and passive assets under size and resource-count limits. CSS `@import` rules and JavaScript module scripts are not bundled: when either is present, preview preparation stops and displays a specific notice. Combine imported CSS into one stylesheet and use classic scripts to preview the project. External network resources remain blocked by the preview CSP; missing or unsupported local assets are omitted with a notice. The iframe keeps only `allow-scripts`, with no same-origin access.

ZIP downloads filter hidden files, credential-oriented filenames and directories, and several recognizable private-key and API-token formats. This filtering is best-effort and cannot guarantee that every credential or sensitive value is removed from generated source. Review generated files before sharing an archive, and avoid putting real secrets in project descriptions or generated code.

Before completion, output validation checks local HTML/CSS references and static JavaScript imports without executing generated code. Extensionless JavaScript imports resolve only when there is one exact file or one supported `.js`, `.mjs`, `.cjs`, or directory `index` candidate; ambiguous matches fail validation. Package imports and external URLs are ignored. Dynamic JavaScript imports are not resolved because the validator does not use a complete JavaScript AST parser; a future implementation should use a bounded parser and resolve only statically analyzable specifiers. Node's syntax-only `--check` is required for ordinary `.js`, `.mjs`, and `.cjs` files and inline scripts. If Node is unavailable, cannot start, times out, or fails, validation blocks completion; the API returns its generic safe generation error. `.cjs` is checked in CommonJS mode and `.js`/`.mjs` plus inline scripts in module mode. JSX and TypeScript files receive no syntax or type checking (only the limited static-import scan); HTML and CSS checks are structural checks, not complete grammar parsers.

## Licensing

There is currently no top-level project `LICENSE` or `NOTICE` file. The project’s redistribution and reuse terms need an owner decision. Third-party libraries and assets remain subject to their respective licenses.
