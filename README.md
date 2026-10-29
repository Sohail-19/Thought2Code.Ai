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

The API is intended for local development and must bind to `127.0.0.1`; do not expose it on a public interface. CORS is configured for the local Vite origin, but CORS is not authentication and does not prevent direct HTTP clients from calling the API. Any remote exposure requires a separate access-control design before deployment. Generation is synchronous and returns when the workflow finishes.

Generation limits can be tuned with backend environment variables. Defaults include a 64 KiB HTTP request-body limit (`THOUGHT2CODE_MAX_REQUEST_BODY_BYTES`), a 300-second soft job deadline, a 30-second timeout per model request, one transient retry, 4,000 prompt characters, 6,000 Planner context tokens, 12,000 Architect context tokens, 4,096 Planner output tokens, 8,192 Architect output tokens, 16,000 Coder and repair context tokens, 40 planned implementation steps, 8 Coder model iterations per file, 1 MiB per generated file, 50 MiB total output, 200 files per project, and one concurrent generation. The Coder's LangGraph recursion cap is derived from its model-iteration budget to account for the pre-model hook, model node, and tool node; the cap still bounds the internal graph independently. Relevant settings use the `THOUGHT2CODE_` prefix, including `THOUGHT2CODE_MAX_CODER_AGENT_STEPS`, `THOUGHT2CODE_PLANNER_CONTEXT_TOKENS`, `THOUGHT2CODE_ARCHITECT_CONTEXT_TOKENS`, `THOUGHT2CODE_PLANNER_MAX_OUTPUT_TOKENS`, `THOUGHT2CODE_ARCHITECT_MAX_OUTPUT_TOKENS`, `THOUGHT2CODE_CODER_CONTEXT_TOKENS`, `THOUGHT2CODE_REPAIR_CONTEXT_TOKENS`, `THOUGHT2CODE_MODEL_TIMEOUT_SECONDS`, and `THOUGHT2CODE_MAX_PROMPT_CHARACTERS`; values are clamped to conservative bounds. Context budgets use a deterministic UTF-8 byte-based estimate (bytes rounded up by three), not the provider tokenizer. It is an approximate budget rather than an exact token count; prompts and file contents are never silently truncated.

For local diagnosis only, set `THOUGHT2CODE_ENV=local` and `THOUGHT2CODE_LOCAL_DIAGNOSTICS=1`, and run the API on loopback (`127.0.0.1`). This opt-in mode records allowlisted workflow events (node entry/exit, task index, model iteration count, tool name/outcome, and termination reason) without prompt text, tool arguments, generated source, provider messages, or credentials. Hosted-runtime markers disable it. Unset both variables to turn it off.

Abandoned partial jobs are retained for seven days after generation ends, then eligible for cleanup. Each generation request scans at most 100 job-root entries, removes at most 10 stale jobs, and examines at most 1,001 entries within any one job during the preflight scan. The deletion walk independently enforces the configured tree-entry budget using descriptor-relative, no-follow operations; if concurrent growth exceeds the budget, cleanup stops and retains the partially cleaned job directory for a later attempt. A small amount of work may have happened before growth is detected, so this is a bound on each cleanup walk, not an atomic filesystem snapshot. Repeated requests continue from a bounded scan cursor so older entries are eventually considered. Tune this with `THOUGHT2CODE_ORPHAN_JOB_RETENTION_SECONDS`, `THOUGHT2CODE_ORPHAN_CLEANUP_MAX_ENTRIES`, `THOUGHT2CODE_ORPHAN_CLEANUP_MAX_JOBS`, and `THOUGHT2CODE_ORPHAN_CLEANUP_MAX_TREE_ENTRIES`. Jobs larger than the deletion-work limit are retained. Completed jobs are retained indefinitely so their files and ZIPs remain available. Cleanup runs before generation requests and uses per-job OS locks to skip jobs active in another local API process.

The end-to-end deadline is cooperative: synchronous Python/LangGraph work cannot be safely force-cancelled mid-node. Model calls have their own network timeout, agent and output sizes are bounded, and deadline checks prevent a timed-out run from being marked complete after the timeout is observed. An already-running synchronous node may therefore return after the configured soft deadline.

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
