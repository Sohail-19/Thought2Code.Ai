# Thought2Code.Ai

**Ideas into code. Code into reality.**

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

The API is intended for local development. Generation is synchronous and returns when the workflow finishes.

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

## Licensing

There is currently no top-level project `LICENSE` or `NOTICE` file. The project’s redistribution and reuse terms need an owner decision. Third-party libraries and assets remain subject to their respective licenses.
