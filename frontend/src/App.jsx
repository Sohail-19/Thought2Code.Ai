import { useState } from "react";
import ProjectFiles from "./ProjectFiles.jsx";
import { API_BASE_URL } from "./api.js";
import { parseGenerateResponse } from "./generationResponse.js";
import { requestProjectArchive } from "./projectDownload.js";

const EXAMPLES = [
  {
    label: "Personal finance dashboard",
    prompt:
      "Build a personal finance dashboard to track monthly spending, budgets, and savings goals with clear charts and a calm, modern interface.",
    icon: "◒",
  },
  {
    label: "Team task manager",
    prompt:
      "Create a lightweight team task manager with projects, priorities, due dates, and a kanban board that works beautifully on mobile.",
    icon: "▦",
  },
  {
    label: "Recipe discovery app",
    prompt:
      "Make a recipe discovery app where people can search by ingredients, save favorites, and follow simple step-by-step cooking instructions.",
    icon: "✳",
  },
];

function Brand() {
  return (
    <a className="brand" href="#top" aria-label="Thought2Code home">
      <span className="brand-mark" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span className="brand-name">thought<span>2</span>code</span>
    </a>
  );
}

function Header() {
  return (
    <header className="topbar">
      <Brand />
      <div className="topbar-meta">
        <span className="local-indicator"><i /> Local workspace</span>
        <span className="topbar-divider" />
        <span className="version-label">EARLY ACCESS</span>
      </div>
    </header>
  );
}

function ExamplePrompts({ onSelect, disabled }) {
  return (
    <div className="examples" aria-label="Example prompts">
      <span className="examples-label">NEED A STARTING POINT?</span>
      <div className="example-list">
        {EXAMPLES.map((example) => (
          <button
            className="example-chip"
            key={example.label}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(example.prompt)}
          >
            <span aria-hidden="true">{example.icon}</span>
            {example.label}
            <span className="chip-arrow" aria-hidden="true">↗</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function ProjectSummary({ project }) {
  return (
    <div className="summary-content">
      <div className="result-kicker"><span className="success-dot" /> PROJECT PLAN</div>
      <h3>{project.name}</h3>
      <p className="project-description">{project.description}</p>
      <div className="summary-divider" />
      <div className="summary-label">TECH STACK</div>
      <p className="tech-stack">{project.techstack}</p>
      <div className="summary-label feature-heading">FEATURES</div>
      <ul className="feature-list">
        {project.features.map((feature, index) => (
          <li key={`${index}-${feature}`}><span aria-hidden="true">✓</span>{feature}</li>
        ))}
      </ul>
    </div>
  );
}

function ResultsPanel({
  result,
  loading,
  error,
  downloading,
  downloadError,
  onDownload,
}) {
  return (
    <aside className={`results-panel${result ? " has-result" : ""}`} aria-live="polite">
      <div className="panel-heading">
        <div>
          <div className="eyebrow">YOUR WORKSPACE</div>
          <h2>{result ? "Generation complete" : "Your project starts here"}</h2>
        </div>
        <div className="panel-heading-actions">
          <button
            className="download-button"
            type="button"
            disabled={!result || result.status !== "completed" || downloading}
            onClick={() => result && onDownload(result.job_id)}
          >
            {downloading ? (
              <><span className="button-spinner" /> Preparing ZIP</>
            ) : (
              <>Download ZIP <span aria-hidden="true">↓</span></>
            )}
          </button>
          <span className="panel-icon" aria-hidden="true">⌘</span>
        </div>
      </div>

      {loading ? (
        <div className="waiting-state">
          <div className="loading-orbit"><span /></div>
          <div>
            <strong>Bringing your idea to life</strong>
            <p>The AI is planning your project and generating its files. This can take a little while.</p>
          </div>
          <div className="loading-track"><span /></div>
          <span className="waiting-caption">WORKING ON YOUR REQUEST</span>
        </div>
      ) : error ? (
        <div className="panel-message error-message">
          <span className="message-symbol">!</span>
          <strong>We couldn’t generate that project</strong>
          <p>{error}</p>
        </div>
      ) : result ? (
        <>
          {downloadError && <p className="download-error" role="alert">{downloadError}</p>}
          <ProjectSummary project={result.project} />
          <div className="file-output">
            <ProjectFiles
              key={result.job_id}
              jobId={result.job_id}
              fallbackFiles={result.generated_files}
            />
            <div className="job-id"><span>JOB ID</span><code>{result.job_id}</code></div>
          </div>
        </>
      ) : (
        <div className="empty-state">
          <div className="empty-art" aria-hidden="true">
            <div className="empty-window">
              <span /><span /><span />
              <div className="empty-window-lines"><i /><i /><i /></div>
            </div>
            <div className="empty-spark spark-one">✳</div>
            <div className="empty-spark spark-two">✦</div>
          </div>
          <strong>Your next idea, made real.</strong>
          <p>Describe what you want to build. Your project plan and generated files will show up here.</p>
          <div className="empty-footnote"><span>01</span> ONE PROMPT, A WHOLE NEW PROJECT</div>
        </div>
      )}
    </aside>
  );
}

function isGenerateResponse(value) {
  return Boolean(
    value &&
      typeof value.job_id === "string" &&
      typeof value.status === "string" &&
      value.status === "completed" &&
      value.project &&
      typeof value.project.name === "string" &&
      typeof value.project.description === "string" &&
      typeof value.project.techstack === "string" &&
      Array.isArray(value.project.features) &&
      value.project.features.every((feature) => typeof feature === "string") &&
      Array.isArray(value.generated_files) &&
      value.generated_files.every((file) => typeof file === "string"),
  );
}

function App() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState("");

  const promptLength = prompt.trim().length;
  const promptError = promptLength > 0 && promptLength < 10
    ? "Add a little more detail (at least 10 characters)."
    : promptLength > 4000
      ? "Keep your description under 4,000 characters."
      : "";

  async function handleGenerate(event) {
    event.preventDefault();
    if (loading) return;
    if (promptLength < 10) {
      setError("Please describe your idea in at least 10 characters.");
      return;
    }
    if (promptLength > 4000) {
      setError("Your description is too long. Please keep it under 4,000 characters.");
      return;
    }

    setLoading(true);
    setError("");
    setDownloadError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_description: prompt.trim() }),
      });

      const body = await parseGenerateResponse(response);
      if (!isGenerateResponse(body)) {
        throw new Error("The API response did not match the expected project result.");
      }
      setResult(body);
    } catch (requestError) {
      if (requestError instanceof TypeError) {
        setError("Could not reach the local API. Make sure the backend is running at the configured address.");
      } else {
        setError(requestError.message || "Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleDownload(jobId) {
    if (
      !result ||
      result.status !== "completed" ||
      result.job_id !== jobId ||
      downloading
    ) {
      return;
    }
    setDownloading(true);
    setDownloadError("");
    try {
      const archive = await requestProjectArchive(jobId, API_BASE_URL);
      const objectUrl = URL.createObjectURL(archive);
      const link = document.createElement("a");
      link.href = objectUrl;
      link.download = `thought2code-${jobId}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (downloadRequestError) {
      setDownloadError(
        downloadRequestError instanceof TypeError
          ? "Could not reach the local API to download this project. Check that the backend is running."
          : downloadRequestError.message || "The project download failed. Please try again.",
      );
    } finally {
      setDownloading(false);
    }
  }

  function chooseExample(examplePrompt) {
    setPrompt(examplePrompt);
    setError("");
    setDownloadError("");
    setResult(null);
  }

  return (
    <div className="app-shell" id="top">
      <div className="ambient-glow" aria-hidden="true" />
      <Header />
      <main className="main-content">
        <section className="hero">
          <div className="hero-eyebrow"><span className="eyebrow-line" /> AI-POWERED APP GENERATION</div>
          <h1>Ideas in.<br /><span>Applications out.</span></h1>
          <p className="hero-copy">Describe the thing you wish existed. Thought2Code will plan it and turn your idea into a working project.</p>
        </section>

        <section className="workspace" aria-label="Project generator">
          <div className="composer-card">
            <form onSubmit={handleGenerate}>
              <div className="composer-topline">
                <label htmlFor="project-prompt">WHAT WOULD YOU LIKE TO BUILD?</label>
                <span className="prompt-format"><span className="format-dot" /> NATURAL LANGUAGE</span>
              </div>
              <textarea
                id="project-prompt"
                value={prompt}
                onChange={(event) => {
                  setPrompt(event.target.value);
                  if (error) setError("");
                }}
                placeholder="A clear, thoughtful description helps us build something closer to what you have in mind..."
                maxLength={5000}
                disabled={loading}
                aria-describedby="prompt-hint prompt-counter"
              />
              <div className="composer-bottomline">
                <span className={`prompt-hint${promptError ? " prompt-error" : ""}`} id="prompt-hint">
                  {promptError || "Be specific about the features, style, and who it’s for."}
                </span>
                <span className={`prompt-counter${promptLength > 4000 ? " over-limit" : ""}`} id="prompt-counter">
                  {promptLength.toLocaleString()} <span>/ 4,000</span>
                </span>
              </div>
              <div className="composer-actions">
                <span className="private-note"><span aria-hidden="true">⌑</span> Your idea stays on your local workspace</span>
                <button className="generate-button" type="submit" disabled={loading || promptLength < 10 || promptLength > 4000}>
                  {loading ? <><span className="button-spinner" /> Generating</> : <>Generate project <span aria-hidden="true">↗</span></>}
                </button>
              </div>
              {error && <p className="form-error" role="alert">{error}</p>}
            </form>
            <ExamplePrompts onSelect={chooseExample} disabled={loading} />
          </div>
          <ResultsPanel
            result={result}
            loading={loading}
            error={error && !result ? error : ""}
            downloading={downloading}
            downloadError={downloadError}
            onDownload={handleDownload}
          />
        </section>

        <div className="workflow-note"><span className="workflow-icon">✳</span><span><strong>One request, one complete build.</strong> Generation runs synchronously, so keep this page open while your project is being created.</span></div>
      </main>
      <footer className="footer">
        <Brand />
        <span>From an idea to working code.</span>
        <span className="footer-build">BUILT FOR WHAT’S NEXT <i>✳</i></span>
      </footer>
    </div>
  );
}

export default App;
