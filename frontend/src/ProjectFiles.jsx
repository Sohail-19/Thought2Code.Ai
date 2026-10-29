import { useEffect, useMemo, useState } from "react";
import { getJobFile, getJobFiles, getJobPreviewAsset } from "./api.js";
import { PREVIEW_BOOTSTRAP } from "./previewBootstrap.js";
import {
  containsCssImportRule,
  CSS_IMPORT_PREVIEW_NOTICE,
  createPreviewResourceLoader,
  formatOmittedAssetNotice,
  isNonLocalReference,
  isPreviewAssetPath,
  MODULE_PREVIEW_NOTICE,
  resolveProjectResource,
} from "./previewAssets.js";

const PREVIEW_CSP = [
  "default-src 'none'",
  "script-src data: 'unsafe-inline'",
  "style-src data: 'unsafe-inline'",
  "img-src data: blob:",
  "font-src data:",
  "connect-src 'none'",
  "media-src data: blob:",
  "object-src 'none'",
  "frame-src 'none'",
  "worker-src 'none'",
  "form-action 'none'",
  "base-uri 'none'",
].join("; ");

function extensionOf(path) {
  return path.split(".").pop()?.toLowerCase() || "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(bytes < 10240 ? 1 : 0)} KB`;
}

function encodeDataUri(mime, content) {
  const bytes = new TextEncoder().encode(content);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  }
  return `data:${mime};base64,${btoa(binary)}`;
}

async function buildPreview(jobId, htmlPath, htmlText, files) {
  const doc = new DOMParser().parseFromString(htmlText, "text/html");
  const projectPaths = new Set(files.map((file) => file.path));
  const resourceLoader = createPreviewResourceLoader();
  let bundledBytes = new TextEncoder().encode(htmlText).length;
  let omittedAssets = 0;

  async function loadTextResource(path) {
    const result = await resourceLoader.load(path, async () => {
      const response = await getJobFile(jobId, path);
      if (typeof response.content !== "string" || response.path !== path) {
        throw new Error("The API returned an invalid project file.");
      }
      const resourceBytes = new TextEncoder().encode(response.content).length;
      if (bundledBytes + resourceBytes > 2 * 1024 * 1024) {
        throw new Error("This project is too large to preview safely (2 MB limit).");
      }
      bundledBytes += resourceBytes;
      return response.content;
    });
    if (result.limited) throw new Error("This project references too many local files to preview safely.");
    if (result.error) throw result.error;
    return result.value;
  }

  async function loadAssetDataUri(path) {
    if (!isPreviewAssetPath(path)) {
      omittedAssets += 1;
      return null;
    }
    const result = await resourceLoader.load(path, async () => {
      const response = await getJobPreviewAsset(jobId, path);
      const bytes = new Uint8Array(response.content);
      if (bundledBytes + bytes.byteLength > 2 * 1024 * 1024) {
        throw new Error("This project is too large to preview safely (2 MB limit).");
      }
      bundledBytes += bytes.byteLength;
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 8192) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
      }
      return `data:${response.contentType};base64,${btoa(binary)}`;
    });
    if (result.limited) {
      omittedAssets += 1;
      return null;
    }
    if (result.error) {
      if (result.error.message?.includes("2 MB limit")) throw result.error;
      omittedAssets += 1;
      return null;
    }
    return result.value;
  }

  async function rewriteCss(css, sourcePath) {
    if (containsCssImportRule(css)) throw new Error(CSS_IMPORT_PREVIEW_NOTICE);
    const matches = [...css.matchAll(/url\(\s*(?:(["'])(.*?)\1|([^)]*?))\s*\)/gi)];
    let result = css;
    for (const match of matches) {
      const raw = (match[2] ?? match[3] ?? "").trim();
      if (isNonLocalReference(raw)) continue;
      const localPath = resolveProjectResource(raw, sourcePath, projectPaths);
      const replacement = localPath ? await loadAssetDataUri(localPath) : null;
      if (!replacement) {
        result = result.replace(match[0], 'url("")');
        if (!localPath) omittedAssets += 1;
      } else {
        result = result.replace(match[0], `url("${replacement}")`);
      }
    }
    return result;
  }

  async function embedAssetAttribute(element, attribute, sourcePath) {
    const value = element.getAttribute(attribute);
    if (!value || isNonLocalReference(value)) return;
    const localPath = resolveProjectResource(value, sourcePath, projectPaths);
    if (localPath && isPreviewAssetPath(localPath)) {
      const uri = await loadAssetDataUri(localPath);
      if (uri) {
        element.setAttribute(attribute, uri);
        return;
      }
    }
    element.removeAttribute(attribute);
    omittedAssets += 1;
  }

  async function embedSrcset(element, sourcePath) {
    const value = element.getAttribute("srcset");
    if (!value || isNonLocalReference(value)) return;
    const entries = value.split(",").map((entry) => entry.trim()).filter(Boolean);
    const rewritten = [];
    for (const entry of entries) {
      const [reference, ...descriptor] = entry.split(/\s+/);
      if (isNonLocalReference(reference)) {
        rewritten.push(entry);
        continue;
      }
      const localPath = resolveProjectResource(reference, sourcePath, projectPaths);
      const uri = localPath && isPreviewAssetPath(localPath)
        ? await loadAssetDataUri(localPath)
        : null;
      if (uri) rewritten.push([uri, ...descriptor].join(" "));
      else omittedAssets += 1;
    }
    if (rewritten.length) element.setAttribute("srcset", rewritten.join(", "));
    else element.removeAttribute("srcset");
  }

  doc.querySelectorAll("base, iframe, object, embed, portal").forEach((node) => node.remove());
  doc.querySelectorAll('meta[http-equiv="refresh" i]').forEach((node) => node.remove());

  for (const link of [...doc.querySelectorAll("link")]) {
    if (!link.rel.toLowerCase().split(/\s+/).includes("stylesheet")) {
      link.remove();
      continue;
    }
    const href = link.getAttribute("href");
    const localPath = resolveProjectResource(href, htmlPath, projectPaths);
    if (!localPath || extensionOf(localPath) !== "css") {
      if (!isNonLocalReference(href)) {
        link.remove();
        omittedAssets += 1;
      }
      continue;
    }
    const css = await loadTextResource(localPath);
    const rewrittenCss = await rewriteCss(css, localPath);
    link.setAttribute("href", encodeDataUri("text/css;charset=utf-8", rewrittenCss));
    link.removeAttribute("integrity");
    link.removeAttribute("crossorigin");
  }

  if ([...doc.querySelectorAll("script")].some(
    (script) => script.getAttribute("type")?.trim().toLowerCase() === "module",
  )) {
    throw new Error(MODULE_PREVIEW_NOTICE);
  }

  for (const script of [...doc.querySelectorAll("script[src]")]) {
    const src = script.getAttribute("src");
    const localPath = resolveProjectResource(src, htmlPath, projectPaths);
    if (!localPath || extensionOf(localPath) !== "js") {
      if (!isNonLocalReference(src)) {
        script.remove();
        omittedAssets += 1;
      }
      continue;
    }
    const source = await loadTextResource(localPath);
    script.setAttribute("src", encodeDataUri("text/javascript;charset=utf-8", source));
    script.removeAttribute("integrity");
    script.removeAttribute("crossorigin");
  }

  for (const element of doc.querySelectorAll("img, audio, video, source, track, input[type='image']")) {
    for (const attribute of ["src", "poster"]) await embedAssetAttribute(element, attribute, htmlPath);
    await embedSrcset(element, htmlPath);
  }
  for (const link of doc.querySelectorAll('link[rel~="icon" i]')) {
    await embedAssetAttribute(link, "href", htmlPath);
  }
  for (const style of doc.querySelectorAll("style")) {
    style.textContent = await rewriteCss(style.textContent, htmlPath);
  }
  for (const element of doc.querySelectorAll("[style]")) {
    element.setAttribute("style", await rewriteCss(element.getAttribute("style"), htmlPath));
  }

  const csp = doc.createElement("meta");
  csp.httpEquiv = "Content-Security-Policy";
  csp.content = PREVIEW_CSP;
  doc.head.prepend(csp);
  const bootstrap = doc.createElement("script");
  bootstrap.textContent = PREVIEW_BOOTSTRAP;
  doc.head.insertBefore(bootstrap, csp.nextSibling);
  const html = `<!doctype html>${doc.documentElement.outerHTML}`;
  return { html, omittedAssets };
}

function FileList({ files, selectedPath, onSelect }) {
  return (
    <ul className="browse-file-list">
      {files.map((file) => (
        <li key={file.path}>
          <button
            className={`browse-file${selectedPath === file.path ? " selected" : ""}`}
            type="button"
            disabled={!file.viewable}
            onClick={() => onSelect(file)}
            title={file.viewable ? `Open ${file.path}` : "This file type or size is not available for viewing"}
          >
            <span className={`file-type-icon type-${file.language}`}>{file.language === "binary" ? "◇" : "⌁"}</span>
            <code>{file.path}</code>
            <span className="file-size">{formatBytes(file.size_bytes)}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function CodeViewer({ file, loading, error }) {
  if (loading) return <div className="file-view-state">Loading file…</div>;
  if (error) return <div className="file-view-state file-view-error">{error}</div>;
  if (!file) return <div className="file-view-state">Choose a generated text file to view its contents.</div>;

  const lines = file.content.split("\n");
  return (
    <div className="code-viewer">
      <div className="code-toolbar">
        <span className="code-filename"><i />{file.path}</span>
        <span className="language-badge">{file.language}</span>
      </div>
      <pre className={`code-content language-${file.language}`}><code>
        {lines.map((line, index) => (
          <span className="code-line" key={index}>
            <i>{index + 1}</i><span>{line || " "}</span>
          </span>
        ))}
      </code></pre>
    </div>
  );
}

function PreviewPane({ state, message, srcDoc, selectedHtml, onSelectHtml, htmlFiles, omittedAssets }) {
  if (state === "loading") {
    return <div className="preview-message"><span className="preview-spinner" />Preparing a sandboxed preview…</div>;
  }
  if (state === "error") {
    return <div className="preview-message preview-error">The preview could not be prepared. {message}</div>;
  }
  if (state === "unavailable") {
    return <div className="preview-message">This project does not include an HTML, CSS, and JavaScript set that can be previewed.</div>;
  }
  return (
    <div className="preview-frame-wrap">
      {htmlFiles.length > 1 && (
        <label className="preview-file-picker">HTML entry
          <select value={selectedHtml} onChange={(event) => onSelectHtml(event.target.value)}>
            {htmlFiles.map((file) => <option key={file.path} value={file.path}>{file.path}</option>)}
          </select>
        </label>
      )}
      <iframe
        className="project-preview-frame"
        title="Sandboxed generated application preview"
        sandbox="allow-scripts"
        referrerPolicy="no-referrer"
        srcDoc={srcDoc}
      />
      {omittedAssets > 0 && <p className="preview-safety-note">{formatOmittedAssetNotice(omittedAssets)}</p>}
      <p className="preview-safety-note">Sandboxed preview. External resources and connections are blocked; generated local CSS and JavaScript are loaded from this job only. Browser storage is temporary within the preview.</p>
    </div>
  );
}

export default function ProjectFiles({ jobId, fallbackFiles = [] }) {
  const [files, setFiles] = useState([]);
  const [listState, setListState] = useState("loading");
  const [listError, setListError] = useState("");
  const [tab, setTab] = useState("files");
  const [selectedPath, setSelectedPath] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [fileError, setFileError] = useState("");
  const [previewState, setPreviewState] = useState("idle");
  const [previewDoc, setPreviewDoc] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [omittedAssets, setOmittedAssets] = useState(0);
  const [selectedHtml, setSelectedHtml] = useState("");

  useEffect(() => {
    let cancelled = false;
    setFiles([]);
    setListState("loading");
    setListError("");
    setSelectedPath("");
    setSelectedFile(null);
    setPreviewState("idle");
    setPreviewDoc("");
    getJobFiles(jobId)
      .then((body) => {
        if (cancelled) return;
        if (body?.job_id !== jobId || !Array.isArray(body.files) || !body.files.every(
          (file) => typeof file?.path === "string" &&
            Number.isInteger(file.size_bytes) &&
            typeof file.language === "string" &&
            typeof file.viewable === "boolean",
        )) {
          throw new Error("The API returned an invalid file listing.");
        }
        setFiles(body.files);
        setListState("ready");
      })
      .catch((error) => {
        if (cancelled) return;
        setListError(error.message || "Could not load the generated file list.");
        setFiles(fallbackFiles.map((path) => ({ path, size_bytes: 0, language: "text", viewable: true })));
        setListState("error");
      });
    return () => { cancelled = true; };
  }, [jobId, fallbackFiles]);

  const htmlFiles = useMemo(
    () => files.filter((file) => file.viewable && ["html", "htm"].includes(extensionOf(file.path)))
      .sort((a, b) => Number(b.path.toLowerCase().endsWith("index.html")) - Number(a.path.toLowerCase().endsWith("index.html"))),
    [files],
  );
  const canPreview = htmlFiles.length > 0 &&
    files.some((file) => file.viewable && extensionOf(file.path) === "css") &&
    files.some((file) => file.viewable && extensionOf(file.path) === "js");

  async function selectFile(file) {
    if (!file.viewable) return;
    setTab("files");
    setSelectedPath(file.path);
    setSelectedFile(null);
    setFileError("");
    setFileLoading(true);
    try {
      const body = await getJobFile(jobId, file.path);
      if (body?.job_id !== jobId || body.path !== file.path || typeof body.content !== "string" || typeof body.language !== "string") {
        throw new Error("The API returned an invalid file response.");
      }
      setSelectedFile(body);
    } catch (error) {
      setFileError(error.message || "Could not read this generated file.");
    } finally {
      setFileLoading(false);
    }
  }

  async function preparePreview(htmlPath) {
    setSelectedHtml(htmlPath);
    setPreviewState("loading");
    setPreviewDoc("");
    setPreviewError("");
    if (!canPreview) {
      setPreviewState("unavailable");
      return;
    }
    try {
      const html = await getJobFile(jobId, htmlPath);
      const built = await buildPreview(jobId, htmlPath, html.content, files);
      setPreviewDoc(built.html);
      setOmittedAssets(built.omittedAssets);
      setPreviewState("ready");
    } catch (error) {
      setPreviewError(error.message || "One or more project files could not be loaded.");
      setPreviewState("error");
    }
  }

  function showPreview() {
    setTab("preview");
    const htmlPath = selectedHtml && htmlFiles.some((file) => file.path === selectedHtml)
      ? selectedHtml
      : htmlFiles[0]?.path;
    if (!htmlPath) {
      setPreviewState("unavailable");
      return;
    }
    preparePreview(htmlPath);
  }

  return (
    <section className="project-files" aria-label="Generated project files">
      <div className="file-tabs" role="tablist" aria-label="Project output">
        <button className={tab === "files" ? "active" : ""} type="button" role="tab" aria-selected={tab === "files"} onClick={() => setTab("files")}>Files <span>{files.length}</span></button>
        <button className={tab === "preview" ? "active" : ""} type="button" role="tab" aria-selected={tab === "preview"} onClick={showPreview}>Preview</button>
      </div>

      {tab === "files" ? (
        <div className="file-workspace">
          <div className="file-explorer">
            <div className="file-explorer-heading">PROJECT FILES</div>
            {listState === "loading" ? <p className="file-list-note">Loading project files…</p> : files.length ? (
              <FileList files={files} selectedPath={selectedPath} onSelect={selectFile} />
            ) : (
              <p className="file-list-note">{listError || "This project has no generated files."}</p>
            )}
            {listState === "error" && files.length > 0 && <p className="file-list-warning">{listError}</p>}
          </div>
          <CodeViewer file={selectedFile} loading={fileLoading} error={fileError} />
        </div>
      ) : (
        <PreviewPane
          state={previewState}
          message={previewError}
          srcDoc={previewDoc}
          selectedHtml={selectedHtml}
          onSelectHtml={preparePreview}
          htmlFiles={htmlFiles}
          omittedAssets={omittedAssets}
        />
      )}
    </section>
  );
}
