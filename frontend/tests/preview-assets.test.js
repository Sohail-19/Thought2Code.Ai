import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  containsCssImportRule,
  CSS_IMPORT_PREVIEW_NOTICE,
  createPreviewResourceLoader,
  formatOmittedAssetNotice,
  isNonLocalReference,
  isPreviewAssetPath,
  MODULE_PREVIEW_NOTICE,
  PREVIEW_MAX_RESOURCE_REQUESTS,
  resolveProjectResource,
} from "../src/previewAssets.js";

const projectFiles = new Set([
  "index.html",
  "assets/logo.png",
  "assets/fonts/regular.woff2",
  "css/site.css",
  "js/app.js",
]);

test("preview resources resolve relative and project-root paths within the job", () => {
  assert.equal(resolveProjectResource("../assets/logo.png?v=1", "css/site.css", projectFiles), "assets/logo.png");
  assert.equal(resolveProjectResource("/js/app.js", "index.html", projectFiles), "js/app.js");
  assert.equal(resolveProjectResource("./fonts/regular.woff2#font", "assets/css/site.css", projectFiles), null);
});

test("preview resource resolution rejects traversal and malformed local paths", () => {
  for (const reference of ["../../outside.png", "%2e%2e/%2e%2e/outside.png", "..%2Fsecret.js", "\\\\server\\asset.png", "bad%ZZ.png"]) {
    assert.equal(resolveProjectResource(reference, "index.html", projectFiles), null, reference);
  }
});

test("missing local resources resolve as unavailable instead of escaping the job", () => {
  assert.equal(resolveProjectResource("./missing.css", "index.html", projectFiles), null);
});

test("CSS import detection ignores comments and strings but catches import rules", () => {
  assert.equal(containsCssImportRule('/* @import "comment.css"; */ .x { content: "@import fake.css"; }'), false);
  assert.equal(containsCssImportRule('@IMPORT url("theme.css");'), true);
  assert.equal(containsCssImportRule('@import "theme.css";'), true);
  assert.equal(containsCssImportRule('@\\69mport "escaped.css";'), true);
  assert.match(CSS_IMPORT_PREVIEW_NOTICE, /combine imported styles/i);
});

test("module imports have an explicit preview limitation notice", async () => {
  assert.match(MODULE_PREVIEW_NOTICE, /module scripts are not supported/i);
  const source = await readFile(new URL("../src/ProjectFiles.jsx", import.meta.url), "utf8");
  assert.match(source, /getAttribute\("type"\).*module/);
  assert.match(source, /setPreviewError\(error\.message/);
});

test("data, fragment, and external resource references remain classified as non-local", () => {
  for (const reference of ["data:image/png;base64,abc", "blob:https://site/id", "#icon", "https://cdn.invalid/a.png", "//cdn.invalid/a.png"]) {
    assert.equal(isNonLocalReference(reference), true);
    assert.equal(resolveProjectResource(reference, "index.html", projectFiles), null);
  }
  assert.equal(isNonLocalReference("assets/logo.png"), false);
});

test("only passive allowlisted static asset extensions are fetched for preview", () => {
  for (const path of ["assets/a.png", "assets/a.webp", "fonts/a.woff2", "media/a.mp4"]) {
    assert.equal(isPreviewAssetPath(path), true);
  }
  for (const path of ["page.html", "script.js", "image.svg", "unknown.bin"]) {
    assert.equal(isPreviewAssetPath(path), false);
  }
});

test("failed unique resources consume the request budget and later requests are omitted", async () => {
  const limit = 4;
  const resources = createPreviewResourceLoader(limit);
  let requests = 0;
  for (let index = 0; index < limit + 12; index += 1) {
    const result = await resources.load(`missing-${index}.png`, async () => {
      requests += 1;
      throw new Error("not found");
    });
    if (index < limit) assert.match(result.error.message, /not found/);
    else assert.equal(result.limited, true);
  }
  assert.equal(requests, limit);
  assert.equal(resources.attemptedCount, limit);
});

test("repeated references to the same missing resource reuse the cached failure", async () => {
  const resources = createPreviewResourceLoader(3);
  let requests = 0;
  for (let index = 0; index < 8; index += 1) {
    const result = await resources.load("missing.png", async () => {
      requests += 1;
      throw new Error("not found");
    });
    assert.match(result.error.message, /not found/);
  }
  assert.equal(requests, 1);
  assert.equal(resources.attemptedCount, 1);
});

test("successful and failed requests share the exact budget and successful results stay cached", async () => {
  const resources = createPreviewResourceLoader(2);
  let requests = 0;
  const fetch = (value, fail = false) => resources.load(value, async () => {
    requests += 1;
    if (fail) throw new Error("missing");
    return value;
  });

  assert.deepEqual(await fetch("style.css"), { value: "style.css" });
  assert.deepEqual(await fetch("missing.png", true).then(({ error }) => ({ message: error.message })), { message: "missing" });
  assert.deepEqual(await fetch("style.css"), { value: "style.css" });
  assert.equal((await fetch("third.png")).limited, true);
  assert.equal(requests, 2);
  assert.equal(resources.attemptedCount, 2);
});

test("the configured request budget admits its boundary and makes no requests after exhaustion", async () => {
  const resources = createPreviewResourceLoader(PREVIEW_MAX_RESOURCE_REQUESTS);
  let requests = 0;
  for (let index = 0; index < PREVIEW_MAX_RESOURCE_REQUESTS; index += 1) {
    const result = await resources.load(`asset-${index}.png`, async () => {
      requests += 1;
      return "loaded";
    });
    assert.equal(result.value, "loaded");
  }
  assert.equal((await resources.load("over-budget.png", async () => {
    requests += 1;
    return "unexpected";
  })).limited, true);
  assert.equal(requests, PREVIEW_MAX_RESOURCE_REQUESTS);
  assert.equal(resources.attemptedCount, PREVIEW_MAX_RESOURCE_REQUESTS);
});

test("omission notice remains clear for missing or unsupported resources", () => {
  assert.equal(formatOmittedAssetNotice(0), "");
  assert.equal(formatOmittedAssetNotice(1), "1 missing or unsupported local asset was omitted from this preview.");
  assert.equal(formatOmittedAssetNotice(3), "3 missing or unsupported local assets were omitted from this preview.");
});

test("preview continues to use restrictive iframe and CSP while rewriting local assets", async () => {
  const source = await readFile(new URL("../src/ProjectFiles.jsx", import.meta.url), "utf8");
  assert.match(source, /sandbox="allow-scripts"/);
  assert.doesNotMatch(source, /sandbox="[^"]*allow-same-origin/);
  assert.match(source, /img-src data: blob:/);
  assert.match(source, /font-src data:/);
  assert.match(source, /connect-src 'none'/);
  assert.match(source, /rewriteCss\(css, localPath\)/);
  assert.match(source, /embedAssetAttribute\(element, attribute, htmlPath\)/);
  assert.match(source, /getJobPreviewAsset\(jobId, path\)/);
});
