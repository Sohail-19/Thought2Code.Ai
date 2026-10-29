import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("Thought2Code.Ai branding is consistent across the app shell and metadata", async () => {
  const [html, app, readme, packageJson, pyproject, uvLock] = await Promise.all([
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    readFile(new URL("../../README.md", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../../pyproject.toml", import.meta.url), "utf8"),
    readFile(new URL("../../uv.lock", import.meta.url), "utf8"),
  ]);

  assert.match(html, /<title>Thought2Code\.Ai — Describe it\. Build it\.<\/title>/);
  assert.match(html, /href="\/favicon\.svg"/);
  assert.match(app, /aria-label="Thought2Code\.Ai home"/);
  assert.match(app, /Thought2Code<span>\.Ai<\/span>/);
  assert.match(app, /Describe it\.<br \/><span>Build it\.<\/span>/);
  assert.match(app, /Describe it\. Build it\./);
  assert.match(app, /Describe the application you have in mind\./);
  assert.match(readme, /# Thought2Code\.Ai\n\n\*\*Describe it\. Build it\.\*\*/);
  assert.match(readme, /github\.com\/Sohail-19\/Thought2Code\.Ai\.git/);
  assert.equal(JSON.parse(packageJson).name, "thought2code-ai-frontend");
  assert.match(pyproject, /^name = "thought2code-ai"$/m);
  assert.match(uvLock, /^name = "thought2code-ai"$/m);
});
