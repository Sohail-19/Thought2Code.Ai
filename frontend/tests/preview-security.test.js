import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../src/ProjectFiles.jsx", import.meta.url), "utf8");

test("generated preview iframe has only script permission", () => {
  assert.match(source, /sandbox="allow-scripts"/);
  assert.doesNotMatch(source, /sandbox="[^"]*(allow-same-origin|allow-top-navigation|allow-popups)/);
});

test("preview CSP blocks network connections and nested active content", () => {
  for (const directive of [
    "default-src 'none'",
    "connect-src 'none'",
    "object-src 'none'",
    "frame-src 'none'",
    "worker-src 'none'",
    "form-action 'none'",
    "base-uri 'none'",
  ]) {
    assert.ok(source.includes(`"${directive}"`), `Missing preview policy: ${directive}`);
  }
  assert.match(source, /querySelectorAll\("base, iframe, object, embed, portal"\)/);
});

test("preview combines same-job local styles and scripts before rendering", () => {
  assert.match(source, /resolveProjectResource\(href, htmlPath, projectPaths\)/);
  assert.match(source, /resolveProjectResource\(src, htmlPath, projectPaths\)/);
  assert.match(source, /setAttribute\("href", encodeDataUri\("text\/css;charset=utf-8", rewrittenCss\)\)/);
  assert.match(source, /setAttribute\("src", encodeDataUri\("text\/javascript;charset=utf-8", source\)\)/);
  assert.match(source, /head\.insertBefore\(bootstrap, csp\.nextSibling\)/);
});
