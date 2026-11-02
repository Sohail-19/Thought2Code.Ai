import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  DEFAULT_THEME,
  THEME_STORAGE_KEY,
  getInitialTheme,
  persistTheme,
} from "../src/theme.js";

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    values,
  };
}

test("first visits default to dark and restore a saved theme", () => {
  const emptyStorage = memoryStorage();
  assert.equal(getInitialTheme(emptyStorage), "dark");
  assert.equal(DEFAULT_THEME, "dark");

  const darkStorage = memoryStorage({ [THEME_STORAGE_KEY]: "dark" });
  assert.equal(getInitialTheme(darkStorage), "dark");
  assert.equal(getInitialTheme(memoryStorage({ [THEME_STORAGE_KEY]: "light" })), "light");
  assert.equal(getInitialTheme(memoryStorage({ [THEME_STORAGE_KEY]: "invalid" })), "dark");
});

test("theme preference persists safely and invalid values are ignored", () => {
  const storage = memoryStorage();
  assert.equal(persistTheme("dark", storage), true);
  assert.equal(storage.values.get(THEME_STORAGE_KEY), "dark");
  assert.equal(getInitialTheme(storage), "dark");
  assert.equal(persistTheme("sepia", storage), false);
  assert.equal(persistTheme("light", null), false);
  assert.equal(persistTheme("light", { setItem() { throw new Error("storage blocked"); } }), false);
});

test("the header exposes an accessible theme action and the page avoids a load flash", async () => {
  const [app, html, css] = await Promise.all([
    readFile(new URL("../src/App.jsx", import.meta.url), "utf8"),
    readFile(new URL("../index.html", import.meta.url), "utf8"),
    readFile(new URL("../src/styles.css", import.meta.url), "utf8"),
  ]);

  assert.match(app, /aria-label=\{`Switch to \$\{nextTheme\} mode`\}/);
  assert.match(app, /aria-pressed=\{theme === "dark"\}/);
  assert.match(html, /data-theme="dark"/);
  assert.match(html, /let theme = "dark"/);
  assert.match(html, /localStorage\.getItem\("thought2code-theme"\)/);
  assert.match(css, /color-scheme: dark/);
  assert.match(css, /:root\[data-theme="light"\]/);
});
