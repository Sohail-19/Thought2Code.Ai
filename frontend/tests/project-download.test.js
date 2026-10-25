import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { requestProjectArchive } from "../src/projectDownload.js";

test("download requests the current job archive and returns its blob", async () => {
  const archive = new Blob(["zip-bytes"], { type: "application/zip" });
  let requestedUrl;
  const result = await requestProjectArchive(
    "job id/with unsafe chars",
    "http://127.0.0.1:8000/",
    async (url) => {
      requestedUrl = url;
      return {
        ok: true,
        headers: { get: () => "application/zip" },
        blob: async () => archive,
      };
    },
  );

  assert.equal(
    requestedUrl,
    "http://127.0.0.1:8000/jobs/job%20id%2Fwith%20unsafe%20chars/download",
  );
  assert.equal(result, archive);
});

test("download surfaces the API's safe HTTP error detail", async () => {
  await assert.rejects(
    requestProjectArchive("job-id", "http://localhost:8000", async () => ({
      ok: false,
      json: async () => ({ detail: "Completed job not found" }),
    })),
    { message: "Completed job not found" },
  );
});

test("download handles malformed API errors and network failures", async () => {
  await assert.rejects(
    requestProjectArchive("job-id", "http://localhost:8000", async () => ({
      ok: false,
      json: async () => { throw new Error("not json"); },
    })),
    { message: "The project download failed. Please try again." },
  );
  await assert.rejects(
    requestProjectArchive("job-id", "http://localhost:8000", async () => ({
      ok: true,
      headers: { get: () => "text/html" },
      blob: async () => new Blob(["not a zip"]),
    })),
    { message: "The project download failed. Please try again." },
  );
  await assert.rejects(
    requestProjectArchive("job-id", "http://localhost:8000", async () => {
      throw new TypeError("offline");
    }),
    { message: "offline" },
  );
});

test("download action is disabled without a completed job and while downloading", async () => {
  const appSource = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(appSource, /disabled=\{!result \|\| result\.status !== "completed" \|\| downloading\}/);
  assert.match(appSource, /requestProjectArchive\(jobId, API_BASE_URL\)/);
  assert.match(appSource, /link\.download = `thought2code-\$\{jobId\}\.zip`/);
});
