import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { parseGenerateResponse } from "../src/generationResponse.js";

const rateLimitDetail = "The AI provider's usage limit has been reached. Please wait and try again later.";

test("generation keeps the API's safe HTTP 429 detail for the UI", async () => {
  const appSource = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  const response = {
    ok: false,
    status: 429,
    json: async () => ({ detail: rateLimitDetail }),
  };

  await assert.rejects(parseGenerateResponse(response), { message: rateLimitDetail });
  assert.match(appSource, /parseGenerateResponse\(response\)/);
  assert.match(appSource, /setError\(requestError\.message/);
});

test("generation displays a safe retry delay from structured provider errors", async () => {
  const response = {
    ok: false,
    status: 429,
    json: async () => ({
      detail: {
        message: "The AI provider has temporarily limited this request.",
        error_type: "rate_limit_error",
        category: "rate_limit",
        retry_after_seconds: 12,
      },
    }),
  };

  await assert.rejects(parseGenerateResponse(response), (error) => {
    assert.equal(error.message, "The AI provider has temporarily limited this request. Try again in about 12 seconds.");
    assert.equal(error.category, "rate_limit");
    assert.equal(error.retryAfterSeconds, 12);
    return true;
  });
});

test("generation displays safe workflow-limit details without exposing internals", async () => {
  const response = {
    ok: false,
    status: 422,
    json: async () => ({
      detail: {
        message: "Generation reached a bounded workflow step limit. Try a smaller or simpler project description.",
        category: "workflow_limit",
        error_type: "coder_step_limit",
        stage: "Coder",
        configured_limit: 8,
        configured_limit_unit: "model_iterations",
        observed_model_iterations: 8,
      },
    }),
  };

  await assert.rejects(parseGenerateResponse(response), (error) => {
    assert.equal(error.message, "Generation reached a bounded workflow step limit. Try a smaller or simpler project description. (Coder model iteration limit: 8/8.)");
    assert.equal(error.category, "workflow_limit");
    assert.equal(error.stage, "Coder");
    assert.equal(error.configuredLimit, 8);
    assert.equal(error.configuredLimitUnit, "model_iterations");
    assert.equal(error.observedModelIterations, 8);
    return true;
  });
});

test("generation distinguishes an inner ReAct graph limit from model iterations", async () => {
  await assert.rejects(parseGenerateResponse({
    ok: false,
    status: 422,
    json: async () => ({
      detail: {
        message: "Generation reached a bounded workflow step limit.",
        category: "workflow_limit",
        error_type: "coder_react_recursion_limit",
        stage: "Coder",
        configured_limit: 24,
        configured_limit_unit: "graph_supersteps",
        observed_model_iterations: 3,
      },
    }),
  }), (error) => {
    assert.match(error.message, /Coder ReAct graph limit: 24 supersteps after 3 model iterations/);
    assert.equal(error.configuredLimitUnit, "graph_supersteps");
    return true;
  });
});

test("generation displays provider-reported reset timing without guessing the limit cause", async () => {
  const response = {
    ok: false,
    status: 429,
    json: async () => ({
      detail: {
        message: "The AI provider has temporarily limited this request.",
        category: "rate_limit",
        rate_limit_reset_seconds: {
          reset_requests_seconds: 13,
          reset_tokens_seconds: 120,
        },
      },
    }),
  };

  await assert.rejects(
    parseGenerateResponse(response),
    { message: "The AI provider has temporarily limited this request. Provider reset timing: request window ~13s; token window ~120s." },
  );
});

test("generation safely falls back for malformed structured provider errors", async () => {
  await assert.rejects(
    parseGenerateResponse({
      ok: false,
      json: async () => ({ detail: { message: 42, retry_after_seconds: "soon" } }),
    }),
    { message: "The request was rejected. Check your description and try again." },
  );
});

test("generation response parsing retains a safe fallback for malformed responses", async () => {
  await assert.rejects(
    parseGenerateResponse({ ok: false, json: async () => { throw new Error("invalid json"); } }),
    { message: "The API returned an unreadable response. Please try again." },
  );
});

test("generation shows a safe local API message for network failures", async () => {
  const appSource = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");

  assert.match(appSource, /requestError instanceof TypeError/);
  assert.match(appSource, /Could not reach the local API/);
});
