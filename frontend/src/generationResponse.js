export async function parseGenerateResponse(response) {
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("The API returned an unreadable response. Please try again.");
  }

  if (!response.ok) {
    const detail = body?.detail;
    const message = typeof detail === "string"
      ? detail
      : typeof detail?.message === "string"
        ? detail.message
        : "The request was rejected. Check your description and try again.";
    const retryAfter = detail && typeof detail === "object"
      ? detail.retry_after_seconds
      : null;
    const safeDelay = Number.isFinite(retryAfter) && retryAfter >= 0 && retryAfter <= 86400
      ? retryAfter
      : null;
    let userMessage = message;
    if (safeDelay === 0) userMessage += " You can try again now.";
    else if (safeDelay !== null) {
      userMessage += ` Try again in about ${safeDelay} second${safeDelay === 1 ? "" : "s"}.`;
    }

    const resetTiming = detail?.category === "rate_limit" && detail.rate_limit_reset_seconds
      ? detail.rate_limit_reset_seconds
      : null;
    const requestReset = resetTiming?.reset_requests_seconds;
    const tokenReset = resetTiming?.reset_tokens_seconds;
    const validReset = (value) => Number.isInteger(value) && value >= 0 && value <= 86400;
    const resetParts = [];
    if (validReset(requestReset)) resetParts.push(`request window ~${requestReset}s`);
    if (validReset(tokenReset)) resetParts.push(`token window ~${tokenReset}s`);
    if (resetParts.length) userMessage += ` Provider reset timing: ${resetParts.join("; ")}.`;
    if (detail?.category === "workflow_limit") {
      const configured = detail.configured_limit;
      const stage = ["Planner", "Architect", "Coder", "unknown"].includes(detail.stage)
        ? detail.stage
        : "Workflow";
      if (Number.isInteger(configured) && configured >= 0 && configured <= 1000) {
        if (detail.error_type === "coder_step_limit") {
          const observed = detail.observed_model_iterations;
          if (Number.isInteger(observed) && observed >= 0 && observed <= 1000) {
            userMessage += ` (Coder model iteration limit: ${observed}/${configured}.)`;
          }
        } else if (detail.error_type === "coder_react_recursion_limit") {
          const observed = detail.observed_model_iterations;
          const observedText = Number.isInteger(observed) && observed >= 0 && observed <= 1000
            ? ` after ${observed} model iterations`
            : "";
          userMessage += ` (Coder ReAct graph limit: ${configured} supersteps${observedText}.)`;
        } else {
          const observed = detail.observed_node_steps;
          if (Number.isInteger(observed) && observed >= 0 && observed <= 1000) {
            userMessage += ` (${stage} outer graph limit: ${observed}/${configured} node steps.)`;
          }
        }
      }
    }

    const error = new Error(userMessage);
    const safeCategories = new Set([
      "rate_limit",
      "authentication",
      "invalid_request",
      "provider_unavailable",
      "provider_error",
      "provider_timeout",
      "provider_connection",
      "workflow_limit",
    ]);
    if (safeCategories.has(detail?.category)) error.category = detail.category;
    if (safeDelay !== null) error.retryAfterSeconds = safeDelay;
    if (resetParts.length) {
      error.rateLimitResetSeconds = {
        ...(validReset(requestReset) ? { reset_requests_seconds: requestReset } : {}),
        ...(validReset(tokenReset) ? { reset_tokens_seconds: tokenReset } : {}),
      };
    }
    if (["Planner", "Architect", "Coder", "repair"].includes(detail?.stage)) {
      error.stage = detail.stage;
    }
    if (Number.isInteger(detail?.configured_limit) && detail.configured_limit >= 0 && detail.configured_limit <= 1000) {
      error.configuredLimit = detail.configured_limit;
    }
    if (["graph_supersteps", "model_iterations"].includes(detail?.configured_limit_unit)) {
      error.configuredLimitUnit = detail.configured_limit_unit;
    }
    if (Number.isInteger(detail?.observed_node_steps) && detail.observed_node_steps >= 0 && detail.observed_node_steps <= 1000) {
      error.observedNodeSteps = detail.observed_node_steps;
    }
    if (Number.isInteger(detail?.observed_model_iterations) && detail.observed_model_iterations >= 0 && detail.observed_model_iterations <= 1000) {
      error.observedModelIterations = detail.observed_model_iterations;
    }
    throw error;
  }

  return body;
}
