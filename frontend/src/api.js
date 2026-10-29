export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function getJson(url) {
  const response = await fetch(url);
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error("The API returned an unreadable response.");
  }
  if (!response.ok) {
    throw new Error(
      typeof body?.detail === "string" ? body.detail : "The file request failed.",
    );
  }
  return body;
}

export function getJobFiles(jobId) {
  return getJson(`${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/files`);
}

export function getJobFile(jobId, filePath) {
  const encodedPath = filePath.split("/").map(encodeURIComponent).join("/");
  return getJson(
    `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/files/${encodedPath}`,
  );
}

export async function getJobPreviewAsset(jobId, filePath) {
  const encodedPath = filePath.split("/").map(encodeURIComponent).join("/");
  const response = await fetch(
    `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/preview-assets/${encodedPath}`,
  );
  if (!response.ok) {
    let body;
    try { body = await response.json(); } catch { /* Use the safe fallback below. */ }
    throw new Error(
      typeof body?.detail === "string" ? body.detail : "A preview asset could not be loaded.",
    );
  }
  const contentType = response.headers.get("content-type")?.split(";")[0].toLowerCase();
  if (!contentType || !/^(image|font|audio|video)\//.test(contentType)) {
    throw new Error("The API returned an unsupported preview asset.");
  }
  return { content: await response.arrayBuffer(), contentType };
}
