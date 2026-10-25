const DOWNLOAD_FALLBACK = "The project download failed. Please try again.";

export async function requestProjectArchive(jobId, apiBaseUrl, fetchImpl = fetch) {
  const baseUrl = apiBaseUrl.replace(/\/$/, "");
  const response = await fetchImpl(
    `${baseUrl}/jobs/${encodeURIComponent(jobId)}/download`,
  );

  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch {
      throw new Error(DOWNLOAD_FALLBACK);
    }
    throw new Error(
      typeof body?.detail === "string" ? body.detail : DOWNLOAD_FALLBACK,
    );
  }

  const contentType = response.headers?.get?.("content-type") || "";
  if (
    !contentType.toLowerCase().startsWith("application/zip") ||
    typeof response.blob !== "function"
  ) {
    throw new Error(DOWNLOAD_FALLBACK);
  }
  return response.blob();
}
