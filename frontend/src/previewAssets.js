const PREVIEW_ASSET_MIME_TYPES = new Set([
  "avif", "gif", "ico", "jpeg", "jpg", "mp3", "mp4", "oga", "ogg",
  "opus", "otf", "png", "ttf", "wav", "webm", "webp", "woff", "woff2",
]);

export const PREVIEW_MAX_RESOURCE_REQUESTS = 30;

/**
 * Tracks actual per-job file requests independently from their results. Failed
 * requests are cached too, so repeated missing references neither retry nor
 * bypass the request budget.
 */
export function createPreviewResourceLoader(limit = PREVIEW_MAX_RESOURCE_REQUESTS) {
  const attempted = new Set();
  const results = new Map();

  return {
    async load(path, loader) {
      if (results.has(path)) return results.get(path);
      if (attempted.size >= limit) return { limited: true, value: null };

      attempted.add(path);
      const pending = Promise.resolve()
        .then(loader)
        .then((value) => ({ value }))
        .catch((error) => ({ error, value: null }));
      results.set(path, pending);
      const result = await pending;
      results.set(path, result);
      return result;
    },
    get attemptedCount() {
      return attempted.size;
    },
  };
}

export function formatOmittedAssetNotice(count) {
  if (!Number.isInteger(count) || count <= 0) return "";
  return `${count} missing or unsupported local asset${count === 1 ? " was" : "s were"} omitted from this preview.`;
}

export const CSS_IMPORT_PREVIEW_NOTICE =
  "CSS @import rules are not supported in the sandboxed preview. Combine imported styles into one stylesheet to preview this project.";
export const MODULE_PREVIEW_NOTICE =
  "JavaScript module scripts are not supported in the sandboxed preview because relative module imports cannot be safely bundled.";

function cssIdentifierAfterAt(css, start) {
  let value = "";
  let index = start;
  while (index < css.length) {
    const character = css[index];
    if (/[a-z0-9_-]/i.test(character) || character.charCodeAt(0) >= 0x80) {
      value += character;
      index += 1;
      continue;
    }
    if (character !== "\\" || index + 1 >= css.length) break;

    index += 1;
    if (/[\da-f]/i.test(css[index])) {
      const startHex = index;
      while (index < css.length && index - startHex < 6 && /[\da-f]/i.test(css[index])) index += 1;
      const codePoint = Number.parseInt(css.slice(startHex, index), 16);
      value += String.fromCodePoint(codePoint === 0 || codePoint > 0x10ffff ? 0xfffd : codePoint);
      if (/\s/.test(css[index] || "")) {
        if (css[index] === "\r" && css[index + 1] === "\n") index += 1;
        index += 1;
      }
    } else {
      value += css[index];
      index += 1;
    }
  }
  return { value, index };
}

export function containsCssImportRule(css) {
  let quote = "";
  for (let index = 0; index < css.length; index += 1) {
    const character = css[index];
    if (quote) {
      if (character === "\\") index += 1;
      else if (character === quote) quote = "";
      continue;
    }
    if (character === "/" && css[index + 1] === "*") {
      const end = css.indexOf("*/", index + 2);
      if (end === -1) return false;
      index = end + 1;
      continue;
    }
    if (character === "'" || character === '"') {
      quote = character;
      continue;
    }
    if (character === "@") {
      const identifier = cssIdentifierAfterAt(css, index + 1);
      if (identifier.value.toLowerCase() === "import") return true;
      index = identifier.index - 1;
    }
  }
  return false;
}

export function isNonLocalReference(reference) {
  const value = reference?.trim() || "";
  return /^(?:data:|blob:|#|https?:|\/\/)/i.test(value);
}

export function resolveProjectResource(reference, sourcePath, filePaths) {
  const value = reference?.trim();
  if (!value || isNonLocalReference(value) || value.startsWith("\\") || value.includes("\\") || value.includes("\0")) {
    return null;
  }

  try {
    const withoutSuffix = value.split(/[?#]/, 1)[0];
    const decoded = decodeURIComponent(withoutSuffix);
    if (decoded.includes("\\") || decoded.includes("\0")) return null;
    const segments = decoded.startsWith("/")
      ? []
      : sourcePath.split("/").slice(0, -1);
    for (const segment of decoded.replace(/^\/+/, "").split("/")) {
      if (!segment || segment === ".") continue;
      if (segment === "..") {
        if (!segments.length) return null;
        segments.pop();
      } else {
        segments.push(segment);
      }
    }
    const normalized = segments.join("/");
    return normalized && filePaths.has(normalized) ? normalized : null;
  } catch {
    return null;
  }
}

export function isPreviewAssetPath(path) {
  const extension = path?.split(".").pop()?.toLowerCase();
  return PREVIEW_ASSET_MIME_TYPES.has(extension);
}
