export const THEME_STORAGE_KEY = "thought2code-theme";
export const DEFAULT_THEME = "dark";

export function getInitialTheme(storage) {
  try {
    const savedTheme = (storage ?? globalThis.localStorage)?.getItem(THEME_STORAGE_KEY);
    return savedTheme === "dark" || savedTheme === "light"
      ? savedTheme
      : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

export function persistTheme(theme, storage) {
  if (theme !== "dark" && theme !== "light") return false;
  try {
    const target = storage ?? globalThis.localStorage;
    if (!target) return false;
    target.setItem(THEME_STORAGE_KEY, theme);
    return true;
  } catch {
    return false;
  }
}
