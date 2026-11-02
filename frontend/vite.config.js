import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

const sensitiveViteVariable = /^VITE_.*(?:KEY|SECRET|TOKEN|PASSWORD|PRIVATE|CREDENTIAL|BEARER|REDIS|DATABASE|CONNECTION[_-]?STRING)/i;

export default defineConfig(({ mode }) => {
  const publicBuildEnv = {
    ...loadEnv(mode, process.cwd(), ""),
    ...process.env,
  };
  const unsafeNames = Object.keys(publicBuildEnv).filter((name) =>
    sensitiveViteVariable.test(name),
  );
  if (unsafeNames.length) {
    throw new Error(
      `Refusing to expose sensitive Vite environment variable names: ${unsafeNames.join(", ")}`,
    );
  }

  return {
    envPrefix: "VITE_",
    plugins: [react()],
  };
});
