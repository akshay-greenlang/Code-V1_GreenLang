import path from "node:path";
import type { StorybookConfig } from "@storybook/react-vite";

const config: StorybookConfig = {
  stories: ["../packages/shell-ui/src/**/*.stories.@(ts|tsx)", "../src/**/*.stories.@(ts|tsx)"],
  addons: ["@storybook/addon-essentials"],
  framework: {
    name: "@storybook/react-vite",
    options: {}
  },
  async viteFinal(viteConfig) {
    const shellUi = path.resolve(__dirname, "../packages/shell-ui/src/index.ts");
    const alias = viteConfig.resolve?.alias;
    const nextAlias = Array.isArray(alias)
      ? [...alias, { find: "@greenlang/shell-ui", replacement: shellUi }]
      : { ...(alias as Record<string, string> | undefined), "@greenlang/shell-ui": shellUi };
    return {
      ...viteConfig,
      resolve: { ...viteConfig.resolve, alias: nextAlias }
    };
  }
};

export default config;
