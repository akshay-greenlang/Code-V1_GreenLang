/** Design tokens for the GreenLang V2 enterprise shell (consumable primitives). */
export const shellColorTokens = {
  primary: "#4f7cff",
  secondary: "#00d4b4",
  backgroundDefault: "#0a1222",
  backgroundPaper: "#111b33"
} as const;

export const shellRadii = {
  sm: 8,
  md: 12,
  lg: 16
} as const;

export const shellFocusRing = {
  outline: `2px solid ${shellColorTokens.primary}`,
  outlineOffset: 2
} as const;
