import { createTheme } from "@mui/material/styles";
import { shellColorTokens, shellRadii } from "./tokens";

/** App chrome contrast tuned for WCAG AA with white/near-white nav labels. */
const appBarBg = "#0a1628";
const appBarFg = "#f1f5f9";

export function createShellTheme() {
  return createTheme({
    palette: {
      mode: "dark",
      primary: { main: shellColorTokens.primary },
      secondary: { main: shellColorTokens.secondary },
      background: { default: shellColorTokens.backgroundDefault, paper: shellColorTokens.backgroundPaper },
      text: {
        primary: "#e8eef8",
        secondary: "#b8c5dc"
      }
    },
    shape: { borderRadius: shellRadii.md },
    components: {
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: appBarBg,
            color: appBarFg
          }
        }
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            "& .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(255,255,255,0.55)"
            },
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(255,255,255,0.75)"
            },
            "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
              borderColor: "rgba(255,255,255,0.95)"
            }
          }
        }
      }
    }
  });
}

export const shellTheme = createShellTheme();
