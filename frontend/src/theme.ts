import { createTheme } from "@mui/material/styles";

export const shellTheme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#4f7cff" },
    secondary: { main: "#00d4b4" },
    background: { default: "#0a1222", paper: "#111b33" }
  },
  shape: { borderRadius: 12 }
});
