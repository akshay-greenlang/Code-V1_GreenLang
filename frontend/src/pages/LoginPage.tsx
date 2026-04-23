/**
 * LoginPage — minimal placeholder so AdminGate has a redirect target.
 *
 * The real authentication flow lives in the hosted Factors API
 * (Track B-1) and stores its JWT under `gl.auth.token`. This page is
 * the inert landing spot that the gate sends unauthenticated users to;
 * once a real login UI ships it can replace this file.
 *
 * Owned by GL-FrontendDeveloper as part of the AdminGate wiring.
 */
import { useEffect, useState } from "react";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Link from "@mui/material/Link";
import Stack from "@mui/material/Stack";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

const TOKEN_STORAGE_KEY = "gl.auth.token";

export function LoginPage() {
  const [searchParams] = useSearchParams();
  const next = searchParams.get("next") || "/factors/explorer";
  const [token, setToken] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const cur = window.localStorage.getItem(TOKEN_STORAGE_KEY);
    if (cur) setToken(cur);
  }, []);

  const save = () => {
    if (typeof window === "undefined") return;
    if (token.trim()) {
      window.localStorage.setItem(TOKEN_STORAGE_KEY, token.trim());
    } else {
      window.localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
    setSaved(true);
  };

  return (
    <Box sx={{ p: 4, maxWidth: 720, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        Sign in
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Operator console is admin-gated. Paste a JWT with <code>role: admin</code> or the{" "}
        <code>factors:admin</code> scope, or switch to the Admin role from the shell header (dev
        only).
      </Typography>

      <Card>
        <CardContent>
          <Stack spacing={2}>
            <TextField
              fullWidth
              multiline
              minRows={3}
              label="JWT bearer token"
              placeholder="eyJhbGciOi..."
              value={token}
              onChange={(e) => setToken(e.target.value)}
            />
            <Stack direction="row" spacing={2}>
              <Button variant="contained" onClick={save}>
                Save token
              </Button>
              <Button
                variant="outlined"
                component={RouterLink}
                to={next}
                disabled={!saved && !token.trim()}
              >
                Continue to {next}
              </Button>
            </Stack>
            {saved && (
              <Alert severity="success">
                Token saved to <code>localStorage[gl.auth.token]</code>. Reload the destination
                page if it doesn't redirect automatically.
              </Alert>
            )}
            <Typography variant="caption" color="text.secondary">
              Need access? Contact a workspace admin or open the{" "}
              <Link component={RouterLink} to="/factors/status">
                public catalog status
              </Link>{" "}
              dashboard.
            </Typography>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}

export default LoginPage;
