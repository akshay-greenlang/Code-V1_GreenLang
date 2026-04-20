import { useEffect, useMemo, useState } from "react";
import { Link as RouterLink, Outlet } from "react-router-dom";
import Alert from "@mui/material/Alert";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import InputBase from "@mui/material/InputBase";
import Link from "@mui/material/Link";
import Paper from "@mui/material/Paper";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import { ShellConnectorIncidentsAlert } from "@greenlang/shell-ui";
import { fetchShellChromeContext, type ShellChromeContext } from "../api";
import { readRoleFromStorage, roleRouteAllowlist, type ShellRole, writeRoleToStorage } from "../authz";

const links = [
  { label: "CBAM", to: "/apps/cbam" },
  { label: "CSRD", to: "/apps/csrd" },
  { label: "VCCI", to: "/apps/vcci" },
  { label: "EUDR", to: "/apps/eudr" },
  { label: "GHG", to: "/apps/ghg" },
  { label: "ISO14064", to: "/apps/iso14064" },
  { label: "SB253", to: "/apps/sb253" },
  { label: "Taxonomy", to: "/apps/taxonomy" },
  { label: "Runs", to: "/runs" },
  { label: "Governance", to: "/governance" },
  { label: "Admin", to: "/admin" },
  { label: "Factors", to: "/factors/explorer" },
  { label: "Catalog", to: "/factors/status" }
];

export function ShellLayout() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [role, setRole] = useState<ShellRole>(readRoleFromStorage());
  const [highContrast, setHighContrast] = useState(false);
  const [apiHealthy, setApiHealthy] = useState<boolean | null>(null);
  const [chromeContext, setChromeContext] = useState<ShellChromeContext | null>(null);

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      try {
        const response = await fetch("/health", { method: "GET", cache: "no-store" });
        if (!cancelled) setApiHealthy(response.ok);
      } catch {
        if (!cancelled) setApiHealthy(false);
      }
    };
    void ping();
    const timer = window.setInterval(() => void ping(), 60_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      fetchShellChromeContext()
        .then((ctx) => {
          if (!cancelled) setChromeContext(ctx);
        })
        .catch(() => {
          if (!cancelled) setChromeContext(null);
        });
    };
    load();
    const timer = window.setInterval(load, 90_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const visibleLinks = useMemo(() => {
    const allowed = new Set(roleRouteAllowlist[role]);
    return links.filter((item) => allowed.has(item.to));
  }, [role]);

  const filteredLinks = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return visibleLinks;
    return visibleLinks.filter((item) => item.label.toLowerCase().includes(q) || item.to.includes(q));
  }, [query, visibleLinks]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        backgroundColor: "background.default",
        filter: highContrast ? "contrast(1.25)" : "none"
      }}
    >
      <Link
        href="#main-content"
        sx={{
          position: "absolute",
          left: 8,
          top: -40,
          zIndex: 2000,
          px: 1,
          py: 0.5,
          backgroundColor: "background.paper",
          "&:focus": { top: 8 }
        }}
      >
        Skip to main content
      </Link>
      <AppBar position="sticky">
        <Toolbar>
          <Typography variant="h6" sx={{ mr: 2 }}>
            GreenLang V2.2 Shell
          </Typography>
          <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
            {visibleLinks.map((item) => (
              <Link
                key={item.to}
                component={RouterLink}
                to={item.to}
                underline="hover"
                color="inherit"
                sx={{ fontSize: 14 }}
              >
                {item.label}
              </Link>
            ))}
          </Stack>
          <Stack direction="row" spacing={1} sx={{ ml: "auto" }}>
            <FormControl size="small" sx={{ minWidth: 160 }} variant="outlined">
              <InputLabel id="gl-shell-role-label" sx={{ color: "#e2e8f0" }}>
                Role
              </InputLabel>
              <Select
                labelId="gl-shell-role-label"
                id="gl-shell-role"
                label="Role"
                value={role}
                onChange={(event) => {
                  const nextRole = event.target.value as ShellRole;
                  setRole(nextRole);
                  writeRoleToStorage(nextRole);
                }}
                sx={{ color: "white" }}
              >
                <MenuItem value="operator">Operator</MenuItem>
                <MenuItem value="auditor">Auditor</MenuItem>
                <MenuItem value="compliance">Compliance Lead</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>
            <Button size="small" variant="outlined" color="inherit" onClick={() => setHighContrast((v) => !v)}>
              Contrast
            </Button>
            <Button size="small" variant="outlined" color="inherit" onClick={() => setPaletteOpen(true)}>
              Cmd+K
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>
      {apiHealthy === false && (
        <Alert severity="warning" role="status" sx={{ borderRadius: 0 }}>
          API health check failed. Navigation may work, but runs and governance data may be stale until the service
          recovers.
        </Alert>
      )}
      {chromeContext && (
        <Paper
          component="aside"
          square
          elevation={0}
          sx={{
            borderBottom: "1px solid",
            borderColor: "divider",
            px: 2,
            py: 1,
            bgcolor: "background.paper"
          }}
          aria-label="Compliance and policy summary"
        >
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" useFlexGap>
            <Typography variant="caption" color="text.secondary">
              Managed packs: <strong>{chromeContext.compliance_rail.managed_pack_count}</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Policy bundles: <strong>{chromeContext.compliance_rail.policy_bundle_count}</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Deprecated agents: <strong>{chromeContext.compliance_rail.deprecated_agent_count}</strong>
            </Typography>
            <Link component={RouterLink} to="/governance" variant="caption" underline="hover">
              Open governance center
            </Link>
          </Stack>
        </Paper>
      )}
      <ShellConnectorIncidentsAlert
        incidents={chromeContext?.connector_incidents ?? []}
        adminLink={
          chromeContext?.connector_incidents?.length ? (
            <Link component={RouterLink} to="/admin" variant="body2" sx={{ mt: 0.5, display: "inline-block" }}>
              View admin console for registry details
            </Link>
          ) : undefined
        }
      />
      <Container maxWidth="xl" sx={{ py: 2 }} id="main-content" component="main" aria-label="Workspace content">
        <Typography variant="caption" color="text.secondary" aria-live="polite">
          Active role: {role}
        </Typography>
        <Outlet />
      </Container>
      <Dialog
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        fullWidth
        maxWidth="sm"
        aria-labelledby="command-palette-title"
      >
        <DialogTitle id="command-palette-title">Command Palette</DialogTitle>
        <DialogContent>
          <InputBase
            autoFocus
            fullWidth
            aria-label="Command search"
            placeholder="Search apps, runs, artifacts, docs..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            sx={{ border: "1px solid #495e8d", borderRadius: 1, px: 1.5, py: 1, mb: 2 }}
          />
          <Stack spacing={1}>
            {filteredLinks.map((item) => (
              <Link key={item.to} component={RouterLink} to={item.to} onClick={() => setPaletteOpen(false)}>
                {item.label} ({item.to})
              </Link>
            ))}
          </Stack>
        </DialogContent>
      </Dialog>
    </Box>
  );
}
