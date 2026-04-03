import { useEffect, useMemo, useState } from "react";
import { Link as RouterLink, Outlet } from "react-router-dom";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import InputBase from "@mui/material/InputBase";
import Link from "@mui/material/Link";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import { readRoleFromStorage, roleRouteAllowlist, type ShellRole, writeRoleToStorage } from "../authz";

const links = [
  { label: "CBAM", to: "/apps/cbam" },
  { label: "CSRD", to: "/apps/csrd" },
  { label: "VCCI", to: "/apps/vcci" },
  { label: "EUDR", to: "/apps/eudr" },
  { label: "GHG", to: "/apps/ghg" },
  { label: "ISO14064", to: "/apps/iso14064" },
  { label: "Runs", to: "/runs" },
  { label: "Governance", to: "/governance" },
  { label: "Admin", to: "/admin" }
];

export function ShellLayout() {
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [role, setRole] = useState<ShellRole>(readRoleFromStorage());
  const [highContrast, setHighContrast] = useState(false);

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
            <Select
              size="small"
              value={role}
              aria-label="Role selector"
              onChange={(event) => {
                const nextRole = event.target.value as ShellRole;
                setRole(nextRole);
                writeRoleToStorage(nextRole);
              }}
              sx={{ color: "white", minWidth: 140 }}
            >
              <MenuItem value="operator">Operator</MenuItem>
              <MenuItem value="auditor">Auditor</MenuItem>
              <MenuItem value="compliance">Compliance Lead</MenuItem>
              <MenuItem value="admin">Admin</MenuItem>
            </Select>
            <Button size="small" variant="outlined" color="inherit" onClick={() => setHighContrast((v) => !v)}>
              Contrast
            </Button>
            <Button size="small" variant="outlined" color="inherit" onClick={() => setPaletteOpen(true)}>
              Cmd+K
            </Button>
          </Stack>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ py: 2 }} id="main-content" component="main" aria-label="Workspace content">
        <Typography variant="caption" color="text.secondary">
          Active role: {role}
        </Typography>
        <Outlet />
      </Container>
      <Dialog open={paletteOpen} onClose={() => setPaletteOpen(false)} fullWidth maxWidth="sm">
        <DialogTitle>Command Palette</DialogTitle>
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
