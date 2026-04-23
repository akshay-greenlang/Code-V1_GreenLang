/**
 * Track C-5 — OEM branding editor.
 *
 * Wires to:
 *   GET  /v1/oem/me           (load current branding)
 *   POST /v1/oem/branding     (save branding payload)
 *
 * Lets the OEM admin paste a logo URL, pick primary/secondary colours,
 * configure the custom domain + support contact, and preview how an
 * API response renders with the branding metadata embedded.
 *
 * The OEM identifier is read from localStorage (set during signup) and
 * sent as the ``X-OEM-Id`` header on every request. In production this
 * is replaced by the gateway's signed-token middleware.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Divider from "@mui/material/Divider";
import FormControlLabel from "@mui/material/FormControlLabel";
import Grid from "@mui/material/Grid";
import LinearProgress from "@mui/material/LinearProgress";
import Stack from "@mui/material/Stack";
import Switch from "@mui/material/Switch";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

interface BrandingPayload {
  logo_url?: string | null;
  primary_color?: string | null;
  secondary_color?: string | null;
  support_email?: string | null;
  support_url?: string | null;
  custom_domain?: string | null;
  attribution_required: boolean;
  powered_by_text: string;
}

interface MeResponse {
  id: string;
  branding?: BrandingPayload | null;
}

const DEFAULT_BRANDING: BrandingPayload = {
  logo_url: "",
  primary_color: "",
  secondary_color: "",
  support_email: "",
  support_url: "",
  custom_domain: "",
  attribution_required: true,
  powered_by_text: "Powered by GreenLang",
};

function readOemIdFromStorage(): string {
  try {
    return window.localStorage.getItem("greenlang.oem.id") ?? "";
  } catch {
    return "";
  }
}

export function OemBranding() {
  const [oemId, setOemId] = useState<string>(readOemIdFromStorage());
  const [branding, setBranding] = useState<BrandingPayload>(DEFAULT_BRANDING);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<Date | null>(null);

  useEffect(() => {
    if (!oemId) return;
    let cancelled = false;
    setLoading(true);
    fetch(`/v1/oem/me`, { headers: { "X-OEM-Id": oemId } })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const body = (await r.json()) as MeResponse;
        if (cancelled) return;
        if (body.branding) {
          setBranding({ ...DEFAULT_BRANDING, ...body.branding });
        }
        setError(null);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [oemId]);

  const previewResponse = useMemo(() => {
    const meta: Record<string, unknown> = {};
    Object.entries(branding).forEach(([key, value]) => {
      if (value === null || value === undefined || value === "") return;
      meta[key] = value;
    });
    if (oemId) meta.oem_id = oemId;
    return {
      data: [
        {
          factor_id: "epa_hub.electricity.us.kg_co2e_per_kwh",
          co2e_per_unit: 0.42,
          unit: "kg_co2e/kWh",
        },
      ],
      branding: meta,
    };
  }, [branding, oemId]);

  const update = (key: keyof BrandingPayload, value: string | boolean) => {
    setBranding((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    if (!oemId) {
      setError("OEM ID is required.");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      // Strip empty strings - backend treats them as None.
      const payload: Record<string, unknown> = { ...branding };
      Object.keys(payload).forEach((k) => {
        if (payload[k] === "") {
          delete payload[k];
        }
      });
      const response = await fetch(`/v1/oem/branding`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-OEM-Id": oemId,
        },
        body: JSON.stringify({ branding: payload }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `HTTP ${response.status}`);
      }
      setSavedAt(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Box sx={{ p: 3, maxWidth: 1100, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        OEM branding
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Configure the white-label payload that decorates every API response
        sent under your OEM key. Changes apply on the next request.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      {savedAt && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Branding saved at {savedAt.toLocaleTimeString()}.
        </Alert>
      )}
      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Editor
              </Typography>
              <Stack spacing={2}>
                <TextField
                  label="OEM ID"
                  value={oemId}
                  onChange={(e) => {
                    setOemId(e.target.value);
                    try {
                      window.localStorage.setItem(
                        "greenlang.oem.id",
                        e.target.value
                      );
                    } catch {
                      /* ignore */
                    }
                  }}
                  fullWidth
                  helperText="Sent as the X-OEM-Id header on every request."
                />
                <TextField
                  label="Logo URL (https://...)"
                  value={branding.logo_url ?? ""}
                  onChange={(e) => update("logo_url", e.target.value)}
                  fullWidth
                />
                <Stack direction="row" spacing={2}>
                  <TextField
                    label="Primary color"
                    value={branding.primary_color ?? ""}
                    onChange={(e) => update("primary_color", e.target.value)}
                    type="color"
                    sx={{ width: 140 }}
                  />
                  <TextField
                    label="Secondary color"
                    value={branding.secondary_color ?? ""}
                    onChange={(e) => update("secondary_color", e.target.value)}
                    type="color"
                    sx={{ width: 140 }}
                  />
                </Stack>
                <TextField
                  label="Support email"
                  value={branding.support_email ?? ""}
                  onChange={(e) => update("support_email", e.target.value)}
                  fullWidth
                  type="email"
                />
                <TextField
                  label="Support URL"
                  value={branding.support_url ?? ""}
                  onChange={(e) => update("support_url", e.target.value)}
                  fullWidth
                />
                <TextField
                  label="Custom domain (no scheme)"
                  value={branding.custom_domain ?? ""}
                  onChange={(e) => update("custom_domain", e.target.value)}
                  fullWidth
                  placeholder="factors.acme.com"
                />
                <TextField
                  label="Powered-by text"
                  value={branding.powered_by_text}
                  onChange={(e) => update("powered_by_text", e.target.value)}
                  fullWidth
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={branding.attribution_required}
                      onChange={(e) =>
                        update("attribution_required", e.target.checked)
                      }
                    />
                  }
                  label="Attribution required"
                />
                <Divider />
                <Stack direction="row" spacing={2}>
                  <Button
                    variant="contained"
                    onClick={handleSave}
                    disabled={saving || !oemId}
                  >
                    {saving ? "Saving..." : "Save branding"}
                  </Button>
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Live preview
              </Typography>
              <Box
                sx={{
                  p: 2,
                  borderRadius: 1,
                  border: "1px solid",
                  borderColor: "divider",
                  background: branding.secondary_color || "#fafafa",
                }}
              >
                <Stack direction="row" spacing={2} alignItems="center">
                  {branding.logo_url ? (
                    <img
                      src={branding.logo_url}
                      alt="OEM logo preview"
                      style={{ height: 40 }}
                    />
                  ) : (
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: 1,
                        background: branding.primary_color || "#cccccc",
                      }}
                    />
                  )}
                  <Typography
                    variant="subtitle1"
                    sx={{ color: branding.primary_color || "inherit" }}
                  >
                    {branding.custom_domain || "factors.example.com"}
                  </Typography>
                </Stack>
                {branding.attribution_required && (
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ mt: 1, display: "block" }}
                  >
                    {branding.powered_by_text}
                  </Typography>
                )}
              </Box>
              <Typography
                variant="subtitle2"
                sx={{ mt: 3, mb: 1 }}
                color="text.secondary"
              >
                Sample API response
              </Typography>
              <Box
                component="pre"
                sx={{
                  p: 2,
                  background: "#0d1117",
                  color: "#e6edf3",
                  borderRadius: 1,
                  overflow: "auto",
                  fontSize: 12,
                  maxHeight: 320,
                }}
              >
                {JSON.stringify(previewResponse, null, 2)}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
