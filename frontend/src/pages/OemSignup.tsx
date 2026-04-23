/**
 * Track C-5 — OEM signup multi-step form.
 *
 * Wires to:
 *   POST /v1/oem/signup
 *
 * Steps:
 *   1. Company info (name + brief description)
 *   2. Primary contact (email)
 *   3. Redistribution grants (per-license-class checkboxes with explanations)
 *   4. Review + submit
 *
 * On success the freshly minted OEM API key is shown ONCE and the user
 * is invited to copy it; the backend will not return it again.
 */
import { useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Checkbox from "@mui/material/Checkbox";
import Chip from "@mui/material/Chip";
import Divider from "@mui/material/Divider";
import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Step from "@mui/material/Step";
import StepLabel from "@mui/material/StepLabel";
import Stepper from "@mui/material/Stepper";
import TextField from "@mui/material/TextField";
import Typography from "@mui/material/Typography";

const STEPS = ["Company", "Contact", "Grants", "Review"] as const;

const PARENT_PLANS: Array<{ value: string; label: string; blurb: string }> = [
  {
    value: "consulting_platform",
    label: "Consulting Platform",
    blurb: "$25k-$75k ACV; 25 sub-tenants included; 99.9% SLA.",
  },
  {
    value: "platform",
    label: "Platform",
    blurb: "$50k+ ACV; 100 sub-tenants; OEM uplift baked in.",
  },
  {
    value: "enterprise",
    label: "Enterprise",
    blurb: "$75k+ ACV; unlimited tenants; redistributable rights.",
  },
];

interface GrantOption {
  value: string;
  label: string;
  description: string;
}

const GRANT_OPTIONS: GrantOption[] = [
  {
    value: "open",
    label: "Open data",
    description:
      "Public-domain factors (EPA, DESNZ, eGRID, IPCC). Always redistributable.",
  },
  {
    value: "public_us_government",
    label: "US Government (public)",
    description:
      "EPA Hub, eGRID, DOE. Citation required; redistribution allowed.",
  },
  {
    value: "uk_open_government",
    label: "UK Open Government",
    description:
      "DESNZ GHG conversion factors. Open Government License v3.0.",
  },
  {
    value: "eu_publication",
    label: "EU publications",
    description:
      "EEA / JRC datasets. Citation + share-alike requirements.",
  },
  {
    value: "public_international",
    label: "International (public)",
    description: "IPCC, IEA Stats, FAO. Citation required.",
  },
  {
    value: "public_in_government",
    label: "India Government",
    description: "MoEF / CEA / BEE datasets. Citation required.",
  },
  {
    value: "wri_wbcsd_terms",
    label: "WRI / WBCSD",
    description:
      "GHG Protocol calculation tools. Attribution required; redistribution restricted.",
  },
  {
    value: "smart_freight_terms",
    label: "Smart Freight Centre",
    description:
      "GLEC / ISO 14083 freight factors. Member-only; commercial restrictions.",
  },
  {
    value: "registry_terms",
    label: "Registries",
    description:
      "Verra / Gold Standard / climate registries. Per-program terms apply.",
  },
  {
    value: "pcaf_attribution",
    label: "PCAF (financed emissions)",
    description:
      "PCAF attribution chain required. Customer must hold their own license.",
  },
  {
    value: "greenlang_terms",
    label: "GreenLang Terms",
    description:
      "Restricted commercial. Redistributable only with signed OEM agreement.",
  },
  {
    value: "commercial_connector",
    label: "Commercial connectors",
    description:
      "Live API only (ecoinvent, Sphera, GaBi). Never redistributed.",
  },
  {
    value: "licensed",
    label: "Licensed (umbrella)",
    description:
      "Catch-all for any factor whose source requires a customer license chain.",
  },
];

interface SignupResult {
  id: string;
  api_key: string;
  parent_plan: string;
  grant: { allowed_classes: string[] };
}

export function OemSignup() {
  const [activeStep, setActiveStep] = useState(0);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [parentPlan, setParentPlan] = useState<string>(PARENT_PLANS[0].value);
  const [selectedGrants, setSelectedGrants] = useState<Set<string>>(new Set(["open"]));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SignupResult | null>(null);

  const grantsArray = useMemo(() => Array.from(selectedGrants), [selectedGrants]);

  const canAdvance = useMemo(() => {
    if (activeStep === 0) return name.trim().length > 0;
    if (activeStep === 1) return /\S+@\S+\.\S+/.test(contactEmail);
    if (activeStep === 2) return selectedGrants.size > 0;
    return true;
  }, [activeStep, name, contactEmail, selectedGrants]);

  const handleToggleGrant = (value: string) => {
    setSelectedGrants((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/v1/oem/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          contact_email: contactEmail.trim(),
          redistribution_grants: grantsArray,
          parent_plan: parentPlan,
          notes: description.trim() || null,
        }),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Signup failed: HTTP ${response.status}`);
      }
      const body = (await response.json()) as SignupResult;
      setResult(body);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  if (result) {
    return (
      <Box sx={{ p: 3, maxWidth: 760, mx: "auto" }}>
        <Card>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              OEM partner provisioned
            </Typography>
            <Alert severity="success" sx={{ my: 2 }}>
              Save the API key below NOW — it will not be shown again.
            </Alert>
            <Stack spacing={1}>
              <Typography variant="body2">
                <strong>OEM ID:</strong> <code>{result.id}</code>
              </Typography>
              <Typography variant="body2">
                <strong>API key:</strong>{" "}
                <code style={{ wordBreak: "break-all" }}>{result.api_key}</code>
              </Typography>
              <Typography variant="body2">
                <strong>Parent plan:</strong> {result.parent_plan}
              </Typography>
              <Typography variant="body2">
                <strong>Grants:</strong>{" "}
                {result.grant.allowed_classes.map((cls) => (
                  <Chip key={cls} label={cls} size="small" sx={{ mr: 0.5 }} />
                ))}
              </Typography>
            </Stack>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3, maxWidth: 760, mx: "auto" }}>
      <Typography variant="h4" gutterBottom>
        OEM partner signup
      </Typography>
      <Stepper activeStep={activeStep} sx={{ my: 3 }}>
        {STEPS.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card>
        <CardContent>
          {activeStep === 0 && (
            <Stack spacing={2}>
              <Typography variant="h6">Company info</Typography>
              <TextField
                label="Company name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                fullWidth
                required
              />
              <TextField
                label="Description (optional)"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                fullWidth
                multiline
                minRows={2}
              />
              <FormControl fullWidth>
                <InputLabel id="oem-plan-label">Parent plan</InputLabel>
                <Select<string>
                  labelId="oem-plan-label"
                  label="Parent plan"
                  value={parentPlan}
                  onChange={(e) => setParentPlan(String(e.target.value))}
                >
                  {PARENT_PLANS.map((p) => (
                    <MenuItem key={p.value} value={p.value}>
                      {p.label} — {p.blurb}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Stack>
          )}

          {activeStep === 1 && (
            <Stack spacing={2}>
              <Typography variant="h6">Primary contact</Typography>
              <TextField
                label="Contact email"
                value={contactEmail}
                onChange={(e) => setContactEmail(e.target.value)}
                fullWidth
                required
                type="email"
              />
              <Typography variant="body2" color="text.secondary">
                Used for OEM admin notifications, license-renewal alerts and
                billing reconciliation. Add additional admins later from the
                OEM console.
              </Typography>
            </Stack>
          )}

          {activeStep === 2 && (
            <Stack spacing={2}>
              <Typography variant="h6">Redistribution grants</Typography>
              <Typography variant="body2" color="text.secondary">
                Pick the license classes your sub-tenants are allowed to
                resolve. You cannot grant your own customers more than what
                you select here.
              </Typography>
              <FormGroup>
                {GRANT_OPTIONS.map((opt) => (
                  <FormControlLabel
                    key={opt.value}
                    control={
                      <Checkbox
                        checked={selectedGrants.has(opt.value)}
                        onChange={() => handleToggleGrant(opt.value)}
                      />
                    }
                    label={
                      <Box>
                        <Typography variant="body1">{opt.label}</Typography>
                        <Typography variant="caption" color="text.secondary">
                          {opt.description}
                        </Typography>
                      </Box>
                    }
                  />
                ))}
              </FormGroup>
            </Stack>
          )}

          {activeStep === 3 && (
            <Stack spacing={2}>
              <Typography variant="h6">Review</Typography>
              <Divider />
              <Typography variant="body2">
                <strong>Company:</strong> {name}
              </Typography>
              {description && (
                <Typography variant="body2">
                  <strong>Description:</strong> {description}
                </Typography>
              )}
              <Typography variant="body2">
                <strong>Contact:</strong> {contactEmail}
              </Typography>
              <Typography variant="body2">
                <strong>Parent plan:</strong> {parentPlan}
              </Typography>
              <Typography variant="body2">
                <strong>Grants:</strong>
              </Typography>
              <Box>
                {grantsArray.map((g) => (
                  <Chip key={g} label={g} size="small" sx={{ mr: 0.5, mb: 0.5 }} />
                ))}
              </Box>
            </Stack>
          )}
        </CardContent>
      </Card>

      <Stack direction="row" spacing={2} sx={{ mt: 3, justifyContent: "space-between" }}>
        <Button
          variant="outlined"
          disabled={activeStep === 0 || submitting}
          onClick={() => setActiveStep((s) => s - 1)}
        >
          Back
        </Button>
        {activeStep < STEPS.length - 1 ? (
          <Button
            variant="contained"
            disabled={!canAdvance}
            onClick={() => setActiveStep((s) => s + 1)}
          >
            Next
          </Button>
        ) : (
          <Button
            variant="contained"
            color="primary"
            disabled={!canAdvance || submitting}
            onClick={handleSubmit}
          >
            {submitting ? "Provisioning..." : "Submit"}
          </Button>
        )}
      </Stack>
    </Box>
  );
}
