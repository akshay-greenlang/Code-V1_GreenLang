/**
 * CheckoutSuccess -- the landing page Stripe redirects to after a successful
 * Checkout Session.
 *
 * The page reads the session id and (if available) the freshly-issued API
 * key from the URL, then renders a Quickstart with curl + python + ts
 * snippets so the user can move from credit card to first API call in
 * under 60 seconds.
 *
 * The page DOES NOT call any backend on mount because Stripe's
 * `checkout.session.completed` webhook is the source of truth for
 * provisioning. We only display what the URL tells us.
 *
 * URL contract:
 *   /checkout/success?plan=<plan_id>&session_id=<cs_...>&api_key=<gl_...>
 *
 * The ``api_key`` query parameter is rendered exactly once (here) and the
 * server sets a one-time-show flag so subsequent views require the user
 * to mint a fresh key from the dev portal.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import Container from "@mui/material/Container";
import IconButton from "@mui/material/IconButton";
import Stack from "@mui/material/Stack";
import Tab from "@mui/material/Tab";
import Tabs from "@mui/material/Tabs";
import Tooltip from "@mui/material/Tooltip";
import Typography from "@mui/material/Typography";

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

interface CheckoutSummary {
  planId: string;
  sessionId: string;
  apiKey: string;
}

function readQueryParams(): CheckoutSummary {
  if (typeof window === "undefined") {
    return { planId: "", sessionId: "", apiKey: "" };
  }
  const params = new URLSearchParams(window.location.search);
  return {
    planId: params.get("plan") ?? "",
    sessionId: params.get("session_id") ?? "",
    apiKey: params.get("api_key") ?? "",
  };
}

// ---------------------------------------------------------------------------
// Quickstart snippets
// ---------------------------------------------------------------------------

function curlSnippet(apiKey: string): string {
  const safeKey = apiKey || "$GL_FACTORS_API_KEY";
  return [
    "# Search the catalog",
    `curl -s "https://api.greenlang.io/api/v1/factors/search?q=natural+gas&limit=3" \\`,
    `  -H "X-API-Key: ${safeKey}" \\`,
    `  -H "Accept: application/json" | jq .`,
    "",
    "# Resolve an activity into a chosen factor",
    `curl -s "https://api.greenlang.io/api/v1/factors/resolve-explain" \\`,
    `  -H "X-API-Key: ${safeKey}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  --data '{"activity":"natural gas combustion","jurisdiction":"US","method_profile":"corporate_scope1","quantity":1000,"unit":"therm"}' | jq .`,
  ].join("\n");
}

function pythonSnippet(apiKey: string): string {
  const safeKey = apiKey || 'os.environ["GL_FACTORS_API_KEY"]';
  return [
    "pip install greenlang-factors==1.0.0",
    "",
    "# quickstart.py",
    "import os",
    "from greenlang_factors import FactorsClient",
    "",
    'with FactorsClient(',
    '    base_url="https://api.greenlang.io",',
    `    api_key=${apiKey ? `"${safeKey}"` : safeKey},`,
    ") as client:",
    '    hits = client.search("natural gas US Scope 1", limit=3)',
    "    for f in hits.factors:",
    "        print(f.factor_id, f.co2e_per_unit, f.unit)",
  ].join("\n");
}

function tsSnippet(apiKey: string): string {
  const safeKey = apiKey || "process.env.GL_FACTORS_API_KEY!";
  return [
    "npm install @greenlang/factors@1.0.0",
    "",
    "// quickstart.ts",
    'import { FactorsClient } from "@greenlang/factors";',
    "",
    "const client = new FactorsClient({",
    '  baseUrl: "https://api.greenlang.io",',
    `  apiKey: ${apiKey ? `"${safeKey}"` : safeKey},`,
    "});",
    "",
    'const hits = await client.search("natural gas US Scope 1", { limit: 3 });',
    "for (const f of hits.factors) {",
    "  console.log(f.factor_id, f.co2e_per_unit, f.unit);",
    "}",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CheckoutSuccess(): JSX.Element {
  const [summary, setSummary] = useState<CheckoutSummary | null>(null);
  const [tab, setTab] = useState<number>(0);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    setSummary(readQueryParams());
  }, []);

  const snippets = useMemo(() => {
    const apiKey = summary?.apiKey ?? "";
    return [
      { label: "curl", code: curlSnippet(apiKey) },
      { label: "Python", code: pythonSnippet(apiKey) },
      { label: "TypeScript", code: tsSnippet(apiKey) },
    ];
  }, [summary]);

  const handleCopy = (label: string, code: string): void => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      void navigator.clipboard.writeText(code);
      setCopied(label);
      setTimeout(() => setCopied(null), 1500);
    }
  };

  if (summary === null) {
    return (
      <Container maxWidth="md" sx={{ py: 6 }}>
        <Typography>Loading...</Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      <Stack spacing={4}>
        <Stack spacing={1}>
          <Typography variant="h3" component="h1" gutterBottom>
            You are in. Welcome to GreenLang Factors.
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Your subscription has been activated. Below is your API key and a
            Quickstart in three flavours -- pick whichever fits your stack and
            you should be calling the API in under sixty seconds.
          </Typography>
        </Stack>

        {summary.apiKey ? (
          <Card variant="outlined">
            <CardContent>
              <Stack spacing={1}>
                <Typography variant="overline" color="text.secondary">
                  Your API key (shown once)
                </Typography>
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Box
                    component="code"
                    sx={{
                      p: 1.5,
                      bgcolor: "grey.100",
                      borderRadius: 1,
                      fontFamily: "monospace",
                      flexGrow: 1,
                      overflowX: "auto",
                      wordBreak: "break-all",
                    }}
                    data-testid="api-key-display"
                  >
                    {summary.apiKey}
                  </Box>
                  <Tooltip title={copied === "key" ? "Copied" : "Copy"}>
                    <IconButton
                      onClick={() => handleCopy("key", summary.apiKey)}
                      aria-label="Copy API key"
                    >
                      <span aria-hidden>copy</span>
                    </IconButton>
                  </Tooltip>
                </Stack>
                <Alert severity="warning">
                  This is the only time we will show this key. Save it in your
                  secret manager (1Password, Vault, AWS Secrets Manager) before
                  leaving this page.
                </Alert>
              </Stack>
            </CardContent>
          </Card>
        ) : (
          <Alert severity="info">
            Your API key is being provisioned. Refresh in a few seconds, or
            visit the developer portal to mint one manually.
          </Alert>
        )}

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h5" gutterBottom>
              Quickstart
            </Typography>
            <Tabs
              value={tab}
              onChange={(_, newTab: number) => setTab(newTab)}
              sx={{ mb: 2 }}
            >
              {snippets.map((snip) => (
                <Tab key={snip.label} label={snip.label} />
              ))}
            </Tabs>
            {snippets.map((snip, idx) =>
              idx === tab ? (
                <Box key={snip.label}>
                  <Box
                    component="pre"
                    sx={{
                      p: 2,
                      bgcolor: "grey.900",
                      color: "grey.100",
                      borderRadius: 1,
                      overflowX: "auto",
                      fontFamily: "monospace",
                      fontSize: 13,
                      lineHeight: 1.6,
                      m: 0,
                    }}
                    data-testid={`snippet-${snip.label.toLowerCase()}`}
                  >
                    {snip.code}
                  </Box>
                  <Stack direction="row" justifyContent="flex-end" sx={{ mt: 1 }}>
                    <Button
                      size="small"
                      onClick={() => handleCopy(snip.label, snip.code)}
                    >
                      {copied === snip.label ? "Copied" : "Copy snippet"}
                    </Button>
                  </Stack>
                </Box>
              ) : null,
            )}
          </CardContent>
        </Card>

        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              What is next
            </Typography>
            <Stack spacing={1.5}>
              <Typography variant="body2">
                1. Save your API key somewhere safe.
              </Typography>
              <Typography variant="body2">
                2. Read the{" "}
                <a href="https://developers.greenlang.ai/quickstart">
                  Quickstart in the developer portal
                </a>{" "}
                for the full end-to-end walkthrough.
              </Typography>
              <Typography variant="body2">
                3. Pin a catalog edition for reproducible reports -- see the{" "}
                <a href="https://developers.greenlang.ai/concepts/editions">
                  editions concept page
                </a>
                .
              </Typography>
              <Typography variant="body2">
                4. Want to mint additional API keys, manage billing, or invite
                teammates? Visit the{" "}
                <a href="https://developers.greenlang.ai">developer portal</a>.
              </Typography>
            </Stack>
            <Stack direction="row" spacing={2} sx={{ mt: 3 }}>
              <Button
                variant="contained"
                href="https://developers.greenlang.ai"
                data-testid="cta-portal"
              >
                Open developer portal
              </Button>
              <Button
                variant="outlined"
                href="https://developers.greenlang.ai/api/resolve"
              >
                Read the API reference
              </Button>
            </Stack>
          </CardContent>
        </Card>

        {summary.sessionId ? (
          <Typography variant="caption" color="text.secondary">
            Stripe session: <code>{summary.sessionId}</code> &middot; Plan:{" "}
            <code>{summary.planId || "unknown"}</code>
          </Typography>
        ) : null}
      </Stack>
    </Container>
  );
}

export default CheckoutSuccess;
