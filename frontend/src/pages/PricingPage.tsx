/**
 * PricingPage -- the public-facing /pricing surface for the FY27 Factors launch.
 *
 * Renders the four canonical SKUs returned by GET /v1/billing/plans.
 *
 *   - Community              -> "Sign up for free" -> POST /v1/billing/checkout/community
 *                              (community is auto-provisioned; the API returns
 *                              the success_url directly without a Stripe redirect)
 *   - Developer Pro          -> "Subscribe" -> POST /v1/billing/checkout/developer_pro
 *                              (real Stripe Checkout Session)
 *   - Consulting / Platform  -> "Contact Sales" -> mailto:sales@greenlang.io
 *   - Enterprise             -> "Contact Sales" -> mailto:sales@greenlang.io
 *
 * The page is intentionally self-contained: it does not require any global
 * state, and its sole network call is `fetch("/v1/billing/plans")`. All
 * styling uses MUI v5 to match the rest of the operator console.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Container from "@mui/material/Container";
import Divider from "@mui/material/Divider";
import Grid from "@mui/material/Grid";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

// ---------------------------------------------------------------------------
// Public types -- mirror the FastAPI response models in api.py.
// ---------------------------------------------------------------------------

interface RateLimitView {
  requests_per_minute: number;
  requests_per_month_included: number;
}

export interface PlanView {
  plan_id: string;
  display_name: string;
  tagline: string;
  price_usd_monthly: string | null;
  price_usd_annual: string | null;
  contact_sales: boolean;
  self_serve: boolean;
  rate_limit: RateLimitView;
  overage_unit_price_usd: string | null;
  license_classes: string[];
  included_premium_packs: string[];
  included_sub_tenants: number;
  oem_redistribution: boolean;
  sla: string | null;
  features: string[];
}

interface PlansResponse {
  plans: PlanView[];
  currency: string;
  stripe_publishable_key: string | null;
}

interface CheckoutResponse {
  session_id: string;
  url: string;
  plan_id: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatPrice(plan: PlanView): { headline: string; cadence: string } {
  if (plan.contact_sales || plan.price_usd_monthly === null) {
    return { headline: "Custom", cadence: "Contact sales" };
  }
  const monthly = Number.parseFloat(plan.price_usd_monthly);
  if (monthly === 0) {
    return { headline: "Free", cadence: "Forever" };
  }
  return {
    headline: `$${monthly.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
    cadence: "/ month",
  };
}

function plansEndpoint(): string {
  // Single source of truth -- match the FastAPI router prefix from api.py.
  return "/v1/billing/plans";
}

function checkoutEndpoint(planId: string): string {
  return `/v1/billing/checkout/${encodeURIComponent(planId)}`;
}

function defaultSuccessUrl(planId: string): string {
  if (typeof window === "undefined") {
    return `https://greenlang.ai/checkout/success?plan=${planId}`;
  }
  const base = `${window.location.protocol}//${window.location.host}`;
  return `${base}/checkout/success?plan=${planId}`;
}

function defaultCancelUrl(): string {
  if (typeof window === "undefined") {
    return "https://greenlang.ai/pricing";
  }
  const base = `${window.location.protocol}//${window.location.host}`;
  return `${base}/pricing`;
}

const SALES_EMAIL = "sales@greenlang.io";

function salesMailto(planId: string): string {
  const subject = encodeURIComponent(
    `Interested in ${planId === "enterprise" ? "Enterprise" : "Consulting / Platform"} plan`,
  );
  const body = encodeURIComponent(
    "Hi GreenLang team,\n\nI'd like to learn more about the " +
      `${planId} plan for the Factors API. ` +
      "Please reach out to discuss.\n\nThanks!",
  );
  return `mailto:${SALES_EMAIL}?subject=${subject}&body=${body}`;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PricingPage(): JSX.Element {
  const [plans, setPlans] = useState<PlanView[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyPlan, setBusyPlan] = useState<string | null>(null);
  const [checkoutError, setCheckoutError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    fetch(plansEndpoint(), { headers: { Accept: "application/json" } })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Failed to load plans: HTTP ${response.status}`);
        }
        const payload = (await response.json()) as PlansResponse;
        if (!cancelled) {
          setPlans(payload.plans);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleCheckout = async (plan: PlanView): Promise<void> => {
    setCheckoutError(null);
    setBusyPlan(plan.plan_id);
    try {
      const response = await fetch(checkoutEndpoint(plan.plan_id), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          success_url: defaultSuccessUrl(plan.plan_id),
          cancel_url: defaultCancelUrl(),
        }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Checkout failed: HTTP ${response.status}`);
      }
      const payload = (await response.json()) as CheckoutResponse;
      if (typeof window !== "undefined") {
        window.location.href = payload.url;
      }
    } catch (err: unknown) {
      setCheckoutError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyPlan(null);
    }
  };

  const sortedPlans = useMemo(() => plans ?? [], [plans]);

  return (
    <Container maxWidth="lg" sx={{ py: 6 }}>
      <Stack spacing={1} sx={{ mb: 5 }} alignItems="center" textAlign="center">
        <Typography variant="h3" component="h1" gutterBottom>
          Pick the plan that fits your audit footprint
        </Typography>
        <Typography variant="body1" color="text.secondary" maxWidth={680}>
          Every plan ships with the same audited factor catalog, edition pinning,
          and signed-receipt verification. Higher tiers unlock more licensed packs,
          higher rate limits, and SLA-backed uptime.
        </Typography>
      </Stack>

      {error ? (
        <Alert severity="error" sx={{ mb: 3 }} role="alert">
          {error}
        </Alert>
      ) : null}

      {checkoutError ? (
        <Alert severity="error" sx={{ mb: 3 }} role="alert">
          {checkoutError}
        </Alert>
      ) : null}

      {plans === null && !error ? (
        <Stack alignItems="center" sx={{ py: 8 }}>
          <CircularProgress />
        </Stack>
      ) : null}

      <Grid container spacing={3} alignItems="stretch">
        {sortedPlans.map((plan) => {
          const price = formatPrice(plan);
          const featured = plan.plan_id === "developer_pro";
          return (
            <Grid item xs={12} md={6} lg={3} key={plan.plan_id}>
              <Card
                variant="outlined"
                sx={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  borderColor: featured ? "primary.main" : undefined,
                  borderWidth: featured ? 2 : 1,
                }}
                data-testid={`plan-card-${plan.plan_id}`}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Stack spacing={1.5}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="h5" component="h2">
                        {plan.display_name}
                      </Typography>
                      {featured ? (
                        <Chip label="Most popular" color="primary" size="small" />
                      ) : null}
                    </Stack>
                    <Typography variant="body2" color="text.secondary" minHeight={48}>
                      {plan.tagline}
                    </Typography>
                    <Box sx={{ py: 1 }}>
                      <Typography variant="h3" component="div">
                        {price.headline}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {price.cadence}
                      </Typography>
                    </Box>
                    <Divider />
                    <List dense disablePadding>
                      {plan.features.map((feature) => (
                        <ListItem key={feature} disableGutters sx={{ py: 0.5 }}>
                          <ListItemText
                            primary={feature}
                            primaryTypographyProps={{ variant: "body2" }}
                          />
                        </ListItem>
                      ))}
                    </List>
                    {plan.sla ? (
                      <Chip
                        size="small"
                        variant="outlined"
                        label={`${plan.sla}% uptime SLA`}
                        sx={{ alignSelf: "flex-start" }}
                      />
                    ) : null}
                    {plan.oem_redistribution ? (
                      <Chip
                        size="small"
                        color="success"
                        label="OEM white-label"
                        sx={{ alignSelf: "flex-start" }}
                      />
                    ) : null}
                  </Stack>
                </CardContent>
                <CardActions sx={{ p: 2, pt: 0 }}>
                  {plan.contact_sales ? (
                    <Button
                      fullWidth
                      variant="outlined"
                      href={salesMailto(plan.plan_id)}
                      data-testid={`plan-cta-${plan.plan_id}`}
                    >
                      Contact Sales
                    </Button>
                  ) : (
                    <Button
                      fullWidth
                      variant={featured ? "contained" : "outlined"}
                      onClick={() => {
                        void handleCheckout(plan);
                      }}
                      disabled={busyPlan !== null}
                      data-testid={`plan-cta-${plan.plan_id}`}
                    >
                      {busyPlan === plan.plan_id
                        ? "Redirecting..."
                        : plan.plan_id === "community"
                        ? "Sign up for free"
                        : "Subscribe"}
                    </Button>
                  )}
                </CardActions>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      <Stack alignItems="center" spacing={1} sx={{ mt: 6 }} textAlign="center">
        <Typography variant="body2" color="text.secondary">
          All prices in USD. Annual contracts available with custom pricing.
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Need help choosing?{" "}
          <a href={`mailto:${SALES_EMAIL}`}>Talk to a human</a> or read the{" "}
          <a href="https://developers.greenlang.ai">developer docs</a>.
        </Typography>
      </Stack>
    </Container>
  );
}

export default PricingPage;
