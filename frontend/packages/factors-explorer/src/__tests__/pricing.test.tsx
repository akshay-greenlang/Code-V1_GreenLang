import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, within, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { PricingPage } from "@/routes/pricing";
import {
  PREMIUM_PACKS,
  PREMIUM_PACK_SKUS,
} from "@/components/pricing/PremiumPackGrid";

/**
 * Render the pricing route inside an in-memory TanStack Router so that
 * `useNavigate`, `Link`, and `createFileRoute` all work without
 * touching window.history.
 */
function renderPricingRoute() {
  const rootRoute = createRootRoute();
  // Mount the PricingPage component directly under an in-memory
  // router so tests do not depend on the generated route tree.
  const route = createRoute({
    getParentRoute: () => rootRoute,
    path: "/pricing",
    component: PricingPage,
  });
  const router = createRouter({
    routeTree: rootRoute.addChildren([route]),
    history: createMemoryHistory({ initialEntries: ["/pricing"] }),
  });
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

describe("PricingPage", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders all 4 tier cards", async () => {
    renderPricingRoute();
    // Wait for the route to mount.
    await screen.findByTestId("tier-grid");
    expect(screen.getByTestId("tier-card-community")).toBeInTheDocument();
    expect(screen.getByTestId("tier-card-pro")).toBeInTheDocument();
    expect(screen.getByTestId("tier-card-platform")).toBeInTheDocument();
    expect(screen.getByTestId("tier-card-enterprise")).toBeInTheDocument();
    // Pro is highlighted and has the Recommended badge.
    expect(
      within(screen.getByTestId("tier-card-pro")).getByText(/Recommended/i)
    ).toBeInTheDocument();
  });

  it("renders all 7 premium packs", async () => {
    renderPricingRoute();
    const grid = await screen.findByTestId("premium-pack-grid");
    expect(PREMIUM_PACKS).toHaveLength(7);
    for (const pack of PREMIUM_PACKS) {
      expect(within(grid).getByTestId(`premium-pack-${pack.sku}`)).toBeInTheDocument();
    }
    // PREMIUM_PACK_SKUS export matches the rendered set.
    expect(PREMIUM_PACK_SKUS).toHaveLength(7);
  });

  it("shows the three-label legend with all three labels", async () => {
    renderPricingRoute();
    const legend = await screen.findByTestId("three-label-legend");
    expect(within(legend).getByTestId("legend-certified")).toBeInTheDocument();
    expect(within(legend).getByTestId("legend-preview")).toBeInTheDocument();
    expect(
      within(legend).getByTestId("legend-connector-only")
    ).toBeInTheDocument();
  });

  it("renders the FAQ section with at least 4 rows", async () => {
    renderPricingRoute();
    const faq = await screen.findByTestId("faq-section");
    const rows = within(faq).getAllByTestId("faq-row");
    expect(rows.length).toBeGreaterThanOrEqual(4);
    // Spot-check a key question.
    expect(faq).toHaveTextContent(/What counts as one API call/i);
  });

  it("Community CTA navigates to the sign-up route", async () => {
    const assignSpy = vi.fn();
    Object.defineProperty(window, "location", {
      value: { ...window.location, assign: assignSpy, origin: "http://t" },
      writable: true,
    });
    renderPricingRoute();
    const community = await screen.findByTestId("tier-card-community");
    const cta = within(community).getByTestId("tier-cta-community");
    await userEvent.click(cta);
    expect(assignSpy).toHaveBeenCalledWith("/sign-up?tier=community");
  });

  it("Pro CTA calls /v1/billing/checkout and redirects to the Stripe URL", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "cs_test_123",
          url: "https://checkout.stripe.com/c/pay/cs_test_123",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);
    const assignSpy = vi.fn();
    Object.defineProperty(window, "location", {
      value: {
        ...window.location,
        assign: assignSpy,
        origin: "https://factors.greenlang.io",
      },
      writable: true,
    });

    renderPricingRoute();
    const proCard = await screen.findByTestId("tier-card-pro");
    const cta = within(proCard).getByTestId("tier-cta-pro");
    await act(async () => {
      await userEvent.click(cta);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/v1/billing/checkout");
    expect(init).toMatchObject({ method: "POST" });
    const sentBody = JSON.parse((init as RequestInit).body as string);
    expect(sentBody).toMatchObject({
      sku_name: "pro_monthly",
      success_url: expect.stringContaining("/pricing?status=success"),
      cancel_url: expect.stringContaining("/pricing?status=cancelled"),
      premium_packs: [],
    });
    expect(assignSpy).toHaveBeenCalledWith(
      "https://checkout.stripe.com/c/pay/cs_test_123"
    );
  });

  it("includes selected premium packs in the Pro checkout payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          session_id: "cs_test_456",
          url: "https://checkout.stripe.com/c/pay/cs_test_456",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);
    Object.defineProperty(window, "location", {
      value: {
        ...window.location,
        assign: vi.fn(),
        origin: "https://factors.greenlang.io",
      },
      writable: true,
    });

    renderPricingRoute();
    const grid = await screen.findByTestId("premium-pack-grid");
    // Toggle the Electricity pack on.
    const electricityCta = within(grid).getByTestId(
      "premium-pack-cta-electricity_premium"
    );
    await userEvent.click(electricityCta);
    expect(electricityCta).toHaveAttribute("aria-pressed", "true");

    const proCard = screen.getByTestId("tier-card-pro");
    const cta = within(proCard).getByTestId("tier-cta-pro");
    await act(async () => {
      await userEvent.click(cta);
    });
    const sentBody = JSON.parse(
      (fetchMock.mock.calls[0][1] as RequestInit).body as string
    );
    expect(sentBody.premium_packs).toEqual(["electricity_premium"]);
  });

  it("Enterprise CTA opens the Talk-to-sales modal with email + use case fields", async () => {
    renderPricingRoute();
    const enterprise = await screen.findByTestId("tier-card-enterprise");
    const cta = within(enterprise).getByTestId("tier-cta-enterprise");
    await userEvent.click(cta);
    const modal = await screen.findByTestId("contact-sales-modal");
    expect(within(modal).getByLabelText(/Work email/i)).toBeInTheDocument();
    expect(within(modal).getByLabelText(/Use case/i)).toBeInTheDocument();
    expect(
      within(modal).getByLabelText(/Company size/i)
    ).toBeInTheDocument();
  });
});
