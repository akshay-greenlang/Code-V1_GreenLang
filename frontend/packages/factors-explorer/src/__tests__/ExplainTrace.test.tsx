import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createMemoryHistory,
  createRouter,
  createRootRoute,
  createRoute,
  RouterProvider,
} from "@tanstack/react-router";
import { ExplainTrace } from "@/components/ExplainTrace";
import { RESOLVED_FACTOR_FIXTURE, toExplainEnvelope } from "./fixtures/explain";

function renderWithRouter(ui: React.ReactNode) {
  const rootRoute = createRootRoute({ component: () => <>{ui}</> });
  const indexRoute = createRoute({
    getParentRoute: () => rootRoute,
    path: "/",
    component: () => <>{ui}</>,
  });
  const router = createRouter({
    routeTree: rootRoute.addChildren([indexRoute]),
    history: createMemoryHistory({ initialEntries: ["/"] }),
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

describe("ExplainTrace", () => {
  it("renders all 7 cascade steps and highlights the chosen rank", () => {
    const explain = toExplainEnvelope(RESOLVED_FACTOR_FIXTURE);
    renderWithRouter(<ExplainTrace explain={explain} />);

    const cascade = screen.getByTestId("cascade-steps");
    const items = cascade.querySelectorAll("li");
    expect(items).toHaveLength(7);

    const chosen = cascade.querySelector('[data-status="chosen"]');
    expect(chosen).not.toBeNull();
    expect(chosen).toHaveAttribute("data-rank", "5");

    // Ranks below chosen are skipped; ranks above are considered
    expect(
      cascade.querySelectorAll('[data-status="skipped"]').length
    ).toBeGreaterThan(0);
    expect(
      cascade.querySelectorAll('[data-status="considered"]').length
    ).toBeGreaterThan(0);

    // "CHOSEN" label only appears once
    expect(screen.getAllByText(/CHOSEN/i)).toHaveLength(1);
  });

  it("renders alternates with why_not_chosen reasons", () => {
    const explain = toExplainEnvelope(RESOLVED_FACTOR_FIXTURE);
    renderWithRouter(<ExplainTrace explain={explain} />);

    const alternates = screen.getByTestId("alternates-list");
    expect(alternates).toBeInTheDocument();
    expect(alternates).toHaveTextContent("IEA-GLOBAL-NG-2023-001");
    expect(alternates).toHaveTextContent("global, not GB-specific");
    expect(alternates).toHaveTextContent("ECOINVENT-NG-GB-3.10");
    expect(alternates).toHaveTextContent("tie 2.10");
  });

  it("shows the why_chosen sentence from derivation", () => {
    const explain = toExplainEnvelope(RESOLVED_FACTOR_FIXTURE);
    renderWithRouter(<ExplainTrace explain={explain} />);
    expect(screen.getByText(/Why chosen:/)).toBeInTheDocument();
    expect(
      screen.getByText(/DEFRA 2024 natural_gas GB matched jurisdiction/)
    ).toBeInTheDocument();
  });

  it("renders deprecation banner when status is deprecated", () => {
    const explain = toExplainEnvelope({
      ...RESOLVED_FACTOR_FIXTURE,
      deprecation_status: "deprecated",
      deprecation_replacement: "DEFRA-NG-GB-2025-001",
    });
    renderWithRouter(<ExplainTrace explain={explain} />);

    const banner = screen.getByTestId("deprecation-banner");
    expect(banner).toBeInTheDocument();
    expect(banner).toHaveTextContent(/deprecated/);
    expect(banner).toHaveTextContent("DEFRA-NG-GB-2025-001");
  });

  it("does not render deprecation banner when status is active", () => {
    const explain = toExplainEnvelope(RESOLVED_FACTOR_FIXTURE);
    renderWithRouter(<ExplainTrace explain={explain} />);
    expect(screen.queryByTestId("deprecation-banner")).toBeNull();
  });

  it("renders unit conversion trace when present", () => {
    const explain = toExplainEnvelope({
      ...RESOLVED_FACTOR_FIXTURE,
      target_unit: "therm",
      unit_conversion_factor: 29.3071,
      converted_co2e_per_unit: 59.479,
      unit_conversion_path: ["m3", "MJ", "therm"],
      unit_conversion_note: "via IEA energy-equivalent table",
    });
    renderWithRouter(<ExplainTrace explain={explain} />);
    expect(screen.getByText(/Unit conversion/)).toBeInTheDocument();
    expect(screen.getByText(/m3 → MJ → therm/)).toBeInTheDocument();
    expect(
      screen.getByText(/via IEA energy-equivalent table/)
    ).toBeInTheDocument();
  });
});
