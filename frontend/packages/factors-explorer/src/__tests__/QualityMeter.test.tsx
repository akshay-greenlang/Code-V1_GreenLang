import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { QualityMeter } from "@/components/QualityMeter";
import type { CompositeFqs } from "@/types/factors";

function makeFqs(overrides: Partial<CompositeFqs>): CompositeFqs {
  return {
    overall: 75,
    rating: "B",
    temporal_representativeness: 80,
    geographic_representativeness: 70,
    technology_representativeness: 75,
    verification: 85,
    completeness: 65,
    uncertainty_95ci: 0.05,
    ...overrides,
  };
}

describe("QualityMeter", () => {
  it("renders composite FQS score and rating", () => {
    render(<QualityMeter fqs={makeFqs({ overall: 92, rating: "A" })} />);
    const pill = screen.getByText(/FQS A/i);
    expect(pill).toBeInTheDocument();
    expect(pill).toHaveAttribute("data-band", "excellent");
    // Arc gauge shows rounded overall value
    expect(screen.getByText("92")).toBeInTheDocument();
  });

  it.each([
    { overall: 90, rating: "A" as const, band: "excellent" },
    { overall: 75, rating: "B" as const, band: "good" },
    { overall: 55, rating: "C" as const, band: "fair" },
    { overall: 30, rating: "D" as const, band: "poor" },
  ])("maps score $overall to $band band", ({ overall, rating, band }) => {
    render(<QualityMeter fqs={makeFqs({ overall, rating })} />);
    const pill = screen.getByText(new RegExp(`FQS ${rating}`));
    expect(pill).toHaveAttribute("data-band", band);
  });

  it("renders all 5 component bars with values", () => {
    render(
      <QualityMeter
        fqs={makeFqs({
          temporal_representativeness: 80,
          geographic_representativeness: 70,
          technology_representativeness: 75,
          verification: 85,
          completeness: 65,
        })}
      />
    );
    expect(screen.getByText("Temporal")).toBeInTheDocument();
    expect(screen.getByText("Geographic")).toBeInTheDocument();
    expect(screen.getByText("Technology")).toBeInTheDocument();
    expect(screen.getByText("Verification")).toBeInTheDocument();
    expect(screen.getByText("Completeness")).toBeInTheDocument();
    // all five progress bars
    const bars = screen.getAllByRole("progressbar");
    expect(bars).toHaveLength(5);
    expect(bars[0]).toHaveAttribute("aria-valuenow", "80");
  });

  it("renders compact variant as inline pill", () => {
    render(
      <QualityMeter fqs={makeFqs({ overall: 84, rating: "B" })} compact />
    );
    expect(screen.getByText(/FQS B/)).toBeInTheDocument();
    expect(screen.getByText(/84\/100/)).toBeInTheDocument();
    // compact mode should NOT render the detail component bars
    expect(screen.queryByText("Temporal")).not.toBeInTheDocument();
  });

  it("shows 95%CI uncertainty when provided", () => {
    render(<QualityMeter fqs={makeFqs({ uncertainty_95ci: 0.025 })} />);
    expect(screen.getByText(/±2.5% 95% CI/)).toBeInTheDocument();
  });
});
