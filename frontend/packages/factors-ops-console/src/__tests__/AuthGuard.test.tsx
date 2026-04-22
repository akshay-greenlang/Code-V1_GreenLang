import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthGuard } from "@/components/AuthGuard";
import { ALICE, CAROL_VIEWER } from "./fixtures/identities";

/**
 * Tests for <AuthGuard/> — role-based redirects + 403 rendering.
 *
 * We stub the /api/v1/auth/me endpoint via a global fetch mock, then assert
 * that the guard renders children / 403 / redirects correctly.
 */
function wrap(children: React.ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

const originalFetch = globalThis.fetch;
const originalLocation = window.location;

beforeEach(() => {
  // Mock window.location.assign so redirectToLogin doesn't actually navigate.
  Object.defineProperty(window, "location", {
    writable: true,
    value: {
      ...originalLocation,
      pathname: "/approvals",
      origin: "http://localhost:5175",
      assign: vi.fn(),
    },
  });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  Object.defineProperty(window, "location", {
    writable: true,
    value: originalLocation,
  });
  vi.restoreAllMocks();
});

function mockFetch(body: unknown, status = 200) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: status < 400,
    status,
    json: async () => body,
  } as Response);
}

describe("<AuthGuard/>", () => {
  it("renders children when the identity satisfies requiredRoles", async () => {
    mockFetch(ALICE);
    render(
      wrap(
        <AuthGuard requiredRoles={["reviewer"]}>
          <div>protected content</div>
        </AuthGuard>
      )
    );
    await waitFor(() =>
      expect(screen.getByText("protected content")).toBeInTheDocument()
    );
  });

  it("renders a 403 page when the identity lacks the required role", async () => {
    mockFetch(CAROL_VIEWER);
    render(
      wrap(
        <AuthGuard requiredRoles={["admin"]}>
          <div>admin-only content</div>
        </AuthGuard>
      )
    );
    await waitFor(() =>
      expect(screen.getByText(/Insufficient role/i)).toBeInTheDocument()
    );
    expect(screen.queryByText("admin-only content")).not.toBeInTheDocument();
  });

  it("redirects to /login when unauthenticated (fetch returns 401)", async () => {
    mockFetch({}, 401);
    render(
      wrap(
        <AuthGuard requiredRoles={["reviewer"]}>
          <div>private</div>
        </AuthGuard>
      )
    );
    await waitFor(() => {
      expect(window.location.assign).toHaveBeenCalled();
    });
    const call = (window.location.assign as ReturnType<typeof vi.fn>).mock
      .calls[0]?.[0] as string;
    expect(String(call)).toContain("/api/v1/auth/login");
    expect(String(call)).toContain("return_to=%2Fapprovals");
  });

  it("enforces requiredAction via the role-action matrix", async () => {
    mockFetch(CAROL_VIEWER); // viewer can read but cannot run ingestion
    render(
      wrap(
        <AuthGuard requiredAction="ingest.run">
          <div>ingestion console</div>
        </AuthGuard>
      )
    );
    await waitFor(() =>
      expect(screen.getByText(/Insufficient role/i)).toBeInTheDocument()
    );
  });

  it("passes identity to children render-prop", async () => {
    mockFetch(ALICE);
    render(
      wrap(
        <AuthGuard>
          {(id) => <div data-testid="hello">hi {id.display_name}</div>}
        </AuthGuard>
      )
    );
    await waitFor(() =>
      expect(screen.getByTestId("hello")).toHaveTextContent("hi Alice Author")
    );
  });
});
