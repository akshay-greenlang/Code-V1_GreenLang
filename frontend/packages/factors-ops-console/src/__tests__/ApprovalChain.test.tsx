import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ApprovalChain } from "@/components/ApprovalChain";
import { ALICE, BOB, CAROL_VIEWER } from "./fixtures/identities";
import { REVIEW_ALICE_AUTHORED, REVIEW_BOB_AUTHORED } from "./fixtures/reviews";
import { canApprove } from "@/lib/auth";

/**
 * SoD tests for <ApprovalChain/>.
 *
 * The critical invariant (spec §3.3): `author.sub !== current_user.sub`.
 * If violated, the Approve button must NOT render. We also verify that the
 * SoD banner is shown and that a valid two-stage completion path works.
 */
describe("ApprovalChain — Segregation of Duties", () => {
  it("renders the SoD banner when the current user is the author", () => {
    const onApprove = vi.fn(async () => {});
    const onReject = vi.fn(async () => {});
    render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={ALICE /* author is alice */}
        onApprove={onApprove}
        onReject={onReject}
      />
    );
    expect(screen.getByTestId("sod-banner")).toBeInTheDocument();
    expect(screen.getByText(/You authored this submission/i)).toBeInTheDocument();
  });

  it("does NOT render an Approve button when author === current user (SoD)", () => {
    render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={ALICE}
        onApprove={vi.fn()}
        onReject={vi.fn()}
      />
    );
    expect(screen.queryByTestId("approve-btn")).not.toBeInTheDocument();
    expect(screen.getByTestId("approve-disabled")).toHaveTextContent(
      /Approve blocked by SoD/i
    );
  });

  it("renders the Approve button when author !== current user AND role is sufficient", () => {
    // Alice authored, Bob is a different reviewer
    render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={BOB}
        onApprove={vi.fn()}
        onReject={vi.fn()}
      />
    );
    expect(screen.getByTestId("approve-btn")).toBeInTheDocument();
  });

  it("canApprove() helper: author cannot approve own submission", () => {
    expect(canApprove(ALICE, REVIEW_ALICE_AUTHORED.author.sub)).toBe(false);
    expect(canApprove(BOB, REVIEW_ALICE_AUTHORED.author.sub)).toBe(true);
    expect(canApprove(BOB, REVIEW_BOB_AUTHORED.author.sub)).toBe(false);
  });

  it("canApprove() helper: viewer role cannot approve even if not author", () => {
    expect(canApprove(CAROL_VIEWER, REVIEW_ALICE_AUTHORED.author.sub)).toBe(false);
  });

  it("blocks submission until reason is >= 10 chars", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn(async () => {});
    render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={BOB}
        onApprove={onApprove}
        onReject={vi.fn()}
      />
    );
    const approveBtn = screen.getByTestId("approve-btn");
    // No reason entered -> disabled
    expect(approveBtn).toBeDisabled();

    await user.type(screen.getByLabelText(/Reason/i), "too short");
    // "too short" = 9 chars -> still disabled
    expect(approveBtn).toBeDisabled();
    expect(onApprove).not.toHaveBeenCalled();
  });

  it("submits onApprove with the reason when SoD is satisfied and reason is long enough", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn(async () => {});
    render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={BOB}
        onApprove={onApprove}
        onReject={vi.fn()}
      />
    );
    await user.type(
      screen.getByLabelText(/Reason/i),
      "Matches QA criteria v4; gas breakdown audited."
    );
    const approveBtn = screen.getByTestId("approve-btn");
    await waitFor(() => expect(approveBtn).not.toBeDisabled());
    await user.click(approveBtn);
    await waitFor(() =>
      expect(onApprove).toHaveBeenCalledWith(
        "Matches QA criteria v4; gas breakdown audited."
      )
    );
  });
});

describe("ApprovalChain — two-stage completion", () => {
  it("renders both author and approver nodes in the chain visual", () => {
    render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={BOB}
        onApprove={vi.fn()}
        onReject={vi.fn()}
      />
    );
    const visual = screen.getByTestId("approval-chain-visual");
    expect(visual).toHaveTextContent(/Alice Author/);
    expect(visual).toHaveTextContent(/author/);
    expect(visual).toHaveTextContent(/approver 1/);
  });

  it("after approval is submitted, the calling code can mark the step done (simulated)", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn(async () => {});
    const { rerender } = render(
      <ApprovalChain
        review={REVIEW_ALICE_AUTHORED}
        identity={BOB}
        onApprove={onApprove}
        onReject={vi.fn()}
      />
    );
    await user.type(screen.getByLabelText(/Reason/i), "Two-stage completion test.");
    await user.click(screen.getByTestId("approve-btn"));
    await waitFor(() => expect(onApprove).toHaveBeenCalled());

    // Simulate the parent updating the review with an approved step.
    const approved = {
      ...REVIEW_ALICE_AUTHORED,
      steps: [
        {
          approver: { sub: "bob@greenlang.io", display_name: "Bob Reviewer" },
          status: "approved" as const,
          at: "2026-04-22T10:00:00Z",
          comment: null,
        },
      ],
    };
    rerender(
      <ApprovalChain
        review={approved}
        identity={BOB}
        onApprove={vi.fn()}
        onReject={vi.fn()}
      />
    );
    const visual = screen.getByTestId("approval-chain-visual");
    expect(visual).toHaveTextContent(/Bob Reviewer/);
  });
});
