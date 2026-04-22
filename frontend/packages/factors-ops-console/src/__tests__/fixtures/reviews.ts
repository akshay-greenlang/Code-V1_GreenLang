import type { ReviewItem } from "@/types/ops";

/** Review authored by Alice, awaiting Bob's approval. */
export const REVIEW_ALICE_AUTHORED: ReviewItem = {
  review_id: "r-412",
  kind: "ingestion",
  author: { sub: "alice@greenlang.io", display_name: "Alice Author" },
  steps: [
    {
      approver: null,
      status: "pending",
      at: null,
      comment: null,
    },
  ],
  age_hours: 2,
  context: {
    job_id: "j-9823",
    source: "DEFRA 2025",
    row_count: 4402,
    parse_success: 0.992,
    dqs_avg: 87,
  },
};

/** Review authored by Bob — any test where Bob tries to approve must be blocked. */
export const REVIEW_BOB_AUTHORED: ReviewItem = {
  review_id: "r-413",
  kind: "mapping",
  author: { sub: "bob@greenlang.io", display_name: "Bob Reviewer" },
  steps: [{ approver: null, status: "pending" }],
  age_hours: 6,
  context: {},
};
