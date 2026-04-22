import type { Identity } from "@/lib/auth";

export const ALICE: Identity = {
  sub: "alice@greenlang.io",
  display_name: "Alice Author",
  email: "alice@greenlang.io",
  tenant_id: "greenlang-internal",
  roles: ["data_curator", "reviewer"],
  packs: ["corporate", "electricity"],
  session_id: "sess-alice-001",
  issued_at: "2026-04-22T09:00:00Z",
  expires_at: "2026-04-22T17:00:00Z",
};

export const BOB: Identity = {
  sub: "bob@greenlang.io",
  display_name: "Bob Reviewer",
  email: "bob@greenlang.io",
  tenant_id: "greenlang-internal",
  roles: ["reviewer", "methodology_lead"],
  packs: ["corporate", "electricity", "freight"],
  session_id: "sess-bob-001",
  issued_at: "2026-04-22T09:00:00Z",
  expires_at: "2026-04-22T17:00:00Z",
};

export const CAROL_VIEWER: Identity = {
  sub: "carol@greenlang.io",
  display_name: "Carol Viewer",
  email: "carol@greenlang.io",
  tenant_id: "greenlang-internal",
  roles: ["viewer"],
  packs: [],
  session_id: "sess-carol-001",
  issued_at: "2026-04-22T09:00:00Z",
  expires_at: "2026-04-22T17:00:00Z",
};

export const ADMIN: Identity = {
  sub: "admin@greenlang.io",
  display_name: "Admin Dan",
  email: "admin@greenlang.io",
  tenant_id: "greenlang-internal",
  roles: ["admin"],
  packs: ["corporate", "electricity", "freight", "eu_policy", "land_removals", "product_carbon", "finance_proxy"],
  session_id: "sess-admin-001",
  issued_at: "2026-04-22T09:00:00Z",
  expires_at: "2026-04-22T17:00:00Z",
};
