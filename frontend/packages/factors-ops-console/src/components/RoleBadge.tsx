import { ShieldCheck, LogOut } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { logout, type Identity, type OpsRole } from "@/lib/auth";

/**
 * Top-bar badge showing the current user + their roles + a logout affordance.
 *
 * The role's color comes from the Tailwind role-* palette. If the user holds
 * multiple roles we show the "highest-privilege" one first (admin > methodology
 * > reviewer > everything else).
 */
const ROLE_PRIORITY: OpsRole[] = [
  "admin",
  "methodology_lead",
  "release_manager",
  "reviewer",
  "data_curator",
  "legal",
  "support",
  "viewer",
];

function highestRole(roles: OpsRole[]): OpsRole {
  for (const r of ROLE_PRIORITY) {
    if (roles.includes(r)) return r;
  }
  return "viewer";
}

const ROLE_BADGE_CLASS: Record<OpsRole, string> = {
  admin: "bg-role-admin/10 text-role-admin border-role-admin/30",
  methodology_lead: "bg-role-methodology/10 text-role-methodology border-role-methodology/30",
  release_manager: "bg-role-methodology/10 text-role-methodology border-role-methodology/30",
  reviewer: "bg-role-reviewer/10 text-role-reviewer border-role-reviewer/30",
  data_curator: "bg-factor-preview-100 text-factor-preview-700 border-factor-preview-500",
  legal: "bg-factor-deprecated-100 text-factor-deprecated-700 border-factor-deprecated-500",
  support: "bg-accent text-accent-foreground border-border",
  viewer: "bg-role-viewer/10 text-role-viewer border-role-viewer/30",
};

export function RoleBadge({ identity }: { identity: Identity }) {
  const top = highestRole(identity.roles);
  return (
    <div className="flex items-center gap-2" aria-label="Signed-in user">
      <div className="flex flex-col items-end text-xs leading-tight">
        <span className="font-medium">{identity.display_name}</span>
        <span className="text-muted-foreground">{identity.tenant_id}</span>
      </div>
      <Badge
        variant="outline"
        className={cn("border gap-1", ROLE_BADGE_CLASS[top])}
        title={`Roles: ${identity.roles.join(", ")}`}
      >
        <ShieldCheck className="h-3 w-3" aria-hidden="true" />
        <span>{top}</span>
      </Badge>
      <Button
        variant="ghost"
        size="icon"
        aria-label="Sign out"
        onClick={() => void logout()}
      >
        <LogOut className="h-4 w-4" />
      </Button>
    </div>
  );
}
