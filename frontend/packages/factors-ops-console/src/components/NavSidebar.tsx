import { Link } from "@tanstack/react-router";
import {
  LayoutDashboard,
  DownloadCloud,
  GitCompare,
  ClipboardCheck,
  ShieldCheck,
  Activity,
  Eye,
  Users,
  Layers,
  Map,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { canDo, type Identity, type OpsAction } from "@/lib/auth";

/**
 * 10-item vertical nav. Items whose action the identity cannot perform are
 * hidden (role-filtered visibility — spec §3.3).
 */
interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  requires?: OpsAction;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/ingestion", label: "Ingestion", icon: DownloadCloud, requires: "ingest.run" },
  { to: "/mapping", label: "Mapping", icon: Map, requires: "mapping.edit" },
  { to: "/qa", label: "QA", icon: ShieldCheck, requires: "qa.remediate" },
  { to: "/diff", label: "Diff", icon: GitCompare },
  { to: "/approvals", label: "Approvals", icon: ClipboardCheck, requires: "review.approve" },
  { to: "/overrides", label: "Overrides", icon: Users, requires: "override.edit" },
  { to: "/impact", label: "Impact Sim", icon: Activity, requires: "impact.run" },
  { to: "/watch", label: "Source Watch", icon: Eye, requires: "watch.classify" },
  { to: "/entitlements", label: "Entitlements", icon: Users, requires: "entitlement.edit" },
  { to: "/editions", label: "Editions", icon: Layers, requires: "edition.promote" },
];

export function NavSidebar({ identity }: { identity: Identity }) {
  const visible = NAV_ITEMS.filter((it) => !it.requires || canDo(identity, it.requires));
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-muted/20 py-4">
      <div className="px-4 pb-3">
        <Link to="/" className="flex items-center gap-2 font-semibold">
          <span
            aria-hidden="true"
            className="inline-block h-5 w-5 rounded bg-factor-certified-500"
          />
          <span className="text-sm">Factors Ops</span>
        </Link>
      </div>
      <nav aria-label="Primary" className="flex flex-1 flex-col gap-0.5 px-2">
        {visible.map((it) => {
          const Icon = it.icon;
          return (
            <Link
              key={it.to}
              to={it.to}
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground",
                "data-[status=active]:bg-accent data-[status=active]:text-foreground"
              )}
              activeProps={{ "data-status": "active" } as never}
              activeOptions={{ exact: it.to === "/" }}
            >
              <Icon className="h-4 w-4" aria-hidden="true" />
              <span>{it.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
