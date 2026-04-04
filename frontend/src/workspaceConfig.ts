import type { AppKey } from "./types";

export interface WorkspaceChecklistItem {
  id: string;
  label: string;
}

export interface WorkspaceConfig {
  regulatoryNotes: string;
  primaryFileHint: string;
  secondaryFileHint?: string;
  checklist: WorkspaceChecklistItem[];
}

export const workspaceByApp: Record<AppKey, WorkspaceConfig> = {
  cbam: {
    regulatoryNotes:
      "CBAM quarterly declaration: validate YAML config and import ledger, then review embedded emissions XML and policy gates before registry export.",
    primaryFileHint: "Config YAML (declarant, reporting period, settings).",
    secondaryFileHint: "Import ledger CSV/XLSX (required for full run; demo uses samples).",
    checklist: [
      { id: "cbam-1", label: "Verify CN codes and default factor exposure in drilldown." },
      { id: "cbam-2", label: "Confirm XML schema PASS before treating export as submission-ready." },
      { id: "cbam-3", label: "Capture evidence ZIP for audit trail." }
    ]
  },
  csrd: {
    regulatoryNotes:
      "CSRD / ESRS evidence bundle: align datapoints to ESRS topical standards and keep narrative traceable to structured inputs.",
    primaryFileHint: "Structured ESG / ESRS-aligned CSV or JSON input.",
    checklist: [
      { id: "csrd-1", label: "Map material topics to datapoints in the evidence bundle." },
      { id: "csrd-2", label: "Stage XBRL or structured export for assurance review." },
      { id: "csrd-3", label: "Record data gaps as explicit assumptions, not silent defaults." }
    ]
  },
  vcci: {
    regulatoryNotes:
      "Scope 3 inventory: prioritize category rules, supplier data quality, and emission-factor provenance for ISO 14064-aligned reporting.",
    primaryFileHint: "Category batch CSV (e.g. purchased goods, transport).",
    checklist: [
      { id: "vcci-1", label: "Reconcile activity data with supplier mapping coverage." },
      { id: "vcci-2", label: "Flag high-uncertainty categories for secondary data justification." },
      { id: "vcci-3", label: "Attach policy evaluation output to inventory sign-off." }
    ]
  },
  eudr: {
    regulatoryNotes:
      "EUDR due diligence: chain-of-custody, geo-risk, and deforestation screening must be evidenced per plot or shipment batch.",
    primaryFileHint: "Due diligence JSON (plots, suppliers, risk flags).",
    checklist: [
      { id: "eudr-1", label: "Verify geo-coordinates and country-of-production consistency." },
      { id: "eudr-2", label: "Document mitigation when risk classifiers fire." },
      { id: "eudr-3", label: "Retain supplier attestations with the DDS artifact set." }
    ]
  },
  ghg: {
    regulatoryNotes:
      "Corporate GHG inventory: scope split, factor libraries, and organizational boundary must match your reporting protocol.",
    primaryFileHint: "Inventory JSON (scopes, activities, factors).",
    checklist: [
      { id: "ghg-1", label: "Confirm organizational boundary matches fiscal control definition." },
      { id: "ghg-2", label: "Resolve market-based vs location-based electricity explicitly." },
      { id: "ghg-3", label: "Cross-check intensity metrics against revenue or production drivers." }
    ]
  },
  iso14064: {
    regulatoryNotes:
      "ISO 14064 verification readiness: materiality, sampling, and evidence folders should mirror verifier expectations.",
    primaryFileHint: "Verification packet JSON (assertions, controls, evidence index).",
    checklist: [
      { id: "iso-1", label: "Tie assertions to measurable GHG quantification equations." },
      { id: "iso-2", label: "List control activities with owner and frequency." },
      { id: "iso-3", label: "Prepare non-conformance workflow if policy blocks export." }
    ]
  },
  sb253: {
    regulatoryNotes:
      "California SB 253 / climate disclosure: scope coverage, assurance staging, and public filing readiness must align to the regulatory calendar.",
    primaryFileHint: "Disclosure draft JSON (scopes, activities, factors).",
    checklist: [
      { id: "sb253-1", label: "Confirm Scope 1/2 boundary matches operator control definition." },
      { id: "sb253-2", label: "Stage assurance artifacts for third-party review checkpoints." },
      { id: "sb253-3", label: "Track filing deadlines vs. internal materiality decisions." }
    ]
  },
  taxonomy: {
    regulatoryNotes:
      "EU Taxonomy alignment: technical screening criteria, DNSH, and minimum safeguards should be evidenced per activity class.",
    primaryFileHint: "Alignment JSON (activities, NACE mapping, KPIs).",
    checklist: [
      { id: "tax-1", label: "Map revenue and CapEx to eligible and aligned activities." },
      { id: "tax-2", label: "Document DNSH considerations where environmental trade-offs exist." },
      { id: "tax-3", label: "Retain minimum safeguards attestations with disclosure drafts." }
    ]
  }
};
