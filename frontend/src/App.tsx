import { Navigate, Route, Routes } from "react-router-dom";
import { RoleGuard } from "./components/RoleGuard";
import { ShellLayout } from "./components/ShellLayout";
import { AdminGate } from "./lib/auth/AdminGate";
import { AdminPage } from "./pages/AdminPage";
import { FactorsApprovalQueue } from "./pages/FactorsApprovalQueue";
import { FactorsCatalogStatus } from "./pages/FactorsCatalogStatus";
import { FactorsDiffViewer } from "./pages/FactorsDiffViewer";
import { FactorsExplorer } from "./pages/FactorsExplorer";
import { FactorsImpactSimulator } from "./pages/FactorsImpactSimulator";
import { FactorsMappingWorkbench } from "./pages/FactorsMappingWorkbench";
import { FactorsOverrideManager } from "./pages/FactorsOverrideManager";
import { FactorsQADashboard } from "./pages/FactorsQADashboard";
import { FactorsSourceConsole } from "./pages/FactorsSourceConsole";
import { GovernancePage } from "./pages/GovernancePage";
import { LoginPage } from "./pages/LoginPage";
import { OemBranding } from "./pages/OemBranding";
import { OemSignup } from "./pages/OemSignup";
import { OemSubTenants } from "./pages/OemSubTenants";
import { PricingPage } from "./pages/PricingPage";
import { CheckoutSuccess } from "./pages/CheckoutSuccess";
import { RunsPage } from "./pages/RunsPage";
import { WorkspacePage } from "./pages/WorkspacePage";

export default function App() {
  return (
    <Routes>
      <Route element={<ShellLayout />}>
        <Route path="/" element={<Navigate to="/apps/cbam" replace />} />
        <Route path="/apps/cbam" element={<RoleGuard><WorkspacePage app="cbam" title="CBAM Workspace" description="Importer compliance and XML export flow." /></RoleGuard>} />
        <Route path="/apps/csrd" element={<RoleGuard><WorkspacePage app="csrd" title="CSRD Workspace" description="ESRS reporting and evidence bundle." /></RoleGuard>} />
        <Route path="/apps/vcci" element={<RoleGuard><WorkspacePage app="vcci" title="VCCI Workspace" description="Scope 3 inventory and policy checks." /></RoleGuard>} />
        <Route path="/apps/eudr" element={<RoleGuard><WorkspacePage app="eudr" title="EUDR Workspace" description="Supplier due diligence and risk signals." /></RoleGuard>} />
        <Route path="/apps/ghg" element={<RoleGuard><WorkspacePage app="ghg" title="GHG Workspace" description="Scope calculations and intensity metrics." /></RoleGuard>} />
        <Route path="/apps/iso14064" element={<RoleGuard><WorkspacePage app="iso14064" title="ISO14064 Workspace" description="Verification controls and conformance results." /></RoleGuard>} />
        <Route path="/apps/sb253" element={<RoleGuard><WorkspacePage app="sb253" title="SB253 Workspace" description="California climate disclosure drafts and evidence pacing." /></RoleGuard>} />
        <Route path="/apps/taxonomy" element={<RoleGuard><WorkspacePage app="taxonomy" title="Taxonomy Workspace" description="EU taxonomy alignment signals and green revenue ratios." /></RoleGuard>} />
        <Route path="/runs" element={<RoleGuard><RunsPage /></RoleGuard>} />
        <Route path="/governance" element={<RoleGuard><GovernancePage /></RoleGuard>} />
        <Route path="/admin" element={<RoleGuard><AdminPage /></RoleGuard>} />
        {/* Public Factors dashboards: no auth required (Track B-2). */}
        <Route path="/factors/status" element={<FactorsCatalogStatus />} />
        <Route path="/factors/qa" element={<FactorsQADashboard />} />
        {/* Operator console (Track B-5): admin-gated via AdminGate. */}
        <Route path="/factors/explorer" element={<AdminGate><FactorsExplorer /></AdminGate>} />
        <Route path="/factors/sources" element={<AdminGate><FactorsSourceConsole /></AdminGate>} />
        <Route path="/factors/mapping" element={<AdminGate><FactorsMappingWorkbench /></AdminGate>} />
        <Route path="/factors/diff" element={<AdminGate><FactorsDiffViewer /></AdminGate>} />
        <Route path="/factors/approvals" element={<AdminGate><FactorsApprovalQueue /></AdminGate>} />
        <Route path="/factors/overrides" element={<AdminGate><FactorsOverrideManager /></AdminGate>} />
        <Route path="/factors/impact" element={<AdminGate><FactorsImpactSimulator /></AdminGate>} />
        {/* Login lands here when AdminGate sees no token. */}
        <Route path="/login" element={<LoginPage />} />
        {/* Track C-5 OEM white-label onboarding */}
        <Route path="/oem/signup" element={<OemSignup />} />
        <Route path="/oem/branding" element={<RoleGuard><OemBranding /></RoleGuard>} />
        <Route path="/oem/subtenants" element={<RoleGuard><OemSubTenants /></RoleGuard>} />
        {/* Track C-1 commercial surface — public, no auth */}
        <Route path="/pricing" element={<PricingPage />} />
        <Route path="/checkout/success" element={<CheckoutSuccess />} />
      </Route>
    </Routes>
  );
}
