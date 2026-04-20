import { Navigate, Route, Routes } from "react-router-dom";
import { RoleGuard } from "./components/RoleGuard";
import { ShellLayout } from "./components/ShellLayout";
import { AdminPage } from "./pages/AdminPage";
import { FactorsCatalogStatus } from "./pages/FactorsCatalogStatus";
import { FactorsExplorer } from "./pages/FactorsExplorer";
import { GovernancePage } from "./pages/GovernancePage";
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
        <Route path="/factors/explorer" element={<RoleGuard><FactorsExplorer /></RoleGuard>} />
        {/* Public: no auth guard on the catalog-status dashboard. */}
        <Route path="/factors/status" element={<FactorsCatalogStatus />} />
      </Route>
    </Routes>
  );
}
