/**
 * GL-GHG Corporate Platform - Root Application Component
 *
 * Provides the top-level layout wrapper and route configuration.
 * All page components are lazy-loaded for optimal bundle splitting.
 *
 * Routes:
 *   /              - Dashboard (executive overview)
 *   /setup         - Inventory setup (org, entities, boundaries)
 *   /scope1        - Scope 1 direct emissions
 *   /scope2        - Scope 2 indirect emissions (electricity/heat/steam)
 *   /scope3        - Scope 3 value chain emissions
 *   /reports       - Report generation and disclosure
 *   /targets       - Reduction targets and SBTi alignment
 *   /verification  - Third-party verification workflow
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';

const DashboardPage = lazy(() => import('./pages/Dashboard'));
const SetupPage = lazy(() => import('./pages/InventorySetup'));
const Scope1Page = lazy(() => import('./pages/Scope1'));
const Scope2Page = lazy(() => import('./pages/Scope2'));
const Scope3Page = lazy(() => import('./pages/Scope3'));
const ReportsPage = lazy(() => import('./pages/Reports'));
const TargetsPage = lazy(() => import('./pages/Targets'));
const VerificationPage = lazy(() => import('./pages/Verification'));

const App: React.FC = () => {
  return (
    <Layout>
      <Suspense fallback={<LoadingSpinner message="Loading page..." />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/setup" element={<SetupPage />} />
          <Route path="/scope1" element={<Scope1Page />} />
          <Route path="/scope2" element={<Scope2Page />} />
          <Route path="/scope3" element={<Scope3Page />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/targets" element={<TargetsPage />} />
          <Route path="/verification" element={<VerificationPage />} />
        </Routes>
      </Suspense>
    </Layout>
  );
};

export default App;
