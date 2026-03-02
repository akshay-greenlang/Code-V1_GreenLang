/**
 * GL-EUDR-APP Root Application Component
 *
 * Configures React Router with lazy-loaded page components wrapped
 * in the application Layout shell. All 7 main routes plus a
 * catch-all redirect.
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';

// ---------------------------------------------------------------------------
// Lazy-loaded page components
// ---------------------------------------------------------------------------

const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const SuppliersPage = lazy(() => import('./pages/SuppliersPage'));
const PlotsPage = lazy(() => import('./pages/PlotsPage'));
const RiskPage = lazy(() => import('./pages/RiskPage'));
const DDSPage = lazy(() => import('./pages/DDSPage'));
const DocumentsPage = lazy(() => import('./pages/DocumentsPage'));
const PipelinePage = lazy(() => import('./pages/PipelinePage'));

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

const App: React.FC = () => {
  return (
    <Layout>
      <Suspense fallback={<LoadingSpinner message="Loading page..." fullPage />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/suppliers/*" element={<SuppliersPage />} />
          <Route path="/plots/*" element={<PlotsPage />} />
          <Route path="/risk/*" element={<RiskPage />} />
          <Route path="/dds/*" element={<DDSPage />} />
          <Route path="/documents/*" element={<DocumentsPage />} />
          <Route path="/pipeline/*" element={<PipelinePage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </Layout>
  );
};

export default App;
