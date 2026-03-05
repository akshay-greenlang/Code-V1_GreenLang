/**
 * GL-Taxonomy-APP Root Application Component
 *
 * Defines all routes for the 14-page EU Taxonomy Alignment Platform.
 * Uses React.lazy for code-splitting and Layout wrapper for consistent navigation.
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import Layout from './components/layout/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const ActivityScreening = lazy(() => import('./pages/ActivityScreening'));
const SubstantialContribution = lazy(() => import('./pages/SubstantialContribution'));
const DNSHAssessment = lazy(() => import('./pages/DNSHAssessment'));
const MinimumSafeguards = lazy(() => import('./pages/MinimumSafeguards'));
const KPICalculator = lazy(() => import('./pages/KPICalculator'));
const GARCalculator = lazy(() => import('./pages/GARCalculator'));
const AlignmentWorkflow = lazy(() => import('./pages/AlignmentWorkflow'));
const Reporting = lazy(() => import('./pages/Reporting'));
const PortfolioManagement = lazy(() => import('./pages/PortfolioManagement'));
const DataQuality = lazy(() => import('./pages/DataQuality'));
const GapAnalysis = lazy(() => import('./pages/GapAnalysis'));
const RegulatoryUpdates = lazy(() => import('./pages/RegulatoryUpdates'));
const Settings = lazy(() => import('./pages/Settings'));

const LoadingFallback: React.FC = () => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '60vh' }}>
    <CircularProgress size={48} />
  </Box>
);

const App: React.FC = () => {
  return (
    <Layout>
      <Suspense fallback={<LoadingFallback />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/screening" element={<ActivityScreening />} />
          <Route path="/substantial-contribution" element={<SubstantialContribution />} />
          <Route path="/dnsh" element={<DNSHAssessment />} />
          <Route path="/safeguards" element={<MinimumSafeguards />} />
          <Route path="/kpi" element={<KPICalculator />} />
          <Route path="/gar" element={<GARCalculator />} />
          <Route path="/alignment" element={<AlignmentWorkflow />} />
          <Route path="/reporting" element={<Reporting />} />
          <Route path="/portfolio" element={<PortfolioManagement />} />
          <Route path="/data-quality" element={<DataQuality />} />
          <Route path="/gap-analysis" element={<GapAnalysis />} />
          <Route path="/regulatory" element={<RegulatoryUpdates />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Suspense>
    </Layout>
  );
};

export default App;
