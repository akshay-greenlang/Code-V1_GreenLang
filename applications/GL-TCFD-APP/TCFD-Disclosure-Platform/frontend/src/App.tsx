/**
 * GL-TCFD-APP Root Application Component
 *
 * Defines all routes for the 15-page TCFD Disclosure Platform.
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import Layout from './components/layout/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Governance = lazy(() => import('./pages/Governance'));
const StrategyRisks = lazy(() => import('./pages/StrategyRisks'));
const StrategyOpportunities = lazy(() => import('./pages/StrategyOpportunities'));
const ScenarioAnalysis = lazy(() => import('./pages/ScenarioAnalysis'));
const PhysicalRisk = lazy(() => import('./pages/PhysicalRisk'));
const TransitionRisk = lazy(() => import('./pages/TransitionRisk'));
const Opportunities = lazy(() => import('./pages/Opportunities'));
const FinancialImpact = lazy(() => import('./pages/FinancialImpact'));
const RiskManagement = lazy(() => import('./pages/RiskManagement'));
const MetricsTargets = lazy(() => import('./pages/MetricsTargets'));
const DisclosureBuilder = lazy(() => import('./pages/DisclosureBuilder'));
const GapAnalysis = lazy(() => import('./pages/GapAnalysis'));
const ISSBCrossWalk = lazy(() => import('./pages/ISSBCrossWalk'));
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
          <Route path="/governance" element={<Governance />} />
          <Route path="/strategy/risks" element={<StrategyRisks />} />
          <Route path="/strategy/opportunities" element={<StrategyOpportunities />} />
          <Route path="/scenarios" element={<ScenarioAnalysis />} />
          <Route path="/physical-risk" element={<PhysicalRisk />} />
          <Route path="/transition-risk" element={<TransitionRisk />} />
          <Route path="/opportunities" element={<Opportunities />} />
          <Route path="/financial-impact" element={<FinancialImpact />} />
          <Route path="/risk-management" element={<RiskManagement />} />
          <Route path="/metrics-targets" element={<MetricsTargets />} />
          <Route path="/disclosure" element={<DisclosureBuilder />} />
          <Route path="/gap-analysis" element={<GapAnalysis />} />
          <Route path="/issb-crosswalk" element={<ISSBCrossWalk />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Suspense>
    </Layout>
  );
};

export default App;
