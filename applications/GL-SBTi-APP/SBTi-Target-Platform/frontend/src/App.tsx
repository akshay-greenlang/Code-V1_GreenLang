/**
 * GL-SBTi-APP Root Application Component
 *
 * Defines all routes for the 14-page SBTi Target Validation & Progress Platform.
 * Uses React.lazy for code-splitting and Layout wrapper for consistent navigation.
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import Layout from './components/layout/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const TargetConfiguration = lazy(() => import('./pages/TargetConfiguration'));
const PathwayCalculator = lazy(() => import('./pages/PathwayCalculator'));
const ValidationChecker = lazy(() => import('./pages/ValidationChecker'));
const ProgressTracking = lazy(() => import('./pages/ProgressTracking'));
const TemperatureScoring = lazy(() => import('./pages/TemperatureScoring'));
const Scope3Screening = lazy(() => import('./pages/Scope3Screening'));
const FLAGAssessment = lazy(() => import('./pages/FLAGAssessment'));
const FinancialInstitutions = lazy(() => import('./pages/FinancialInstitutions'));
const RecalculationReview = lazy(() => import('./pages/RecalculationReview'));
const Reports = lazy(() => import('./pages/Reports'));
const FrameworkAlignment = lazy(() => import('./pages/FrameworkAlignment'));
const GapAnalysis = lazy(() => import('./pages/GapAnalysis'));
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
          <Route path="/targets" element={<TargetConfiguration />} />
          <Route path="/pathways" element={<PathwayCalculator />} />
          <Route path="/validation" element={<ValidationChecker />} />
          <Route path="/progress" element={<ProgressTracking />} />
          <Route path="/temperature" element={<TemperatureScoring />} />
          <Route path="/scope3" element={<Scope3Screening />} />
          <Route path="/flag" element={<FLAGAssessment />} />
          <Route path="/financial-institutions" element={<FinancialInstitutions />} />
          <Route path="/recalculation" element={<RecalculationReview />} />
          <Route path="/reports" element={<Reports />} />
          <Route path="/framework-alignment" element={<FrameworkAlignment />} />
          <Route path="/gap-analysis" element={<GapAnalysis />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Suspense>
    </Layout>
  );
};

export default App;
