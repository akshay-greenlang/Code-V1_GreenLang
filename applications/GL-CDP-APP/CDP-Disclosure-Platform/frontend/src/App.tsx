/**
 * GL-CDP-APP v1.0 - Root Application Component
 *
 * Provides the top-level layout wrapper, MUI ThemeProvider, and
 * React Router route configuration. All page components are
 * lazy-loaded for optimal bundle splitting.
 *
 * Routes (12 pages):
 *   /dashboard             - Executive dashboard with score simulation
 *   /questionnaire         - Module-by-module questionnaire wizard
 *   /questionnaire/:moduleId - Single module view with questions
 *   /scoring               - Scoring simulator and what-if
 *   /gaps                  - Gap analysis with recommendations
 *   /benchmarking          - Sector benchmarking
 *   /supply-chain          - Supply chain management
 *   /transition            - Transition plan builder
 *   /verification          - Verification management
 *   /reports               - Report generation and export
 *   /historical            - Historical year comparison
 *   /settings              - Application settings
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider, CssBaseline } from '@mui/material';
import Layout from './components/layout/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';

// ---------------------------------------------------------------------------
// Lazy-loaded pages
// ---------------------------------------------------------------------------

const DashboardPage = lazy(() => import('./pages/Dashboard'));
const QuestionnaireWizardPage = lazy(() => import('./pages/QuestionnaireWizard'));
const ModuleDetailPage = lazy(() => import('./pages/ModuleDetail'));
const ScoringSimulatorPage = lazy(() => import('./pages/ScoringSimulator'));
const GapAnalysisPage = lazy(() => import('./pages/GapAnalysis'));
const BenchmarkingPage = lazy(() => import('./pages/Benchmarking'));
const SupplyChainPage = lazy(() => import('./pages/SupplyChain'));
const TransitionPlanPage = lazy(() => import('./pages/TransitionPlan'));
const VerificationPage = lazy(() => import('./pages/Verification'));
const ReportsPage = lazy(() => import('./pages/Reports'));
const HistoricalPage = lazy(() => import('./pages/Historical'));
const SettingsPage = lazy(() => import('./pages/Settings'));

// ---------------------------------------------------------------------------
// MUI Theme (GreenLang CDP brand)
// ---------------------------------------------------------------------------

const theme = createTheme({
  palette: {
    primary: {
      main: '#1b5e20',
      light: '#4c8c4a',
      dark: '#003300',
    },
    secondary: {
      main: '#1565c0',
    },
    success: {
      main: '#2e7d32',
    },
    error: {
      main: '#e53935',
    },
    warning: {
      main: '#ef6c00',
    },
    background: {
      default: '#f5f7f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 700,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCard: {
      defaultProps: {
        elevation: 0,
      },
      styleOverrides: {
        root: {
          border: '1px solid #e0e0e0',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
});

// ---------------------------------------------------------------------------
// App Component
// ---------------------------------------------------------------------------

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Layout>
        <Suspense fallback={<LoadingSpinner message="Loading page..." />}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/questionnaire" element={<QuestionnaireWizardPage />} />
            <Route path="/questionnaire/:moduleId" element={<ModuleDetailPage />} />
            <Route path="/scoring" element={<ScoringSimulatorPage />} />
            <Route path="/gaps" element={<GapAnalysisPage />} />
            <Route path="/benchmarking" element={<BenchmarkingPage />} />
            <Route path="/supply-chain" element={<SupplyChainPage />} />
            <Route path="/transition" element={<TransitionPlanPage />} />
            <Route path="/verification" element={<VerificationPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/historical" element={<HistoricalPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Suspense>
      </Layout>
    </ThemeProvider>
  );
};

export default App;
