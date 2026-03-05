/**
 * GL-ISO14064-APP v1.0 - Root Application Component
 *
 * Provides the top-level layout wrapper, MUI ThemeProvider, and
 * React Router route configuration.  All page components are
 * lazy-loaded for optimal bundle splitting.
 *
 * Routes:
 *   /dashboard                        - Executive dashboard
 *   /organizations                    - Organization list/setup
 *   /organizations/:id                - Organization detail/edit
 *   /inventories                      - Inventory listing
 *   /inventories/:id                  - Inventory detail overview
 *   /inventories/:id/emissions        - Emission sources management
 *   /inventories/:id/removals         - Removal sources management
 *   /inventories/:id/categories       - ISO 14064-1 category details
 *   /inventories/:id/significance     - Significance assessment
 *   /inventories/:id/uncertainty      - Uncertainty analysis
 *   /inventories/:id/crosswalk        - ISO/GHG Protocol crosswalk
 *   /verification                     - Verification list
 *   /verification/:id                 - Verification detail
 *   /reports                          - Report generation and compliance
 *   /management                       - Management plan and actions
 *   /quality                          - Data quality scorecard
 *   /settings                         - Platform configuration
 */

import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { createTheme, ThemeProvider, CssBaseline } from '@mui/material';
import Layout from './components/common/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';

// ---------------------------------------------------------------------------
// Lazy-loaded pages
// ---------------------------------------------------------------------------

const DashboardPage = lazy(() => import('./pages/Dashboard'));
const OrganizationSetupPage = lazy(() => import('./pages/OrganizationSetup'));
const InventoryListPage = lazy(() => import('./pages/InventoryList'));
const InventoryDetailPage = lazy(() => import('./pages/InventoryDetail'));
const EmissionsManagementPage = lazy(() => import('./pages/EmissionsManagement'));
const RemovalsManagementPage = lazy(() => import('./pages/RemovalsManagement'));
const CategoriesPage = lazy(() => import('./pages/CategoriesPage'));
const SignificanceAssessmentPage = lazy(() => import('./pages/SignificanceAssessment'));
const UncertaintyAnalysisPage = lazy(() => import('./pages/UncertaintyAnalysis'));
const CrosswalkViewPage = lazy(() => import('./pages/CrosswalkView'));
const VerificationManagementPage = lazy(() => import('./pages/VerificationManagement'));
const ReportsPage = lazy(() => import('./pages/ReportsPage'));
const ManagementPlanPage = lazy(() => import('./pages/ManagementPlanPage'));
const QualityManagementPage = lazy(() => import('./pages/QualityManagement'));
const SettingsPage = lazy(() => import('./pages/Settings'));

// ---------------------------------------------------------------------------
// MUI Theme (GreenLang brand)
// ---------------------------------------------------------------------------

const theme = createTheme({
  palette: {
    primary: {
      main: '#1b5e20',
      light: '#4c8c4a',
      dark: '#003300',
    },
    secondary: {
      main: '#1e88e5',
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
            {/* Dashboard */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<DashboardPage />} />

            {/* Organizations */}
            <Route path="/organizations" element={<OrganizationSetupPage />} />
            <Route path="/organizations/:id" element={<OrganizationSetupPage />} />

            {/* Inventories */}
            <Route path="/inventories" element={<InventoryListPage />} />
            <Route path="/inventories/:id" element={<InventoryDetailPage />} />
            <Route path="/inventories/:id/emissions" element={<EmissionsManagementPage />} />
            <Route path="/inventories/:id/removals" element={<RemovalsManagementPage />} />
            <Route path="/inventories/:id/categories" element={<CategoriesPage />} />
            <Route path="/inventories/:id/significance" element={<SignificanceAssessmentPage />} />
            <Route path="/inventories/:id/uncertainty" element={<UncertaintyAnalysisPage />} />
            <Route path="/inventories/:id/crosswalk" element={<CrosswalkViewPage />} />

            {/* Verification */}
            <Route path="/verification" element={<VerificationManagementPage />} />
            <Route path="/verification/:id" element={<VerificationManagementPage />} />

            {/* Reports */}
            <Route path="/reports" element={<ReportsPage />} />

            {/* Management Plan & Quality */}
            <Route path="/management" element={<ManagementPlanPage />} />
            <Route path="/quality" element={<QualityManagementPage />} />

            {/* Settings */}
            <Route path="/settings" element={<SettingsPage />} />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </Suspense>
      </Layout>
    </ThemeProvider>
  );
};

export default App;
