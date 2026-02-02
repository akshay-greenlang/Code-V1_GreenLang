import { Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';

// Layouts
import AdminLayout from '@/layouts/AdminLayout';
import UserLayout from '@/layouts/UserLayout';
import AuthLayout from '@/layouts/AuthLayout';

// Loading component
const LoadingSpinner = () => (
  <div className="flex h-screen items-center justify-center">
    <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
  </div>
);

// Lazy load pages for code splitting
// Admin Pages
const Dashboard = lazy(() => import('@/pages/admin/Dashboard'));
const AgentList = lazy(() => import('@/pages/admin/AgentList'));
const AgentDetail = lazy(() => import('@/pages/admin/AgentDetail'));
const UserManagement = lazy(() => import('@/pages/admin/UserManagement'));
const TenantManagement = lazy(() => import('@/pages/admin/TenantManagement'));

// User Pages
const Home = lazy(() => import('@/pages/user/Home'));
const FuelAnalyzer = lazy(() => import('@/pages/user/FuelAnalyzer'));
const CBAMCalculator = lazy(() => import('@/pages/user/CBAMCalculator'));
const BuildingEnergy = lazy(() => import('@/pages/user/BuildingEnergy'));
const EUDRCompliance = lazy(() => import('@/pages/user/EUDRCompliance'));
const Reports = lazy(() => import('@/pages/user/Reports'));

// Auth Pages
const Login = lazy(() => import('@/pages/auth/Login'));
const Register = lazy(() => import('@/pages/auth/Register'));
const ForgotPassword = lazy(() => import('@/pages/auth/ForgotPassword'));

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        {/* Auth Routes */}
        <Route element={<AuthLayout />}>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/forgot-password" element={<ForgotPassword />} />
        </Route>

        {/* Admin Routes */}
        <Route path="/admin" element={<AdminLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="agents" element={<AgentList />} />
          <Route path="agents/:agentId" element={<AgentDetail />} />
          <Route path="users" element={<UserManagement />} />
          <Route path="tenants" element={<TenantManagement />} />
        </Route>

        {/* User Portal Routes */}
        <Route path="/" element={<UserLayout />}>
          <Route index element={<Home />} />
          <Route path="fuel-analyzer" element={<FuelAnalyzer />} />
          <Route path="cbam-calculator" element={<CBAMCalculator />} />
          <Route path="building-energy" element={<BuildingEnergy />} />
          <Route path="eudr-compliance" element={<EUDRCompliance />} />
          <Route path="reports" element={<Reports />} />
        </Route>

        {/* Catch all - redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Suspense>
  );
}

export default App;
