/**
 * App Component
 *
 * Main application component with routing and layout.
 */

import React, { Suspense, lazy } from 'react';
import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
} from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { Sidebar } from './components/Sidebar';

// Lazy load pages for code splitting
const Dashboard = lazy(() => import('./components/Dashboard'));
const QueueList = lazy(() => import('./components/QueueList'));
const ItemDetailView = lazy(() => import('./components/ItemDetailView'));

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30 seconds
      retry: 2,
      refetchOnWindowFocus: true,
    },
  },
});

/**
 * Loading fallback component
 */
const LoadingFallback: React.FC = () => (
  <div className="flex items-center justify-center min-h-[400px]">
    <div className="flex flex-col items-center gap-4">
      <div className="w-12 h-12 border-4 border-gl-primary-200 border-t-gl-primary-600 rounded-full animate-spin" />
      <p className="text-sm text-gl-neutral-500">Loading...</p>
    </div>
  </div>
);

/**
 * Analytics page placeholder
 */
const AnalyticsPage: React.FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold text-gl-neutral-900 mb-4">Analytics</h1>
    <div className="card p-8 text-center">
      <p className="text-gl-neutral-500">
        Analytics dashboard coming soon.
      </p>
    </div>
  </div>
);

/**
 * Settings page placeholder
 */
const SettingsPage: React.FC = () => (
  <div className="p-6">
    <h1 className="text-2xl font-bold text-gl-neutral-900 mb-4">Settings</h1>
    <div className="card p-8 text-center">
      <p className="text-gl-neutral-500">
        Settings page coming soon.
      </p>
    </div>
  </div>
);

/**
 * 404 Not Found page
 */
const NotFoundPage: React.FC = () => (
  <div className="flex items-center justify-center min-h-[60vh]">
    <div className="text-center">
      <h1 className="text-6xl font-bold text-gl-neutral-200 mb-4">404</h1>
      <h2 className="text-xl font-semibold text-gl-neutral-900 mb-2">
        Page Not Found
      </h2>
      <p className="text-gl-neutral-500 mb-6">
        The page you are looking for does not exist.
      </p>
      <a href="/" className="btn-primary">
        Go to Dashboard
      </a>
    </div>
  </div>
);

/**
 * Main layout with sidebar
 */
const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // TODO: Fetch pending count from API
  const pendingCount = 42;

  return (
    <div className="flex min-h-screen bg-gl-neutral-50">
      <Sidebar pendingCount={pendingCount} />
      <main className="flex-1 min-w-0">
        {children}
      </main>
    </div>
  );
};

/**
 * App component
 */
export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <MainLayout>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              {/* Dashboard */}
              <Route path="/" element={<Dashboard />} />

              {/* Queue routes */}
              <Route path="/queue" element={<QueueList />} />
              <Route path="/queue/:id" element={<ItemDetailView />} />

              {/* Analytics */}
              <Route path="/analytics" element={<AnalyticsPage />} />

              {/* Settings */}
              <Route path="/settings" element={<SettingsPage />} />

              {/* Redirects */}
              <Route path="/review" element={<Navigate to="/queue" replace />} />

              {/* 404 */}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </MainLayout>

        {/* Toast notifications */}
        <Toaster
          position="bottom-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#18181b',
              color: '#fff',
              borderRadius: '8px',
              padding: '12px 16px',
            },
            success: {
              iconTheme: {
                primary: '#22c55e',
                secondary: '#fff',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
