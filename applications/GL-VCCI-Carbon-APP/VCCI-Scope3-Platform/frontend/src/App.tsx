import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DataUpload from './pages/DataUpload';
import SupplierManagement from './pages/SupplierManagement';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import UncertaintyAnalysis from './pages/UncertaintyAnalysis';
import CDPManagement from './pages/CDPManagement';
import ComplianceDashboard from './pages/ComplianceDashboard';

const App: React.FC = () => {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/data-upload" element={<DataUpload />} />
        <Route path="/suppliers" element={<SupplierManagement />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/uncertainty" element={<UncertaintyAnalysis />} />
        <Route path="/cdp" element={<CDPManagement />} />
        <Route path="/compliance" element={<ComplianceDashboard />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </Layout>
  );
};

export default App;
