/**
 * GL-EUDR-APP API Client
 *
 * Axios-based API client with JWT authentication, request/response
 * interceptors, and type-safe method signatures for all EUDR endpoints.
 */

import axios, { AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import type {
  Supplier,
  SupplierCreateRequest,
  SupplierUpdateRequest,
  SupplierFilterParams,
  Plot,
  PlotCreateRequest,
  PlotFilterParams,
  PlotValidationResult,
  DueDiligenceStatement,
  DDSGenerateRequest,
  DDSFilterParams,
  DDSValidationResult,
  DDSSubmissionResult,
  Document,
  DocumentFilterParams,
  DocumentVerificationResult,
  DocumentGapAnalysis,
  PipelineRun,
  PipelineStartRequest,
  RiskAssessment,
  RiskAlert,
  RiskTrendPoint,
  RiskHeatmapEntry,
  DashboardMetrics,
  ComplianceTrend,
  AlertNotification,
  PaginatedResponse,
  SettingsProfile,
} from '../types';

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const api: AxiosInstance = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ---------------------------------------------------------------------------
// Interceptors
// ---------------------------------------------------------------------------

api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

function unwrap<T>(response: AxiosResponse<T>): T {
  return response.data;
}

// ---------------------------------------------------------------------------
// Suppliers
// ---------------------------------------------------------------------------

async function getSuppliers(
  params?: SupplierFilterParams
): Promise<PaginatedResponse<Supplier>> {
  return unwrap(await api.get('/suppliers', { params }));
}

async function getSupplier(id: string): Promise<Supplier> {
  return unwrap(await api.get(`/suppliers/${id}`));
}

async function createSupplier(data: SupplierCreateRequest): Promise<Supplier> {
  return unwrap(await api.post('/suppliers', data));
}

async function updateSupplier(
  id: string,
  data: SupplierUpdateRequest
): Promise<Supplier> {
  return unwrap(await api.put(`/suppliers/${id}`, data));
}

async function deleteSupplier(id: string): Promise<void> {
  await api.delete(`/suppliers/${id}`);
}

async function bulkImportSuppliers(
  data: SupplierCreateRequest[]
): Promise<{ imported: number; errors: string[] }> {
  return unwrap(await api.post('/suppliers/bulk-import', data));
}

async function getSupplierCompliance(
  id: string
): Promise<{ supplier_id: string; status: string; details: Record<string, unknown> }> {
  return unwrap(await api.get(`/suppliers/${id}/compliance`));
}

async function getSupplierRisk(id: string): Promise<RiskAssessment> {
  return unwrap(await api.get(`/suppliers/${id}/risk`));
}

// ---------------------------------------------------------------------------
// Plots
// ---------------------------------------------------------------------------

async function getPlots(
  params?: PlotFilterParams
): Promise<PaginatedResponse<Plot>> {
  return unwrap(await api.get('/plots', { params }));
}

async function getPlot(id: string): Promise<Plot> {
  return unwrap(await api.get(`/plots/${id}`));
}

async function createPlot(data: PlotCreateRequest): Promise<Plot> {
  return unwrap(await api.post('/plots', data));
}

async function updatePlot(
  id: string,
  data: Partial<PlotCreateRequest>
): Promise<Plot> {
  return unwrap(await api.put(`/plots/${id}`, data));
}

async function validatePlot(id: string): Promise<PlotValidationResult> {
  return unwrap(await api.post(`/plots/${id}/validate`));
}

async function checkPlotOverlaps(
  id: string
): Promise<{ overlaps: Array<{ plot_id: string; overlap_area_hectares: number }> }> {
  return unwrap(await api.post(`/plots/${id}/overlaps`));
}

async function bulkImportPlots(
  data: PlotCreateRequest[]
): Promise<{ imported: number; errors: string[] }> {
  return unwrap(await api.post('/plots/bulk-import', data));
}

// ---------------------------------------------------------------------------
// Due Diligence Statements (DDS)
// ---------------------------------------------------------------------------

async function getDDSList(
  params?: DDSFilterParams
): Promise<PaginatedResponse<DueDiligenceStatement>> {
  return unwrap(await api.get('/dds', { params }));
}

async function getDDS(id: string): Promise<DueDiligenceStatement> {
  return unwrap(await api.get(`/dds/${id}`));
}

async function generateDDS(
  data: DDSGenerateRequest
): Promise<DueDiligenceStatement> {
  return unwrap(await api.post('/dds/generate', data));
}

async function validateDDS(id: string): Promise<DDSValidationResult> {
  return unwrap(await api.post(`/dds/${id}/validate`));
}

async function submitDDS(id: string): Promise<DDSSubmissionResult> {
  return unwrap(await api.post(`/dds/${id}/submit`));
}

async function amendDDS(
  id: string,
  data: Partial<DDSGenerateRequest>
): Promise<DueDiligenceStatement> {
  return unwrap(await api.put(`/dds/${id}/amend`, data));
}

async function bulkGenerateDDS(
  requests: DDSGenerateRequest[]
): Promise<{ generated: number; errors: string[] }> {
  return unwrap(await api.post('/dds/bulk-generate', requests));
}

async function downloadDDS(
  id: string,
  format: 'pdf' | 'xml' | 'json' = 'pdf'
): Promise<Blob> {
  const response = await api.get(`/dds/${id}/download`, {
    params: { format },
    responseType: 'blob',
  });
  return response.data;
}

async function getDDSHistory(
  id: string
): Promise<Array<{ version: number; changed_at: string; changes: Record<string, unknown> }>> {
  return unwrap(await api.get(`/dds/${id}/history`));
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

async function getDocuments(
  params?: DocumentFilterParams
): Promise<PaginatedResponse<Document>> {
  return unwrap(await api.get('/documents', { params }));
}

async function uploadDocument(
  data: FormData
): Promise<Document> {
  return unwrap(
    await api.post('/documents/upload', data, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000,
    })
  );
}

async function verifyDocument(id: string): Promise<DocumentVerificationResult> {
  return unwrap(await api.post(`/documents/${id}/verify`));
}

async function linkDocument(
  id: string,
  data: { supplier_id?: string; plot_id?: string; dds_id?: string }
): Promise<Document> {
  return unwrap(await api.put(`/documents/${id}/link`, data));
}

async function deleteDocument(id: string): Promise<void> {
  await api.delete(`/documents/${id}`);
}

async function getDocumentGapAnalysis(
  supplierId: string,
  ddsId?: string
): Promise<DocumentGapAnalysis> {
  return unwrap(
    await api.get(`/documents/gap-analysis`, {
      params: { supplier_id: supplierId, dds_id: ddsId },
    })
  );
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

async function startPipeline(data: PipelineStartRequest): Promise<PipelineRun> {
  return unwrap(await api.post('/pipeline/start', data));
}

async function getPipelineStatus(id: string): Promise<PipelineRun> {
  return unwrap(await api.get(`/pipeline/${id}`));
}

async function getPipelineHistory(
  params?: { supplier_id?: string; page?: number; per_page?: number }
): Promise<PaginatedResponse<PipelineRun>> {
  return unwrap(await api.get('/pipeline/history', { params }));
}

async function retryPipeline(id: string): Promise<PipelineRun> {
  return unwrap(await api.post(`/pipeline/${id}/retry`));
}

async function cancelPipeline(id: string): Promise<PipelineRun> {
  return unwrap(await api.post(`/pipeline/${id}/cancel`));
}

// ---------------------------------------------------------------------------
// Risk
// ---------------------------------------------------------------------------

async function getRiskAssessment(supplierId: string): Promise<RiskAssessment> {
  return unwrap(await api.get(`/risk/assessment/${supplierId}`));
}

async function getRiskHeatmap(
  params?: { commodity?: string; country?: string }
): Promise<RiskHeatmapEntry[]> {
  return unwrap(await api.get('/risk/heatmap', { params }));
}

async function getRiskAlerts(
  params?: { severity?: string; is_resolved?: boolean; page?: number; per_page?: number }
): Promise<PaginatedResponse<RiskAlert>> {
  return unwrap(await api.get('/risk/alerts', { params }));
}

async function resolveRiskAlert(id: string): Promise<RiskAlert> {
  return unwrap(await api.post(`/risk/alerts/${id}/resolve`));
}

async function getRiskTrends(
  supplierId?: string,
  period?: string
): Promise<RiskTrendPoint[]> {
  return unwrap(
    await api.get('/risk/trends', {
      params: { supplier_id: supplierId, period },
    })
  );
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

async function getDashboardMetrics(): Promise<DashboardMetrics> {
  return unwrap(await api.get('/dashboard/metrics'));
}

async function getComplianceTrends(
  period?: string
): Promise<ComplianceTrend[]> {
  return unwrap(
    await api.get('/dashboard/compliance-trends', { params: { period } })
  );
}

async function getAlertNotifications(
  params?: { is_read?: boolean; type?: string }
): Promise<AlertNotification[]> {
  return unwrap(await api.get('/dashboard/alerts', { params }));
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

async function getSettings(): Promise<SettingsProfile> {
  return unwrap(await api.get('/settings'));
}

async function updateSettings(
  data: Partial<SettingsProfile>
): Promise<SettingsProfile> {
  return unwrap(await api.put('/settings', data));
}

async function testConnection(): Promise<{ status: string; latency_ms: number }> {
  return unwrap(await api.get('/settings/test-connection'));
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

const apiClient = {
  // Suppliers
  getSuppliers,
  getSupplier,
  createSupplier,
  updateSupplier,
  deleteSupplier,
  bulkImportSuppliers,
  getSupplierCompliance,
  getSupplierRisk,

  // Plots
  getPlots,
  getPlot,
  createPlot,
  updatePlot,
  validatePlot,
  checkPlotOverlaps,
  bulkImportPlots,

  // DDS
  getDDSList,
  getDDS,
  generateDDS,
  validateDDS,
  submitDDS,
  amendDDS,
  bulkGenerateDDS,
  downloadDDS,
  getDDSHistory,

  // Documents
  getDocuments,
  uploadDocument,
  verifyDocument,
  linkDocument,
  deleteDocument,
  getDocumentGapAnalysis,

  // Pipeline
  startPipeline,
  getPipelineStatus,
  getPipelineHistory,
  retryPipeline,
  cancelPipeline,

  // Risk
  getRiskAssessment,
  getRiskHeatmap,
  getRiskAlerts,
  resolveRiskAlert,
  getRiskTrends,

  // Dashboard
  getDashboardMetrics,
  getComplianceTrends,
  getAlertNotifications,

  // Settings
  getSettings,
  updateSettings,
  testConnection,
};

export default apiClient;
