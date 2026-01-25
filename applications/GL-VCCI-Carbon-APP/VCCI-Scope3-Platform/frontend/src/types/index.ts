/**
 * GL-VCCI Frontend - TypeScript Type Definitions
 * Central type definitions for the application
 */

// ==============================================================================
// Authentication & User Types
// ==============================================================================

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  tenantId: string;
  tenantName: string;
  createdAt: string;
  lastLogin?: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  error: string | null;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface LoginResponse {
  token: string;
  user: User;
}

// ==============================================================================
// Transaction & Data Types
// ==============================================================================

export interface Transaction {
  id: string;
  transactionId: string;
  date: string;
  supplierName: string;
  supplierId?: string;
  productName: string;
  productCategory?: string;
  quantity: number;
  unit: string;
  spendUsd: number;
  currency: string;
  ghgCategory: number;
  country: string;
  description?: string;
  emissionsKgCO2e?: number;
  dqi?: number;
  status: 'pending' | 'processed' | 'verified' | 'failed';
  createdAt: string;
  updatedAt: string;
}

export interface TransactionUpload {
  file: File;
  format: 'csv' | 'excel' | 'json' | 'xml';
}

export interface UploadResponse {
  jobId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  totalRecords: number;
  processedRecords: number;
  failedRecords: number;
  errors?: Array<{
    row: number;
    field: string;
    message: string;
  }>;
}

// ==============================================================================
// Supplier Types
// ==============================================================================

export interface Supplier {
  id: string;
  name: string;
  supplierId: string;
  country: string;
  industry?: string;
  contact?: {
    email: string;
    phone?: string;
    name?: string;
  };
  totalSpendUsd: number;
  totalEmissionsKgCO2e: number;
  dataQuality: 'tier1' | 'tier2' | 'tier3';
  lastEngagement?: string;
  responseRate: number;
  pcfCount: number;
  status: 'active' | 'pending' | 'inactive';
  createdAt: string;
}

export interface SupplierEngagement {
  supplierId: string;
  campaignId: string;
  status: 'pending' | 'sent' | 'responded' | 'declined';
  sentAt: string;
  respondedAt?: string;
  pcfSubmitted: number;
}

// ==============================================================================
// Emissions & Calculations Types
// ==============================================================================

export interface EmissionFactor {
  id: string;
  name: string;
  category: number;
  subcategory?: string;
  unit: string;
  kgCO2ePerUnit: number;
  source: 'ecoinvent' | 'defra' | 'epa' | 'custom';
  region: string;
  year: number;
  dataQuality: number;
}

export interface EmissionResult {
  transactionId: string;
  category: number;
  emissionsKgCO2e: number;
  emissionsTCO2e: number;
  uncertainty?: {
    lower: number;
    upper: number;
    confidence: number;
  };
  methodology: string;
  factorUsed: EmissionFactor;
  dqi: number;
  provenance: {
    calculatedAt: string;
    calculatedBy: string;
    version: string;
  };
}

// ==============================================================================
// Dashboard & Analytics Types
// ==============================================================================

export interface DashboardMetrics {
  totalEmissionsTCO2e: number;
  totalSpendUsd: number;
  supplierCount: number;
  transactionCount: number;
  dataQualityAvg: number;
  categoryBreakdown: CategoryEmissions[];
  monthlyTrend: MonthlyData[];
  topSuppliers: SupplierEmissions[];
}

export interface CategoryEmissions {
  category: number;
  categoryName: string;
  emissionsTCO2e: number;
  percentage: number;
  spendUsd: number;
}

export interface MonthlyData {
  month: string;
  emissionsTCO2e: number;
  spendUsd: number;
  transactionCount: number;
}

export interface SupplierEmissions {
  supplierId: string;
  supplierName: string;
  emissionsTCO2e: number;
  spendUsd: number;
  percentage: number;
}

export interface HotspotAnalysis {
  hotspots: Hotspot[];
  totalPotentialReduction: number;
  totalCostSavings: number;
  analyzedAt: string;
}

export interface Hotspot {
  id: string;
  type: 'supplier' | 'category' | 'product';
  name: string;
  emissionsTCO2e: number;
  potentialReduction: number;
  reductionPercentage: number;
  recommendations: string[];
  priority: 'high' | 'medium' | 'low';
  roi?: number;
}

// ==============================================================================
// Report Types
// ==============================================================================

export interface Report {
  id: string;
  name: string;
  type: 'esrs_e1' | 'cdp' | 'ghg_protocol' | 'iso_14083' | 'ifrs_s2';
  status: 'draft' | 'generating' | 'completed' | 'failed';
  reportingPeriod: {
    startDate: string;
    endDate: string;
  };
  fileUrl?: string;
  fileFormat: 'pdf' | 'excel' | 'json';
  createdAt: string;
  generatedAt?: string;
  createdBy: string;
}

export interface ReportRequest {
  type: Report['type'];
  startDate: string;
  endDate: string;
  format: Report['fileFormat'];
  options?: {
    includeUncertainty?: boolean;
    includeProvenance?: boolean;
    categories?: number[];
  };
}

// ==============================================================================
// UI State Types
// ==============================================================================

export interface LoadingState {
  [key: string]: boolean;
}

export interface ErrorState {
  [key: string]: string | null;
}

export interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

export interface FilterState {
  search?: string;
  dateRange?: {
    startDate: string;
    endDate: string;
  };
  categories?: number[];
  suppliers?: string[];
  status?: string[];
}

export interface TableColumn {
  field: string;
  headerName: string;
  width?: number;
  sortable?: boolean;
  filterable?: boolean;
  renderCell?: (value: any) => React.ReactNode;
}

// ==============================================================================
// API Response Types
// ==============================================================================

export interface ApiResponse<T> {
  data: T;
  message?: string;
  success: boolean;
}

export interface ApiError {
  error: string;
  message: string;
  statusCode: number;
  requestId?: string;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

// ==============================================================================
// Settings & Configuration Types
// ==============================================================================

export interface UserSettings {
  userId: string;
  preferences: {
    theme: 'light' | 'dark';
    language: 'en' | 'es' | 'fr' | 'de';
    timezone: string;
    dateFormat: string;
    currency: string;
  };
  notifications: {
    email: boolean;
    inApp: boolean;
    weeklyReport: boolean;
  };
  dashboard: {
    defaultView: 'overview' | 'detailed';
    defaultPeriod: '30d' | '90d' | '1y' | 'ytd';
  };
}

// ==============================================================================
// GHG Category Definitions
// ==============================================================================

export const GHG_CATEGORIES = {
  1: 'Purchased Goods & Services',
  2: 'Capital Goods',
  3: 'Fuel & Energy Related Activities',
  4: 'Upstream Transportation & Distribution',
  5: 'Waste Generated in Operations',
  6: 'Business Travel',
  7: 'Employee Commuting',
  8: 'Upstream Leased Assets',
  9: 'Downstream Transportation & Distribution',
  10: 'Processing of Sold Products',
  11: 'Use of Sold Products',
  12: 'End-of-Life Treatment',
  13: 'Downstream Leased Assets',
  14: 'Franchises',
  15: 'Investments',
} as const;

export type GHGCategory = keyof typeof GHG_CATEGORIES;
