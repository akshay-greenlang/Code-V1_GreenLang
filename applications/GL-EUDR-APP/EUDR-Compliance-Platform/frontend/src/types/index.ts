/**
 * GL-EUDR-APP TypeScript Type Definitions
 *
 * All interfaces and enums matching the backend models for the
 * EU Deforestation Regulation compliance platform.
 */

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

export enum EUDRCommodity {
  CATTLE = 'cattle',
  COCOA = 'cocoa',
  COFFEE = 'coffee',
  OIL_PALM = 'oil_palm',
  RUBBER = 'rubber',
  SOYA = 'soya',
  WOOD = 'wood',
}

export enum RiskLevel {
  LOW = 'low',
  STANDARD = 'standard',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export enum DDSStatus {
  DRAFT = 'draft',
  PENDING_REVIEW = 'pending_review',
  VALIDATED = 'validated',
  SUBMITTED = 'submitted',
  ACCEPTED = 'accepted',
  REJECTED = 'rejected',
  AMENDED = 'amended',
}

export enum PipelineStage {
  DATA_COLLECTION = 'data_collection',
  GEO_VALIDATION = 'geo_validation',
  DEFORESTATION_CHECK = 'deforestation_check',
  RISK_ASSESSMENT = 'risk_assessment',
  DDS_GENERATION = 'dds_generation',
  FINAL_REVIEW = 'final_review',
}

export enum DocumentType {
  CERTIFICATE_OF_ORIGIN = 'certificate_of_origin',
  PHYTOSANITARY_CERTIFICATE = 'phytosanitary_certificate',
  BILL_OF_LADING = 'bill_of_lading',
  CUSTOMS_DECLARATION = 'customs_declaration',
  SUSTAINABILITY_CERTIFICATE = 'sustainability_certificate',
  LAND_TITLE = 'land_title',
  SATELLITE_IMAGERY = 'satellite_imagery',
  AUDIT_REPORT = 'audit_report',
  SUPPLIER_DECLARATION = 'supplier_declaration',
  GPS_COORDINATES = 'gps_coordinates',
  OTHER = 'other',
}

export enum ComplianceStatus {
  COMPLIANT = 'compliant',
  NON_COMPLIANT = 'non_compliant',
  PENDING = 'pending',
  UNDER_REVIEW = 'under_review',
  EXPIRED = 'expired',
}

// ---------------------------------------------------------------------------
// Supplier
// ---------------------------------------------------------------------------

export interface Supplier {
  id: string;
  name: string;
  country: string;
  region: string;
  tax_id: string;
  commodities: EUDRCommodity[];
  risk_level: RiskLevel;
  compliance_status: ComplianceStatus;
  contact_name: string;
  contact_email: string;
  contact_phone: string;
  address: string;
  certifications: string[];
  last_audit_date: string | null;
  next_audit_date: string | null;
  total_plots: number;
  active_dds_count: number;
  notes: string;
  created_at: string;
  updated_at: string;
}

export interface SupplierCreateRequest {
  name: string;
  country: string;
  region: string;
  tax_id: string;
  commodities: EUDRCommodity[];
  contact_name: string;
  contact_email: string;
  contact_phone?: string;
  address?: string;
  certifications?: string[];
  notes?: string;
}

export interface SupplierUpdateRequest {
  name?: string;
  country?: string;
  region?: string;
  tax_id?: string;
  commodities?: EUDRCommodity[];
  contact_name?: string;
  contact_email?: string;
  contact_phone?: string;
  address?: string;
  certifications?: string[];
  risk_level?: RiskLevel;
  compliance_status?: ComplianceStatus;
  notes?: string;
}

// ---------------------------------------------------------------------------
// Plot
// ---------------------------------------------------------------------------

export interface GeoCoordinate {
  latitude: number;
  longitude: number;
}

export interface Plot {
  id: string;
  supplier_id: string;
  supplier_name: string;
  name: string;
  country: string;
  region: string;
  commodity: EUDRCommodity;
  area_hectares: number;
  coordinates: GeoCoordinate[];
  centroid: GeoCoordinate;
  deforestation_free: boolean | null;
  deforestation_check_date: string | null;
  satellite_source: string | null;
  risk_level: RiskLevel;
  compliance_status: ComplianceStatus;
  notes: string;
  created_at: string;
  updated_at: string;
}

export interface PlotCreateRequest {
  supplier_id: string;
  name: string;
  country: string;
  region: string;
  commodity: EUDRCommodity;
  area_hectares: number;
  coordinates: GeoCoordinate[];
  notes?: string;
}

export interface PlotValidationResult {
  plot_id: string;
  is_valid: boolean;
  deforestation_free: boolean;
  confidence_score: number;
  satellite_source: string;
  analysis_date: string;
  ndvi_change: number;
  forest_cover_percentage: number;
  reference_date: string;
  issues: PlotValidationIssue[];
}

export interface PlotValidationIssue {
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  affected_area_hectares?: number;
}

// ---------------------------------------------------------------------------
// Due Diligence Statement (DDS)
// ---------------------------------------------------------------------------

export interface DueDiligenceStatement {
  id: string;
  reference_number: string;
  supplier_id: string;
  supplier_name: string;
  commodity: EUDRCommodity;
  status: DDSStatus;
  plot_ids: string[];
  total_quantity_kg: number;
  total_area_hectares: number;
  risk_level: RiskLevel;
  risk_mitigation_measures: string[];
  submission_date: string | null;
  expiry_date: string | null;
  eu_authority: string | null;
  operator_name: string;
  operator_address: string;
  operator_eori: string;
  generated_at: string;
  validated_at: string | null;
  submitted_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface DDSGenerateRequest {
  supplier_id: string;
  commodity: EUDRCommodity;
  plot_ids: string[];
  total_quantity_kg: number;
  operator_name: string;
  operator_address: string;
  operator_eori: string;
  risk_mitigation_measures?: string[];
}

export interface DDSValidationResult {
  dds_id: string;
  is_valid: boolean;
  completeness_score: number;
  issues: DDSValidationIssue[];
  field_validations: Record<string, boolean>;
  recommendation: string;
}

export interface DDSValidationIssue {
  severity: 'error' | 'warning' | 'info';
  field: string;
  code: string;
  message: string;
}

export interface DDSSubmissionResult {
  dds_id: string;
  submitted: boolean;
  submission_reference: string;
  submission_date: string;
  eu_authority: string;
  confirmation_url: string | null;
  errors: string[];
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

export interface Document {
  id: string;
  name: string;
  file_name: string;
  document_type: DocumentType;
  mime_type: string;
  file_size_bytes: number;
  supplier_id: string | null;
  supplier_name: string | null;
  plot_id: string | null;
  dds_id: string | null;
  verification_status: 'pending' | 'verified' | 'rejected' | 'expired';
  verification_date: string | null;
  verification_notes: string | null;
  expiry_date: string | null;
  uploaded_by: string;
  uploaded_at: string;
  created_at: string;
  updated_at: string;
}

export interface DocumentUploadRequest {
  file: File;
  name: string;
  document_type: DocumentType;
  supplier_id?: string;
  plot_id?: string;
  dds_id?: string;
}

export interface DocumentVerificationResult {
  document_id: string;
  is_verified: boolean;
  confidence_score: number;
  extracted_data: Record<string, string>;
  issues: DocumentVerificationIssue[];
}

export interface DocumentVerificationIssue {
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
}

export interface DocumentGapAnalysis {
  supplier_id: string;
  dds_id: string | null;
  required_documents: RequiredDocument[];
  completeness_percentage: number;
  missing_critical: number;
  missing_recommended: number;
}

export interface RequiredDocument {
  document_type: DocumentType;
  label: string;
  required: boolean;
  provided: boolean;
  document_id: string | null;
  verification_status: string | null;
  notes: string;
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

export interface PipelineRun {
  id: string;
  supplier_id: string;
  supplier_name: string;
  commodity: EUDRCommodity;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_stage: PipelineStage;
  stages: StageResult[];
  progress_percentage: number;
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface StageResult {
  stage: PipelineStage;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  output_summary: string;
  errors: string[];
  warnings: string[];
}

export interface PipelineStartRequest {
  supplier_id: string;
  commodity: EUDRCommodity;
  plot_ids?: string[];
  skip_stages?: PipelineStage[];
  force_rerun?: boolean;
}

// ---------------------------------------------------------------------------
// Risk
// ---------------------------------------------------------------------------

export interface RiskAssessment {
  id: string;
  supplier_id: string;
  supplier_name: string;
  overall_risk: RiskLevel;
  country_risk: RiskLevel;
  commodity_risk: RiskLevel;
  deforestation_risk: RiskLevel;
  documentation_risk: RiskLevel;
  compliance_risk: RiskLevel;
  risk_score: number;
  factors: RiskFactor[];
  recommendations: string[];
  assessed_at: string;
}

export interface RiskFactor {
  category: string;
  name: string;
  weight: number;
  score: number;
  description: string;
}

export interface RiskAlert {
  id: string;
  supplier_id: string;
  supplier_name: string;
  alert_type: string;
  severity: RiskLevel;
  title: string;
  description: string;
  is_read: boolean;
  is_resolved: boolean;
  created_at: string;
  resolved_at: string | null;
}

export interface RiskTrendPoint {
  date: string;
  risk_score: number;
  risk_level: RiskLevel;
  event_count: number;
}

export interface RiskHeatmapEntry {
  country: string;
  commodity: EUDRCommodity;
  risk_level: RiskLevel;
  risk_score: number;
  supplier_count: number;
  plot_count: number;
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export interface DashboardMetrics {
  total_suppliers: number;
  total_plots: number;
  total_dds: number;
  total_documents: number;
  compliance_rate: number;
  high_risk_suppliers: number;
  pending_dds: number;
  active_pipelines: number;
  deforestation_free_plots: number;
  avg_risk_score: number;
  documents_verified: number;
  documents_pending: number;
}

export interface ComplianceTrend {
  date: string;
  compliant: number;
  non_compliant: number;
  pending: number;
  total: number;
  compliance_rate: number;
}

export interface AlertNotification {
  id: string;
  type: 'risk' | 'compliance' | 'document' | 'pipeline' | 'system';
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  entity_type: string;
  entity_id: string;
  is_read: boolean;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Generic / Shared
// ---------------------------------------------------------------------------

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface ApiError {
  status: number;
  code: string;
  message: string;
  details: Record<string, string[]> | null;
  timestamp: string;
  request_id: string;
}

export interface PaginationParams {
  page?: number;
  per_page?: number;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
  search?: string;
}

export interface SupplierFilterParams extends PaginationParams {
  country?: string;
  commodity?: EUDRCommodity;
  risk_level?: RiskLevel;
  compliance_status?: ComplianceStatus;
}

export interface PlotFilterParams extends PaginationParams {
  supplier_id?: string;
  country?: string;
  commodity?: EUDRCommodity;
  deforestation_free?: boolean;
  risk_level?: RiskLevel;
}

export interface DDSFilterParams extends PaginationParams {
  supplier_id?: string;
  commodity?: EUDRCommodity;
  status?: DDSStatus;
}

export interface DocumentFilterParams extends PaginationParams {
  supplier_id?: string;
  document_type?: DocumentType;
  verification_status?: string;
}

export interface SettingsProfile {
  organization_name: string;
  operator_name: string;
  operator_address: string;
  operator_eori: string;
  default_eu_authority: string;
  notification_email: string;
  auto_validate_dds: boolean;
  auto_submit_low_risk: boolean;
  risk_threshold_high: number;
  risk_threshold_critical: number;
}
