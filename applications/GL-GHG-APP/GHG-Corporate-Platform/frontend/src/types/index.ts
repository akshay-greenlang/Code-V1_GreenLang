/**
 * GL-GHG Corporate Platform - TypeScript Type Definitions
 *
 * Central type definitions for the GHG Protocol Corporate Standard
 * reporting platform. Covers all scopes, inventory boundaries,
 * targets, verification, and reporting structures.
 */

/* ------------------------------------------------------------------ */
/*  Enums                                                              */
/* ------------------------------------------------------------------ */

export enum ConsolidationApproach {
  OPERATIONAL_CONTROL = 'operational_control',
  FINANCIAL_CONTROL = 'financial_control',
  EQUITY_SHARE = 'equity_share',
}

export enum Scope {
  SCOPE_1 = 'scope_1',
  SCOPE_2 = 'scope_2',
  SCOPE_3 = 'scope_3',
}

export enum InventoryStatus {
  DRAFT = 'draft',
  IN_REVIEW = 'in_review',
  APPROVED = 'approved',
  VERIFIED = 'verified',
  PUBLISHED = 'published',
}

export enum GHGGas {
  CO2 = 'CO2',
  CH4 = 'CH4',
  N2O = 'N2O',
  HFCs = 'HFCs',
  PFCs = 'PFCs',
  SF6 = 'SF6',
  NF3 = 'NF3',
}

export enum Scope1Category {
  STATIONARY_COMBUSTION = 'stationary_combustion',
  MOBILE_COMBUSTION = 'mobile_combustion',
  PROCESS_EMISSIONS = 'process_emissions',
  FUGITIVE_EMISSIONS = 'fugitive_emissions',
}

export enum Scope2Method {
  LOCATION_BASED = 'location_based',
  MARKET_BASED = 'market_based',
}

export enum Scope3Category {
  CAT_1 = 'cat_1',
  CAT_2 = 'cat_2',
  CAT_3 = 'cat_3',
  CAT_4 = 'cat_4',
  CAT_5 = 'cat_5',
  CAT_6 = 'cat_6',
  CAT_7 = 'cat_7',
  CAT_8 = 'cat_8',
  CAT_9 = 'cat_9',
  CAT_10 = 'cat_10',
  CAT_11 = 'cat_11',
  CAT_12 = 'cat_12',
  CAT_13 = 'cat_13',
  CAT_14 = 'cat_14',
  CAT_15 = 'cat_15',
}

export enum MaterialityStatus {
  MATERIAL = 'material',
  IMMATERIAL = 'immaterial',
  NOT_CALCULATED = 'not_calculated',
}

export enum DataQualityTier {
  TIER_1 = 'tier_1',
  TIER_2 = 'tier_2',
  TIER_3 = 'tier_3',
  TIER_4 = 'tier_4',
}

export enum CalculationMethod {
  SPEND_BASED = 'spend_based',
  AVERAGE_DATA = 'average_data',
  SUPPLIER_SPECIFIC = 'supplier_specific',
  HYBRID = 'hybrid',
  DIRECT_MEASUREMENT = 'direct_measurement',
}

export enum VerificationLevel {
  LIMITED = 'limited',
  REASONABLE = 'reasonable',
  NOT_VERIFIED = 'not_verified',
}

export enum VerificationStatus {
  NOT_STARTED = 'not_started',
  IN_PROGRESS = 'in_progress',
  PENDING_REVIEW = 'pending_review',
  APPROVED = 'approved',
  REJECTED = 'rejected',
}

export enum VerificationStage {
  DRAFT = 'draft',
  INTERNAL_REVIEW = 'internal_review',
  APPROVED = 'approved',
  EXTERNAL_VERIFICATION = 'external_verification',
  VERIFIED = 'verified',
}

export enum TargetType {
  ABSOLUTE = 'absolute',
  INTENSITY = 'intensity',
}

export enum TargetStatus {
  ON_TRACK = 'on_track',
  AT_RISK = 'at_risk',
  OFF_TRACK = 'off_track',
  ACHIEVED = 'achieved',
}

export enum SBTiPathway {
  ONE_POINT_FIVE = '1.5C',
  WELL_BELOW_TWO = 'well_below_2C',
}

export enum ReportFormat {
  PDF = 'pdf',
  EXCEL = 'excel',
  JSON = 'json',
  CSV = 'csv',
}

export enum ReportStatus {
  DRAFT = 'draft',
  REVIEW = 'review',
  FINAL = 'final',
  PUBLISHED = 'published',
}

export enum FindingSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export enum FindingStatus {
  OPEN = 'open',
  IN_PROGRESS = 'in_progress',
  RESOLVED = 'resolved',
  ACCEPTED = 'accepted',
}

export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
}

/* ------------------------------------------------------------------ */
/*  Core Data Models                                                   */
/* ------------------------------------------------------------------ */

export interface Organization {
  id: string;
  name: string;
  industry: string;
  country: string;
  consolidation_approach: ConsolidationApproach;
  base_year: number;
  reporting_currency: string;
  created_at: string;
  updated_at: string;
}

export interface Entity {
  id: string;
  organization_id: string;
  name: string;
  parent_id: string | null;
  entity_type: 'parent' | 'subsidiary' | 'facility' | 'operation';
  country: string;
  ownership_percentage: number;
  operational_control: boolean;
  financial_control: boolean;
  included_in_boundary: boolean;
  children: Entity[];
  created_at: string;
  updated_at: string;
}

export interface InventoryBoundary {
  id: string;
  organization_id: string;
  reporting_year: number;
  consolidation_approach: ConsolidationApproach;
  scope1_included: boolean;
  scope2_included: boolean;
  scope3_included: boolean;
  scope3_categories_included: Scope3Category[];
  geographic_scope: string;
  exclusions: string[];
  exclusion_justification: string;
  created_at: string;
  updated_at: string;
}

export interface BaseYear {
  id: string;
  organization_id: string;
  year: number;
  total_emissions_tco2e: number;
  scope1_emissions_tco2e: number;
  scope2_location_emissions_tco2e: number;
  scope2_market_emissions_tco2e: number;
  scope3_emissions_tco2e: number;
  recalculation_policy: string;
  significance_threshold_percent: number;
  locked: boolean;
  recalculation_history: Recalculation[];
  created_at: string;
  updated_at: string;
}

export interface Recalculation {
  id: string;
  organization_id: string;
  base_year_id: string;
  trigger_reason: string;
  original_emissions_tco2e: number;
  recalculated_emissions_tco2e: number;
  change_percent: number;
  recalculation_date: string;
  approved_by: string;
  notes: string;
}

/* ------------------------------------------------------------------ */
/*  GHG Gas Breakdown                                                  */
/* ------------------------------------------------------------------ */

export interface GHGGasBreakdown {
  co2_tonnes: number;
  ch4_tonnes_co2e: number;
  n2o_tonnes_co2e: number;
  hfcs_tonnes_co2e: number;
  pfcs_tonnes_co2e: number;
  sf6_tonnes_co2e: number;
  nf3_tonnes_co2e: number;
  total_co2e: number;
}

/* ------------------------------------------------------------------ */
/*  Emissions Data                                                     */
/* ------------------------------------------------------------------ */

export interface ScopeEmissions {
  scope: Scope;
  total_tco2e: number;
  gas_breakdown: GHGGasBreakdown;
  data_quality_tier: DataQualityTier;
  methodology: string;
  reporting_period_start: string;
  reporting_period_end: string;
  uncertainty_percent: number;
}

export interface EntityEmissions {
  entity_id: string;
  entity_name: string;
  emissions_tco2e: number;
  percentage_of_total: number;
  ownership_percentage: number;
  equity_share_emissions_tco2e: number;
}

export interface FacilityEmissions {
  facility_id: string;
  facility_name: string;
  country: string;
  source_categories: Record<string, number>;
  total: number;
  percent_of_scope: number;
  data_quality: number;
}

export interface Scope1Summary {
  total_tco2e: number;
  gas_breakdown: GHGGasBreakdown;
  by_category: Scope1CategoryBreakdown[];
  by_entity: EntityEmissions[];
  data_quality_tier: DataQualityTier;
}

export interface Scope1CategoryBreakdown {
  category: Scope1Category;
  emissions_tco2e: number;
  percentage_of_total: number;
  gas_breakdown: GHGGasBreakdown;
  data_quality_tier: DataQualityTier;
  source_count: number;
}

export interface Scope2Summary {
  location_based_tco2e: number;
  market_based_tco2e: number;
  gas_breakdown_location: GHGGasBreakdown;
  gas_breakdown_market: GHGGasBreakdown;
  by_energy_type: EnergyTypeBreakdown[];
  by_entity: EntityEmissions[];
  reconciliation_delta_tco2e: number;
  reconciliation_delta_percent: number;
}

export interface EnergyTypeBreakdown {
  energy_type: string;
  consumption_mwh: number;
  location_based_tco2e: number;
  market_based_tco2e: number;
  emission_factor_source: string;
}

export interface ContractualInstrument {
  id: string;
  type: 'REC' | 'PPA' | 'green_tariff' | 'GEC' | 'self_generation';
  provider: string;
  mwh: number;
  start_date: string;
  end_date: string;
  status: 'active' | 'expired' | 'pending';
  emission_factor: number;
}

export interface ReconciliationData {
  location_total: number;
  market_total: number;
  adjustments: ReconciliationAdjustment[];
}

export interface ReconciliationAdjustment {
  name: string;
  type: 'reduction' | 'increase';
  value: number;
  description: string;
}

/* ------------------------------------------------------------------ */
/*  Scope 3                                                            */
/* ------------------------------------------------------------------ */

export interface Scope3Summary {
  total_tco2e: number;
  by_category: Scope3CategoryBreakdown[];
  upstream_total_tco2e: number;
  downstream_total_tco2e: number;
  materiality_assessment: MaterialityResult[];
  data_quality_tier: DataQualityTier;
}

export interface Scope3CategoryBreakdown {
  category: Scope3Category;
  category_number: number;
  category_name: string;
  emissions_tco2e: number;
  percentage_of_total: number;
  calculation_method: string;
  data_quality_tier: DataQualityTier;
  is_material: boolean;
  is_excluded: boolean;
  exclusion_reason: string | null;
  top_contributors: Contributor[];
}

export interface MaterialityResult {
  category: Scope3Category;
  is_material: boolean;
  materiality_score: number;
  emissions_magnitude: string;
  data_availability: string;
  stakeholder_relevance: string;
  recommendation: string;
}

export interface Contributor {
  name: string;
  emissions: number;
  percentage: number;
}

/* ------------------------------------------------------------------ */
/*  GHG Inventory                                                      */
/* ------------------------------------------------------------------ */

export interface GHGInventory {
  id: string;
  organization_id: string;
  organization_name: string;
  reporting_year: number;
  boundary: InventoryBoundary;
  scope1: ScopeEmissions;
  scope2_location: ScopeEmissions;
  scope2_market: ScopeEmissions;
  scope3: ScopeEmissions;
  total_scope1_2_tco2e: number;
  total_scope1_2_3_tco2e: number;
  base_year: BaseYear;
  change_from_base_year_percent: number;
  intensity_metrics: IntensityMetric[];
  verification_status: VerificationStatus;
  verification_level: VerificationLevel;
  report_status: ReportStatus;
  status: InventoryStatus;
  created_at: string;
  updated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Intensity Metrics                                                  */
/* ------------------------------------------------------------------ */

export interface IntensityMetric {
  id: string;
  inventory_id: string;
  metric_name: string;
  numerator_tco2e: number;
  denominator_value: number;
  denominator_unit: string;
  intensity_value: number;
  intensity_unit: string;
  scope_coverage: Scope[];
  year_over_year_change_percent: number | null;
}

/* ------------------------------------------------------------------ */
/*  Data Quality & Uncertainty                                         */
/* ------------------------------------------------------------------ */

export interface UncertaintyResult {
  id: string;
  inventory_id: string;
  scope: Scope;
  category: string;
  lower_bound_tco2e: number;
  central_estimate_tco2e: number;
  upper_bound_tco2e: number;
  uncertainty_percent: number;
  confidence_level: number;
  methodology: string;
  primary_contributors: string[];
}

export interface CompletenessResult {
  inventory_id: string;
  overall_completeness_percent: number;
  scope1_completeness_percent: number;
  scope2_completeness_percent: number;
  scope3_completeness_percent: number;
  data_gaps: DataGap[];
  recommendations: string[];
  ghg_protocol_compliant: boolean;
  missing_requirements: string[];
}

export interface DataGap {
  id: string;
  scope: Scope;
  category: string;
  description: string;
  impact_assessment: string;
  estimated_emissions_tco2e: number | null;
  remediation_plan: string;
  priority: string;
  status: string;
}

export interface QualityDimensions {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  methodology: number;
}

/* ------------------------------------------------------------------ */
/*  Disclosure & Reporting                                             */
/* ------------------------------------------------------------------ */

export interface Disclosure {
  id: string;
  inventory_id: string;
  disclosure_type: string;
  section: string;
  content: string;
  is_required: boolean;
  is_complete: boolean;
  notes: string;
}

export interface Report {
  id: string;
  inventory_id: string;
  title: string;
  format: ReportFormat;
  status: ReportStatus;
  sections: string[];
  generated_at: string;
  generated_by: string;
  file_url: string | null;
  file_size_bytes: number | null;
  includes_scope1: boolean;
  includes_scope2: boolean;
  includes_scope3: boolean;
  includes_verification: boolean;
  metadata: Record<string, unknown>;
}

export interface ReportSection {
  id: string;
  name: string;
  description: string;
  required: boolean;
}

/* ------------------------------------------------------------------ */
/*  Verification                                                       */
/* ------------------------------------------------------------------ */

export interface VerificationRecord {
  id: string;
  inventory_id: string;
  verifier_name: string;
  verifier_accreditation: string;
  verification_level: VerificationLevel;
  status: VerificationStatus;
  scope_covered: Scope[];
  start_date: string;
  end_date: string | null;
  opinion: string | null;
  findings: Finding[];
  findings_summary: FindingsSummary;
  created_at: string;
  updated_at: string;
}

export interface Finding {
  id: string;
  verification_id: string;
  category: string;
  severity: FindingSeverity;
  status: FindingStatus;
  description: string;
  affected_scope: Scope | null;
  affected_category: string | null;
  emissions_impact_tco2e: number | null;
  recommendation: string;
  management_response: string | null;
  resolution_date: string | null;
  created_at: string;
}

export interface FindingsSummary {
  total_findings: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  open_count: number;
  resolved_count: number;
}

export interface VerificationStageDetail {
  stage: VerificationStage;
  status: 'completed' | 'current' | 'pending';
  completed_at?: string;
  completed_by?: string;
  finding_count: number;
}

/* ------------------------------------------------------------------ */
/*  Targets                                                            */
/* ------------------------------------------------------------------ */

export interface Target {
  id: string;
  organization_id: string;
  name: string;
  target_type: TargetType;
  scope_coverage: Scope[];
  base_year: number;
  base_year_emissions_tco2e: number;
  target_year: number;
  target_reduction_percent: number;
  target_emissions_tco2e: number;
  current_emissions_tco2e: number;
  progress_percent: number;
  status: TargetStatus;
  is_sbti_aligned: boolean;
  sbti_pathway: SBTiPathway | null;
  sbti_category: string | null;
  intensity_metric: string | null;
  annual_reduction_rate: number;
  interim_targets: InterimTarget[];
  created_at: string;
  updated_at: string;
}

export interface InterimTarget {
  year: number;
  target_reduction_percent: number;
  target_emissions_tco2e: number;
  actual_emissions_tco2e: number | null;
  on_track: boolean | null;
}

export interface TargetProgress {
  target_id: string;
  current_emissions_tco2e: number;
  progress_percent: number;
  required_annual_reduction: number;
  actual_annual_reduction: number;
  on_track: boolean;
  gap_to_target: number;
  forecast_data: ForecastPoint[];
}

export interface ForecastPoint {
  year: number;
  actual?: number;
  required: number;
  forecast?: number;
}

export interface SBTiAlignmentCheck {
  is_aligned: boolean;
  pathway: string;
  temperature_target: string;
  required_annual_reduction_percent: number;
  actual_annual_reduction_percent: number;
  gap_percent: number;
  recommendations: string[];
}

/* ------------------------------------------------------------------ */
/*  Dashboard Metrics                                                  */
/* ------------------------------------------------------------------ */

export interface DashboardMetrics {
  organization_name: string;
  reporting_year: number;
  total_emissions_tco2e: number;
  scope1_tco2e: number;
  scope2_location_tco2e: number;
  scope2_market_tco2e: number;
  scope3_tco2e: number;
  year_over_year_change_percent: number;
  base_year_change_percent: number;
  data_quality_score: number;
  completeness_percent: number;
  verification_status: VerificationStatus;
  target_progress_percent: number;
  top_emission_sources: TopEmissionSource[];
  alerts: DashboardAlert[];
}

export interface TopEmissionSource {
  source_name: string;
  scope: Scope;
  emissions_tco2e: number;
  percentage_of_total: number;
}

export interface DashboardAlert {
  id: string;
  severity: AlertSeverity;
  title: string;
  message: string;
  action_url: string | null;
  created_at: string;
  is_read: boolean;
}

export interface TrendDataPoint {
  year: number;
  month: number | null;
  period_label: string;
  scope1_tco2e: number;
  scope2_location_tco2e: number;
  scope2_market_tco2e: number;
  scope3_tco2e: number;
  total_tco2e: number;
}

export interface ScopeBreakdown {
  scope: Scope;
  total_tco2e: number;
  percentage_of_total: number;
  categories: CategoryBreakdownItem[];
  gas_breakdown: GHGGasBreakdown;
}

export interface CategoryBreakdownItem {
  category_key: string;
  category_name: string;
  emissions_tco2e: number;
  percentage_of_scope: number;
  data_quality_tier: DataQualityTier;
}

/* ------------------------------------------------------------------ */
/*  Request Types                                                      */
/* ------------------------------------------------------------------ */

export interface CreateOrganizationRequest {
  name: string;
  industry: string;
  country: string;
  consolidation_approach: ConsolidationApproach;
  base_year: number;
  reporting_currency: string;
}

export interface AddEntityRequest {
  name: string;
  entity_type: string;
  country: string;
  ownership_percentage: number;
  operational_control: boolean;
  financial_control: boolean;
  included_in_boundary: boolean;
}

export interface SetBoundaryRequest {
  reporting_year: number;
  consolidation_approach: ConsolidationApproach;
  scope1_included: boolean;
  scope2_included: boolean;
  scope3_included: boolean;
  scope3_categories_included: Scope3Category[];
  geographic_scope: string;
  exclusions: string[];
  exclusion_justification: string;
}

export interface CreateInventoryRequest {
  organization_id: string;
  reporting_year: number;
}

export interface SubmitScope1DataRequest {
  entity_id: string;
  category: Scope1Category;
  source_description: string;
  activity_data: Record<string, unknown>;
  emission_factor_source: string;
  data_quality_tier: DataQualityTier;
}

export interface SubmitScope2DataRequest {
  entity_id: string;
  energy_type: string;
  consumption_mwh: number;
  supplier: string;
  grid_region: string;
  contractual_instrument: string | null;
  emission_factor_source: string;
  data_quality_tier: DataQualityTier;
}

export interface SubmitScope3DataRequest {
  entity_id: string;
  category: Scope3Category;
  calculation_method: string;
  activity_data: Record<string, unknown>;
  emission_factor_source: string;
  data_quality_tier: DataQualityTier;
}

export interface GenerateReportRequest {
  inventory_id: string;
  format: ReportFormat;
  title: string;
  includes_scope1: boolean;
  includes_scope2: boolean;
  includes_scope3: boolean;
  includes_verification: boolean;
}

export interface SetTargetRequest {
  name: string;
  target_type: TargetType;
  scope_coverage: Scope[];
  base_year: number;
  target_year: number;
  target_reduction_percent: number;
  sbti_pathway: SBTiPathway | null;
  intensity_metric: string | null;
  interim_targets: { year: number; target_reduction_percent: number }[];
}

export interface StartVerificationRequest {
  inventory_id: string;
  verifier_name: string;
  verifier_accreditation: string;
  verification_level: VerificationLevel;
  scope_covered: Scope[];
}

export interface AddFindingRequest {
  category: string;
  severity: FindingSeverity;
  description: string;
  affected_scope: Scope | null;
  affected_category: string | null;
  emissions_impact_tco2e: number | null;
  recommendation: string;
}

export interface ExportDataRequest {
  inventory_id: string;
  format: ReportFormat;
  scopes: Scope[];
}

export interface UpdateSettingsRequest {
  reporting_currency: string;
  consolidation_approach: ConsolidationApproach;
  gwp_source: string;
  default_data_quality_tier: DataQualityTier;
}

/* ------------------------------------------------------------------ */
/*  Response Types                                                     */
/* ------------------------------------------------------------------ */

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface ApiResponse<T> {
  data: T;
  message: string;
  status: 'success' | 'error';
  timestamp: string;
}

export interface ApiError {
  error: string;
  message: string;
  status_code: number;
  details: Record<string, string[]> | null;
}

export interface AggregationResult {
  total_tco2e: number;
  gas_breakdown: GHGGasBreakdown;
  by_entity: EntityEmissions[];
  data_quality_tier: DataQualityTier;
  calculation_timestamp: string;
}

export interface ExportResult {
  file_url: string;
  file_name: string;
  file_size_bytes: number;
  format: ReportFormat;
  generated_at: string;
}

export interface SettingsResponse {
  organization_id: string;
  reporting_currency: string;
  consolidation_approach: ConsolidationApproach;
  gwp_source: string;
  default_data_quality_tier: DataQualityTier;
  available_gwp_sources: string[];
  available_currencies: string[];
}

/* ------------------------------------------------------------------ */
/*  Store State Types                                                  */
/* ------------------------------------------------------------------ */

export interface DashboardState {
  metrics: DashboardMetrics | null;
  trendData: TrendDataPoint[];
  scopeBreakdown: ScopeBreakdown[];
  alerts: DashboardAlert[];
  loading: boolean;
  error: string | null;
}

export interface InventoryState {
  currentInventory: GHGInventory | null;
  organization: Organization | null;
  entities: Entity[];
  boundary: InventoryBoundary | null;
  loading: boolean;
  error: string | null;
}

export interface Scope1State {
  summary: Scope1Summary | null;
  categories: Scope1CategoryBreakdown[];
  loading: boolean;
  error: string | null;
}

export interface Scope2State {
  summary: Scope2Summary | null;
  reconciliation: ReconciliationData | null;
  loading: boolean;
  error: string | null;
}

export interface Scope3State {
  summary: Scope3Summary | null;
  categories: Scope3CategoryBreakdown[];
  materiality: MaterialityResult[];
  loading: boolean;
  error: string | null;
}

export interface ReportsState {
  reports: Report[];
  disclosures: Disclosure[];
  completeness: CompletenessResult | null;
  generating: boolean;
  loading: boolean;
  error: string | null;
}

export interface TargetsState {
  targets: Target[];
  progress: Record<string, TargetProgress>;
  sbtiCheck: SBTiAlignmentCheck | null;
  loading: boolean;
  error: string | null;
}

export interface VerificationState {
  currentVerification: VerificationRecord | null;
  verifications: VerificationRecord[];
  loading: boolean;
  error: string | null;
}

export interface RootState {
  dashboard: DashboardState;
  inventory: InventoryState;
  scope1: Scope1State;
  scope2: Scope2State;
  scope3: Scope3State;
  reports: ReportsState;
  targets: TargetsState;
  verification: VerificationState;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

export const SCOPE3_CATEGORY_NAMES: Record<string, string> = {
  cat_1: 'Purchased Goods & Services',
  cat_2: 'Capital Goods',
  cat_3: 'Fuel & Energy Activities',
  cat_4: 'Upstream Transportation',
  cat_5: 'Waste Generated in Operations',
  cat_6: 'Business Travel',
  cat_7: 'Employee Commuting',
  cat_8: 'Upstream Leased Assets',
  cat_9: 'Downstream Transportation',
  cat_10: 'Processing of Sold Products',
  cat_11: 'Use of Sold Products',
  cat_12: 'End-of-Life Treatment',
  cat_13: 'Downstream Leased Assets',
  cat_14: 'Franchises',
  cat_15: 'Investments',
};

export const GAS_COLORS: Record<GHGGas, string> = {
  [GHGGas.CO2]: '#757575',
  [GHGGas.CH4]: '#1e88e5',
  [GHGGas.N2O]: '#43a047',
  [GHGGas.HFCs]: '#ef6c00',
  [GHGGas.PFCs]: '#8e24aa',
  [GHGGas.SF6]: '#e53935',
  [GHGGas.NF3]: '#00897b',
};

export const SCOPE_COLORS = {
  scope1: '#e53935',
  scope2: '#1e88e5',
  scope3: '#43a047',
} as const;

export const REPORT_SECTIONS: ReportSection[] = [
  { id: 'executive_summary', name: 'Executive Summary', description: 'High-level overview of emissions and key findings', required: true },
  { id: 'org_boundary', name: 'Organizational Boundary', description: 'Consolidation approach and entity coverage', required: true },
  { id: 'operational_boundary', name: 'Operational Boundary', description: 'Scope definitions and category coverage', required: true },
  { id: 'base_year', name: 'Base Year & Recalculations', description: 'Base year selection and recalculation policy', required: true },
  { id: 'scope1_detail', name: 'Scope 1 Emissions', description: 'Direct emissions by source category', required: true },
  { id: 'scope2_detail', name: 'Scope 2 Emissions', description: 'Location and market-based results', required: true },
  { id: 'scope3_detail', name: 'Scope 3 Emissions', description: 'Upstream and downstream categories', required: false },
  { id: 'total_emissions', name: 'Total Emissions Summary', description: 'Aggregated emissions across all scopes', required: true },
  { id: 'trends_analysis', name: 'Trends & Analysis', description: 'Year-over-year trends and intensity metrics', required: false },
  { id: 'targets_progress', name: 'Targets & Progress', description: 'Reduction targets and progress tracking', required: false },
  { id: 'data_quality', name: 'Data Quality Assessment', description: 'Data quality scores and methodology notes', required: true },
  { id: 'methodology', name: 'Methodology', description: 'Calculation methods and emission factors used', required: true },
  { id: 'uncertainties', name: 'Uncertainties & Limitations', description: 'Uncertainty analysis and data gaps', required: false },
  { id: 'verification', name: 'Verification Statement', description: 'Third-party verification results', required: false },
  { id: 'appendices', name: 'Appendices', description: 'Supporting data, EF tables, and detailed breakdowns', required: false },
];
