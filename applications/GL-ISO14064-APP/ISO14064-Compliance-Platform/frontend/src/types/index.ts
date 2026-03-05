/**
 * GL-ISO14064-APP v1.0 - TypeScript Type Definitions
 *
 * Central type definitions for the ISO 14064-1:2018 Compliance Platform.
 * Covers all six ISO emission/removal categories, organizational and
 * operational boundaries, quantification, data quality, verification,
 * reporting, management plans, crosswalks, and dashboard metrics.
 *
 * All monetary values are in USD.  All emissions are in metric tonnes
 * CO2e unless otherwise noted.  Timestamps are UTC ISO 8601 strings.
 */

/* ------------------------------------------------------------------ */
/*  Enums                                                              */
/* ------------------------------------------------------------------ */

export enum ISOCategory {
  CATEGORY_1_DIRECT = 'category_1_direct',
  CATEGORY_2_ENERGY = 'category_2_energy',
  CATEGORY_3_TRANSPORT = 'category_3_transport',
  CATEGORY_4_PRODUCTS_USED = 'category_4_products_used',
  CATEGORY_5_PRODUCTS_FROM_ORG = 'category_5_products_from_org',
  CATEGORY_6_OTHER = 'category_6_other',
}

export enum ConsolidationApproach {
  OPERATIONAL_CONTROL = 'operational_control',
  FINANCIAL_CONTROL = 'financial_control',
  EQUITY_SHARE = 'equity_share',
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

export enum QuantificationMethod {
  CALCULATION_BASED = 'calculation_based',
  DIRECT_MEASUREMENT = 'direct_measurement',
  MASS_BALANCE = 'mass_balance',
}

export enum RemovalType {
  FORESTRY = 'forestry',
  SOIL_CARBON = 'soil_carbon',
  CCS = 'ccs',
  DIRECT_AIR_CAPTURE = 'direct_air_capture',
  BECCS = 'beccs',
  WETLAND_RESTORATION = 'wetland_restoration',
  OCEAN_BASED = 'ocean_based',
  OTHER = 'other',
}

export enum PermanenceLevel {
  PERMANENT = 'permanent',
  LONG_TERM = 'long_term',
  MEDIUM_TERM = 'medium_term',
  SHORT_TERM = 'short_term',
  REVERSIBLE = 'reversible',
}

export enum SignificanceLevel {
  SIGNIFICANT = 'significant',
  NOT_SIGNIFICANT = 'not_significant',
  UNDER_REVIEW = 'under_review',
}

export enum DataQualityTier {
  TIER_1 = 'tier_1',
  TIER_2 = 'tier_2',
  TIER_3 = 'tier_3',
  TIER_4 = 'tier_4',
}

export enum VerificationLevel {
  LIMITED = 'limited',
  REASONABLE = 'reasonable',
  NOT_VERIFIED = 'not_verified',
}

export enum VerificationStage {
  DRAFT = 'draft',
  INTERNAL_REVIEW = 'internal_review',
  APPROVED = 'approved',
  EXTERNAL_VERIFICATION = 'external_verification',
  VERIFIED = 'verified',
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

export enum ActionStatus {
  PLANNED = 'planned',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  DEFERRED = 'deferred',
  CANCELLED = 'cancelled',
}

export enum ActionCategory {
  EMISSION_REDUCTION = 'emission_reduction',
  REMOVAL_ENHANCEMENT = 'removal_enhancement',
  DATA_IMPROVEMENT = 'data_improvement',
  PROCESS_IMPROVEMENT = 'process_improvement',
}

export enum GWPSource {
  AR5 = 'ar5',
  AR6 = 'ar6',
  CUSTOM = 'custom',
}

export enum InventoryStatus {
  DRAFT = 'draft',
  IN_REVIEW = 'in_review',
  APPROVED = 'approved',
  VERIFIED = 'verified',
  PUBLISHED = 'published',
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

export enum ReportingPeriod {
  CALENDAR_YEAR = 'calendar_year',
  FISCAL_YEAR = 'fiscal_year',
  CUSTOM = 'custom',
}

export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
}

/* ------------------------------------------------------------------ */
/*  Organization & Entity Models                                       */
/* ------------------------------------------------------------------ */

export interface Organization {
  id: string;
  name: string;
  industry: string;
  country: string;
  description: string | null;
  entities: Entity[];
  created_at: string;
  updated_at: string;
}

export interface Entity {
  id: string;
  name: string;
  entity_type: string;
  parent_id: string | null;
  ownership_pct: number;
  country: string;
  employees: number | null;
  revenue: number | null;
  floor_area_m2: number | null;
  active: boolean;
  created_at: string;
  updated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Boundary Models                                                    */
/* ------------------------------------------------------------------ */

export interface OrganizationalBoundary {
  id: string;
  org_id: string;
  consolidation_approach: ConsolidationApproach;
  entity_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface CategoryInclusion {
  category: ISOCategory;
  included: boolean;
  significance: SignificanceLevel;
  justification: string | null;
}

export interface OperationalBoundary {
  id: string;
  org_id: string;
  categories: CategoryInclusion[];
  created_at: string;
  updated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Inventory Models                                                   */
/* ------------------------------------------------------------------ */

export interface ISOInventory {
  id: string;
  org_id: string;
  reporting_year: number;
  period_start: string | null;
  period_end: string | null;
  consolidation_approach: ConsolidationApproach;
  gwp_source: GWPSource;
  status: InventoryStatus;
  created_at: string;
  updated_at: string;
  provenance_hash: string;
}

/* ------------------------------------------------------------------ */
/*  Quantification Models                                              */
/* ------------------------------------------------------------------ */

export interface EmissionSource {
  id: string;
  inventory_id: string;
  category: ISOCategory;
  source_name: string;
  facility_id: string | null;
  gas: GHGGas;
  method: QuantificationMethod;
  activity_data: number;
  activity_unit: string;
  emission_factor: number;
  ef_unit: string;
  ef_source: string;
  gwp: number;
  raw_emissions_tonnes: number;
  tco2e: number;
  biogenic_co2: number;
  data_quality_tier: DataQualityTier;
  provenance_hash: string;
  created_at: string;
  updated_at: string;
}

export interface QuantificationResult {
  id: string;
  source_id: string;
  method: QuantificationMethod;
  gas: GHGGas;
  raw_emissions_tonnes: number;
  gwp_applied: number;
  tco2e: number;
  biogenic_co2: number;
  data_quality: DataQualityIndicator;
  provenance_hash: string;
  calculated_at: string;
}

export interface DataQualityIndicator {
  completeness: number;
  accuracy: number;
  consistency: number;
  timeliness: number;
  methodology: number;
  overall_score: number;
}

/* ------------------------------------------------------------------ */
/*  Removal Models                                                     */
/* ------------------------------------------------------------------ */

export interface RemovalSource {
  id: string;
  inventory_id: string;
  facility_id: string | null;
  removal_type: RemovalType;
  source_name: string;
  gross_removals_tco2e: number;
  permanence_level: PermanenceLevel;
  permanence_discount_factor: number;
  credited_removals_tco2e: number;
  biogenic_co2_removals: number;
  biogenic_co2_emissions: number;
  verification_status: VerificationStage;
  monitoring_plan: string | null;
  data_quality_tier: DataQualityTier;
  provenance_hash: string;
  created_at: string;
  updated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Category Aggregation Models                                        */
/* ------------------------------------------------------------------ */

export interface CategoryResult {
  category: ISOCategory;
  category_name: string;
  total_tco2e: number;
  removals_tco2e: number;
  net_tco2e: number;
  by_gas: Record<string, number>;
  by_facility: Record<string, number>;
  by_source: Record<string, number>;
  biogenic_co2: number;
  significance: SignificanceLevel;
  source_count: number;
  data_quality_tier: DataQualityTier;
  provenance_hash: string;
}

export interface InventoryTotals {
  inventory_id: string;
  gross_emissions_tco2e: number;
  total_removals_tco2e: number;
  net_emissions_tco2e: number;
  category_1_net: number;
  significant_indirect_tco2e: number;
  biogenic_co2_total: number;
  by_gas: Record<string, number>;
  by_category: Record<string, number>;
  by_facility: Record<string, number>;
  yoy_change_pct: number | null;
  yoy_change_tco2e: number | null;
  provenance_hash: string;
  computed_at: string;
}

/* ------------------------------------------------------------------ */
/*  Significance Assessment Models (ISO 14064-1 Clause 5.2.2)         */
/* ------------------------------------------------------------------ */

export interface SignificanceCriterion {
  criterion: string;
  weight: number;
  score: number;
  rationale: string;
}

export interface SignificanceAssessment {
  id: string;
  inventory_id: string;
  category: ISOCategory;
  criteria: SignificanceCriterion[];
  total_weighted_score: number;
  threshold: number;
  result: SignificanceLevel;
  estimated_magnitude_tco2e: number | null;
  magnitude_pct_of_total: number | null;
  assessed_by: string;
  assessed_at: string;
}

/* ------------------------------------------------------------------ */
/*  Uncertainty Models (ISO 14064-1 Clause 6.3)                        */
/* ------------------------------------------------------------------ */

export interface UncertaintyInterval {
  confidence_level: number;
  lower_bound: number;
  upper_bound: number;
  half_width: number;
  half_width_pct: number;
}

export interface UncertaintyResult {
  id: string;
  inventory_id: string;
  mean: number;
  std_dev: number;
  cv_percent: number;
  intervals: UncertaintyInterval[];
  iterations: number;
  by_category: Record<string, Record<string, number>>;
  by_gas: Record<string, Record<string, number>>;
  computed_at: string;
}

/* ------------------------------------------------------------------ */
/*  Base Year Models (ISO 14064-1 Clause 5.3)                          */
/* ------------------------------------------------------------------ */

export interface BaseYearRecord {
  id: string;
  org_id: string;
  base_year: number;
  original_emissions_tco2e: number;
  recalculated_emissions_tco2e: number | null;
  recalculation_reason: string | null;
  recalculation_date: string | null;
  recalculation_policy: string;
  significance_threshold_pct: number;
  provenance_hash: string;
  created_at: string;
  updated_at: string;
}

export interface BaseYearTrigger {
  id: string;
  org_id: string;
  trigger_type: string;
  description: string;
  impact_tco2e: number | null;
  impact_pct: number | null;
  requires_recalculation: boolean;
  triggered_at: string;
}

/* ------------------------------------------------------------------ */
/*  Verification & Finding Models (ISO 14064-3:2019)                   */
/* ------------------------------------------------------------------ */

export interface Finding {
  id: string;
  verification_id: string;
  category: string;
  severity: FindingSeverity;
  description: string;
  affected_category: ISOCategory | null;
  emissions_impact_tco2e: number | null;
  recommendation: string;
  management_response: string | null;
  status: FindingStatus;
  created_at: string;
  resolved_at: string | null;
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

export interface VerificationRecord {
  id: string;
  inventory_id: string;
  verifier_name: string;
  verifier_accreditation: string;
  verification_level: VerificationLevel;
  scope_of_verification: string;
  stage: VerificationStage;
  opinion: string | null;
  opinion_date: string | null;
  findings: Finding[];
  findings_summary: FindingsSummary;
  created_at: string;
  updated_at: string;
}

export interface VerificationStageDetail {
  stage: VerificationStage;
  status: 'completed' | 'current' | 'pending';
  completed_at?: string;
  completed_by?: string;
  finding_count: number;
}

/* ------------------------------------------------------------------ */
/*  Report Models (ISO 14064-1 Clause 9)                               */
/* ------------------------------------------------------------------ */

export interface MandatoryElement {
  element_id: string;
  description: string;
  complete: boolean;
  content: string | null;
}

export interface ISOReport {
  id: string;
  inventory_id: string;
  org_id: string;
  title: string;
  reporting_year: number;
  format: ReportFormat;
  mandatory_elements: MandatoryElement[];
  mandatory_completeness_pct: number;
  sections: Record<string, unknown>;
  generated_at: string;
  provenance_hash: string;
}

/* ------------------------------------------------------------------ */
/*  Management Plan Models (ISO 14064-1 Clause 9)                      */
/* ------------------------------------------------------------------ */

export interface ManagementAction {
  id: string;
  org_id: string;
  title: string;
  description: string;
  action_category: ActionCategory;
  target_category: ISOCategory | null;
  status: ActionStatus;
  priority: string;
  estimated_reduction_tco2e: number | null;
  estimated_cost_usd: number | null;
  responsible_person: string;
  start_date: string | null;
  target_date: string | null;
  completion_date: string | null;
  progress_notes: string[];
  created_at: string;
  updated_at: string;
}

export interface ManagementPlan {
  id: string;
  org_id: string;
  reporting_year: number;
  objectives: string[];
  actions: ManagementAction[];
  total_planned_reduction_tco2e: number;
  total_planned_investment_usd: number;
  review_cycle: string;
  created_at: string;
  updated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Quality Management Models (ISO 14064-1 Clause 7)                   */
/* ------------------------------------------------------------------ */

export interface QualityProcedure {
  id: string;
  title: string;
  description: string;
  procedure_type: string;
  responsible: string;
  frequency: string;
  status: string;
  last_review: string | null;
  next_review: string | null;
}

export interface QualityManagementPlan {
  id: string;
  org_id: string;
  procedures: QualityProcedure[];
  data_collection_guidelines: string;
  internal_audit_schedule: string;
  corrective_action_process: string;
  roles_and_responsibilities: Record<string, string>;
  created_at: string;
  updated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Crosswalk Models                                                   */
/* ------------------------------------------------------------------ */

export interface CrosswalkMapping {
  iso_category: ISOCategory;
  iso_category_name: string;
  ghg_scope: string;
  ghg_category: string | null;
  tco2e: number;
  notes: string;
}

export interface CrosswalkResult {
  id: string;
  inventory_id: string;
  mappings: CrosswalkMapping[];
  iso_total_tco2e: number;
  ghg_protocol_total_tco2e: number;
  reconciliation_difference: number;
  reconciliation_pct: number;
  generated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Dashboard Metrics                                                  */
/* ------------------------------------------------------------------ */

export interface DashboardMetrics {
  org_id: string;
  reporting_year: number;
  gross_emissions_tco2e: number;
  total_removals_tco2e: number;
  net_emissions_tco2e: number;
  by_category: Record<string, number>;
  by_gas: Record<string, number>;
  biogenic_co2: number;
  yoy_change_pct: number | null;
  data_quality_score: number;
  completeness_pct: number;
  verification_stage: string;
  significant_categories: string[];
  management_plan_actions: number;
}

export interface TrendDataPoint {
  year: number;
  month: number | null;
  period_label: string;
  category_1_tco2e: number;
  category_2_tco2e: number;
  category_3_tco2e: number;
  category_4_tco2e: number;
  category_5_tco2e: number;
  category_6_tco2e: number;
  gross_total_tco2e: number;
  net_total_tco2e: number;
  removals_tco2e: number;
}

export interface CategoryBreakdownItem {
  category: ISOCategory;
  category_name: string;
  emissions_tco2e: number;
  removals_tco2e: number;
  net_tco2e: number;
  percentage_of_total: number;
  data_quality_tier: DataQualityTier;
  significance: SignificanceLevel;
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

/* ------------------------------------------------------------------ */
/*  Request Types                                                      */
/* ------------------------------------------------------------------ */

export interface CreateOrganizationRequest {
  name: string;
  industry: string;
  country: string;
  description?: string | null;
}

export interface AddEntityRequest {
  name: string;
  entity_type: string;
  parent_id?: string | null;
  ownership_pct?: number;
  country: string;
  employees?: number | null;
  revenue?: number | null;
  floor_area_m2?: number | null;
}

export interface UpdateEntityRequest {
  name?: string;
  ownership_pct?: number;
  employees?: number | null;
  revenue?: number | null;
  floor_area_m2?: number | null;
  active?: boolean;
}

export interface CreateInventoryRequest {
  org_id: string;
  reporting_year: number;
  consolidation_approach?: ConsolidationApproach;
  gwp_source?: GWPSource;
}

export interface AddEmissionSourceRequest {
  category: ISOCategory;
  source_name: string;
  facility_id?: string | null;
  gas?: GHGGas;
  method?: QuantificationMethod;
  activity_data?: number;
  activity_unit?: string;
  emission_factor?: number;
  ef_unit?: string;
  ef_source?: string;
  data_quality_tier?: DataQualityTier;
}

export interface AddRemovalSourceRequest {
  removal_type: RemovalType;
  source_name: string;
  facility_id?: string | null;
  gross_removals_tco2e: number;
  permanence_level?: PermanenceLevel;
  monitoring_plan?: string | null;
  data_quality_tier?: DataQualityTier;
}

export interface SetBaseYearRequest {
  base_year: number;
  emissions_tco2e: number;
  recalculation_policy?: string | null;
}

export interface RecalculateBaseYearRequest {
  trigger_type: string;
  description?: string;
  new_emissions_tco2e: number;
}

export interface AddFindingRequest {
  category: string;
  severity?: FindingSeverity;
  description: string;
  affected_category?: ISOCategory | null;
  emissions_impact_tco2e?: number | null;
  recommendation?: string;
}

export interface CreateVerificationRequest {
  inventory_id: string;
  verifier_name?: string;
  verifier_accreditation?: string;
  verification_level?: VerificationLevel;
  scope_of_verification?: string;
}

export interface CreateManagementActionRequest {
  title: string;
  description?: string;
  action_category?: ActionCategory;
  target_category?: ISOCategory | null;
  priority?: string;
  estimated_reduction_tco2e?: number | null;
  estimated_cost_usd?: number | null;
  responsible_person?: string;
  target_date?: string | null;
}

export interface GenerateReportRequest {
  inventory_id: string;
  format?: ReportFormat;
  title?: string;
}

export interface ExportDataRequest {
  inventory_id: string;
  format?: ReportFormat;
  categories?: ISOCategory[];
}

export interface SetOrganizationalBoundaryRequest {
  consolidation_approach: ConsolidationApproach;
  entity_ids: string[];
}

export interface SetOperationalBoundaryRequest {
  categories: CategoryInclusion[];
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

export interface ExportResult {
  file_url: string;
  file_name: string;
  file_size_bytes: number;
  format: ReportFormat;
  generated_at: string;
}

/* ------------------------------------------------------------------ */
/*  Store State Types                                                  */
/* ------------------------------------------------------------------ */

export interface OrganizationState {
  organization: Organization | null;
  entities: Entity[];
  organizationalBoundary: OrganizationalBoundary | null;
  operationalBoundary: OperationalBoundary | null;
  loading: boolean;
  error: string | null;
}

export interface InventoryState {
  currentInventory: ISOInventory | null;
  inventories: ISOInventory[];
  totals: InventoryTotals | null;
  categoryResults: CategoryResult[];
  baseYear: BaseYearRecord | null;
  baseYearTriggers: BaseYearTrigger[];
  loading: boolean;
  error: string | null;
}

export interface EmissionsState {
  sources: EmissionSource[];
  quantificationResults: QuantificationResult[];
  dataQuality: DataQualityIndicator | null;
  uncertainty: UncertaintyResult | null;
  loading: boolean;
  error: string | null;
}

export interface RemovalsState {
  removals: RemovalSource[];
  loading: boolean;
  error: string | null;
}

export interface SignificanceState {
  assessments: SignificanceAssessment[];
  loading: boolean;
  error: string | null;
}

export interface VerificationState {
  currentVerification: VerificationRecord | null;
  verifications: VerificationRecord[];
  loading: boolean;
  error: string | null;
}

export interface ReportsState {
  reports: ISOReport[];
  mandatoryElements: MandatoryElement[];
  completeness_pct: number;
  generating: boolean;
  loading: boolean;
  error: string | null;
}

export interface ManagementState {
  plan: ManagementPlan | null;
  actions: ManagementAction[];
  qualityPlan: QualityManagementPlan | null;
  loading: boolean;
  error: string | null;
}

export interface CrosswalkState {
  crosswalk: CrosswalkResult | null;
  loading: boolean;
  error: string | null;
}

export interface DashboardState {
  metrics: DashboardMetrics | null;
  trendData: TrendDataPoint[];
  categoryBreakdown: CategoryBreakdownItem[];
  alerts: DashboardAlert[];
  loading: boolean;
  error: string | null;
}

export interface RootState {
  organization: OrganizationState;
  inventory: InventoryState;
  emissions: EmissionsState;
  removals: RemovalsState;
  significance: SignificanceState;
  verification: VerificationState;
  reports: ReportsState;
  management: ManagementState;
  crosswalk: CrosswalkState;
  dashboard: DashboardState;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

export const ISO_CATEGORY_NAMES: Record<ISOCategory, string> = {
  [ISOCategory.CATEGORY_1_DIRECT]: 'Category 1 - Direct GHG emissions and removals',
  [ISOCategory.CATEGORY_2_ENERGY]: 'Category 2 - Indirect GHG emissions from imported energy',
  [ISOCategory.CATEGORY_3_TRANSPORT]: 'Category 3 - Indirect GHG emissions from transportation',
  [ISOCategory.CATEGORY_4_PRODUCTS_USED]: 'Category 4 - Indirect GHG emissions from products used by the organization',
  [ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG]: 'Category 5 - Indirect GHG emissions associated with the use of products from the organization',
  [ISOCategory.CATEGORY_6_OTHER]: 'Category 6 - Indirect GHG emissions from other sources',
};

export const ISO_CATEGORY_SHORT_NAMES: Record<ISOCategory, string> = {
  [ISOCategory.CATEGORY_1_DIRECT]: 'Cat 1 - Direct',
  [ISOCategory.CATEGORY_2_ENERGY]: 'Cat 2 - Energy',
  [ISOCategory.CATEGORY_3_TRANSPORT]: 'Cat 3 - Transport',
  [ISOCategory.CATEGORY_4_PRODUCTS_USED]: 'Cat 4 - Products Used',
  [ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG]: 'Cat 5 - Products From Org',
  [ISOCategory.CATEGORY_6_OTHER]: 'Cat 6 - Other',
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

export const CATEGORY_COLORS: Record<ISOCategory, string> = {
  [ISOCategory.CATEGORY_1_DIRECT]: '#e53935',
  [ISOCategory.CATEGORY_2_ENERGY]: '#1e88e5',
  [ISOCategory.CATEGORY_3_TRANSPORT]: '#43a047',
  [ISOCategory.CATEGORY_4_PRODUCTS_USED]: '#ef6c00',
  [ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG]: '#8e24aa',
  [ISOCategory.CATEGORY_6_OTHER]: '#00897b',
};

export const MANDATORY_REPORTING_ELEMENTS: string[] = [
  'MRE-01',
  'MRE-02',
  'MRE-03',
  'MRE-04',
  'MRE-05',
  'MRE-06',
  'MRE-07',
  'MRE-08',
  'MRE-09',
  'MRE-10',
  'MRE-11',
  'MRE-12',
  'MRE-13',
  'MRE-14',
];

export const MRE_DESCRIPTIONS: Record<string, string> = {
  'MRE-01': 'Reporting organization description',
  'MRE-02': 'Responsible person',
  'MRE-03': 'Reporting period',
  'MRE-04': 'Organizational boundary and consolidation approach',
  'MRE-05': 'Direct GHG emissions (Category 1)',
  'MRE-06': 'Indirect GHG emissions from imported energy (Category 2)',
  'MRE-07': 'Quantification methodology description',
  'MRE-08': 'GHG emissions and removals by gas type',
  'MRE-09': 'Emission factors and GWP values used',
  'MRE-10': 'Biogenic CO2 emissions reported separately',
  'MRE-11': 'Base year and recalculation policy',
  'MRE-12': 'Significance assessment for indirect categories (3-6)',
  'MRE-13': 'Exclusions with justification',
  'MRE-14': 'Uncertainty assessment',
};

export const MRV_AGENT_TO_ISO_CATEGORY: Record<string, ISOCategory> = {
  stationary_combustion: ISOCategory.CATEGORY_1_DIRECT,
  mobile_combustion: ISOCategory.CATEGORY_1_DIRECT,
  process_emissions: ISOCategory.CATEGORY_1_DIRECT,
  fugitive_emissions: ISOCategory.CATEGORY_1_DIRECT,
  refrigerants: ISOCategory.CATEGORY_1_DIRECT,
  land_use: ISOCategory.CATEGORY_1_DIRECT,
  waste_treatment: ISOCategory.CATEGORY_1_DIRECT,
  agricultural: ISOCategory.CATEGORY_1_DIRECT,
  scope2_location: ISOCategory.CATEGORY_2_ENERGY,
  scope2_market: ISOCategory.CATEGORY_2_ENERGY,
  steam_heat_purchase: ISOCategory.CATEGORY_2_ENERGY,
  cooling_purchase: ISOCategory.CATEGORY_2_ENERGY,
  upstream_transportation: ISOCategory.CATEGORY_3_TRANSPORT,
  downstream_transportation: ISOCategory.CATEGORY_3_TRANSPORT,
  business_travel: ISOCategory.CATEGORY_3_TRANSPORT,
  employee_commuting: ISOCategory.CATEGORY_3_TRANSPORT,
  purchased_goods_services: ISOCategory.CATEGORY_4_PRODUCTS_USED,
  capital_goods: ISOCategory.CATEGORY_4_PRODUCTS_USED,
  fuel_energy_activities: ISOCategory.CATEGORY_4_PRODUCTS_USED,
  waste_generated: ISOCategory.CATEGORY_4_PRODUCTS_USED,
  upstream_leased_assets: ISOCategory.CATEGORY_4_PRODUCTS_USED,
  processing_sold_products: ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
  use_of_sold_products: ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
  end_of_life_treatment: ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
  downstream_leased_assets: ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
  franchises: ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
  investments: ISOCategory.CATEGORY_6_OTHER,
};
