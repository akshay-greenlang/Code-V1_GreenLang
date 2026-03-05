/**
 * GL-SBTi-APP Type Definitions
 *
 * Comprehensive TypeScript interfaces for the SBTi Target Validation & Progress
 * Tracking Platform. Covers target setting, pathway calculation, validation,
 * progress monitoring, temperature scoring, and financial institution portfolios.
 */

// ─── Enums as String Unions ───────────────────────────────────────────────────

export type TargetType = 'absolute' | 'intensity';

export type TargetMethod =
  | 'cross_sector_aca'
  | 'sector_specific_sda'
  | 'portfolio_coverage'
  | 'sectoral_decarbonization'
  | 'temperature_rating'
  | 'engagement_threshold';

export type TargetScope =
  | 'scope_1'
  | 'scope_2'
  | 'scope_1_2'
  | 'scope_3'
  | 'scope_1_2_3';

export type TargetTimeframe = 'near_term' | 'long_term';

export type TargetStatus =
  | 'draft'
  | 'submitted'
  | 'under_review'
  | 'approved'
  | 'validated'
  | 'committed'
  | 'rejected'
  | 'expired'
  | 'withdrawn';

export type PathwayAlignment =
  | '1.5C'
  | 'well_below_2C'
  | '2C'
  | 'above_2C'
  | 'not_aligned';

export type SBTiSector =
  | 'power_generation'
  | 'oil_and_gas'
  | 'transport'
  | 'buildings'
  | 'cement'
  | 'steel'
  | 'aluminum'
  | 'chemicals'
  | 'pulp_and_paper'
  | 'aviation'
  | 'shipping'
  | 'apparel_footwear'
  | 'forest_land_agriculture'
  | 'financial_institutions'
  | 'ict'
  | 'general';

export type RAGStatus = 'on_track' | 'at_risk' | 'off_track';

export type IntensityMetric =
  | 'per_revenue'
  | 'per_employee'
  | 'per_unit_product'
  | 'per_square_meter'
  | 'per_passenger_km'
  | 'per_tonne_km'
  | 'per_mwh'
  | 'per_tonne_cement'
  | 'per_tonne_steel'
  | 'custom';

export type RecalculationTrigger =
  | 'structural_change'
  | 'methodology_change'
  | 'discovery_of_error'
  | 'acquisition_divestiture'
  | 'outsourcing_insourcing'
  | 'base_year_update'
  | 'organic_growth';

export type FLAGCommodity =
  | 'beef'
  | 'dairy'
  | 'poultry'
  | 'pork'
  | 'rice'
  | 'wheat'
  | 'maize'
  | 'soy'
  | 'palm_oil'
  | 'timber'
  | 'other_crops';

export type FIAssetClass =
  | 'listed_equity'
  | 'corporate_bonds'
  | 'business_loans'
  | 'project_finance'
  | 'commercial_real_estate'
  | 'mortgages'
  | 'motor_vehicle_loans'
  | 'sovereign_bonds';

export type PCAFDataQuality = 1 | 2 | 3 | 4 | 5;

export type ReviewOutcome =
  | 'targets_revalidated'
  | 'targets_updated'
  | 'targets_withdrawn'
  | 'recalculation_required'
  | 'in_progress';

export type ExportFormat = 'pdf' | 'xlsx' | 'json' | 'csv' | 'docx';

export type ComplianceFramework =
  | 'sbti'
  | 'ghg_protocol'
  | 'iso_14064'
  | 'cdp'
  | 'tcfd'
  | 'csrd'
  | 'sec_climate'
  | 'issb';

export type Currency = 'USD' | 'EUR' | 'GBP' | 'JPY' | 'AUD' | 'CAD' | 'CHF';

// ─── Organization ─────────────────────────────────────────────────────────────

export interface Organization {
  id: string;
  name: string;
  industry_sector: SBTiSector;
  sub_sector: string;
  isic_code: string;
  country: string;
  region: string;
  employee_count: number;
  annual_revenue: number;
  revenue_currency: Currency;
  fiscal_year_end_month: number;
  sbti_commitment_date: string | null;
  sbti_validation_date: string | null;
  sbti_status: TargetStatus;
  is_sme: boolean;
  is_financial_institution: boolean;
  flag_applicable: boolean;
  created_at: string;
  updated_at: string;
}

// ─── Emissions Inventory ──────────────────────────────────────────────────────

export interface EmissionsInventory {
  id: string;
  organization_id: string;
  reporting_year: number;
  base_year: boolean;
  scope_1: number;
  scope_2_location: number;
  scope_2_market: number;
  scope_3_total: number;
  scope_3_categories: Scope3CategoryEmission[];
  total_scope_1_2: number;
  total_scope_1_2_3: number;
  flag_emissions: number | null;
  non_flag_emissions: number | null;
  biogenic_emissions: number;
  carbon_removals: number;
  data_quality_score: number;
  methodology: string;
  verification_status: 'unverified' | 'limited_assurance' | 'reasonable_assurance';
  verified_by: string | null;
  created_at: string;
  updated_at: string;
}

export interface Scope3CategoryEmission {
  category_number: number;
  category_name: string;
  emissions_tco2e: number;
  percentage_of_scope3: number;
  data_quality: PCAFDataQuality;
  methodology: string;
  included_in_target: boolean;
}

// ─── Targets ──────────────────────────────────────────────────────────────────

export interface Target {
  id: string;
  organization_id: string;
  name: string;
  description: string;
  target_type: TargetType;
  target_method: TargetMethod;
  target_scope: TargetScope;
  target_timeframe: TargetTimeframe;
  status: TargetStatus;
  base_year: number;
  base_year_emissions: number;
  target_year: number;
  target_reduction_pct: number;
  target_emissions: number;
  current_year_emissions: number;
  progress_pct: number;
  annual_reduction_rate: number;
  pathway_alignment: PathwayAlignment;
  intensity_metric: IntensityMetric | null;
  intensity_base_value: number | null;
  intensity_target_value: number | null;
  scope_coverage_pct: number;
  scope_3_categories_included: number[];
  boundary_description: string;
  exclusions: string[];
  sbti_criteria_met: boolean;
  submission_date: string | null;
  validation_date: string | null;
  next_review_date: string;
  scopes: TargetScopeDetail[];
  created_at: string;
  updated_at: string;
}

export interface TargetScopeDetail {
  scope: TargetScope;
  base_year_emissions: number;
  target_year_emissions: number;
  current_emissions: number;
  coverage_pct: number;
  reduction_pct: number;
}

// ─── Pathways ─────────────────────────────────────────────────────────────────

export interface Pathway {
  id: string;
  target_id: string;
  method: TargetMethod;
  alignment: PathwayAlignment;
  sector: SBTiSector;
  base_year: number;
  target_year: number;
  base_emissions: number;
  target_emissions: number;
  annual_reduction_rate: number;
  milestones: PathwayMilestone[];
  parameters: PathwayParameters;
  created_at: string;
  updated_at: string;
}

export interface PathwayMilestone {
  year: number;
  expected_emissions: number;
  actual_emissions: number | null;
  reduction_from_base_pct: number;
  on_track: boolean | null;
  cumulative_budget: number;
  notes: string;
}

export interface PathwayParameters {
  contraction_rate: number;
  convergence_year: number | null;
  sector_intensity_target: number | null;
  global_budget_share: number | null;
  temperature_alignment: number;
  scenario_source: string;
  carbon_budget_gt: number | null;
}

export interface PathwayComparison {
  pathway_id: string;
  pathway_name: string;
  method: TargetMethod;
  alignment: PathwayAlignment;
  annual_rate: number;
  milestones: { year: number; emissions: number }[];
}

// ─── Validation ───────────────────────────────────────────────────────────────

export interface ValidationResult {
  id: string;
  target_id: string;
  validation_date: string;
  overall_pass: boolean;
  readiness_score: number;
  criteria_checks: CriterionCheck[];
  summary: ValidationSummary;
  recommendations: string[];
  issues: ValidationIssue[];
  created_at: string;
}

export interface CriterionCheck {
  criterion_id: string;
  criterion_code: string;
  criterion_name: string;
  description: string;
  category: 'boundary' | 'timeframe' | 'ambition' | 'scope_coverage' | 'reporting' | 'methodology' | 'recalculation' | 'general';
  status: 'pass' | 'fail' | 'warning' | 'not_applicable';
  details: string;
  evidence: string;
  remediation: string | null;
}

export interface ValidationSummary {
  total_criteria: number;
  passed: number;
  failed: number;
  warnings: number;
  not_applicable: number;
  scope_1_2_coverage: number;
  scope_3_coverage: number;
  ambition_level: PathwayAlignment;
  near_term_compliant: boolean;
  long_term_compliant: boolean;
}

export interface ValidationIssue {
  id: string;
  criterion_code: string;
  severity: 'critical' | 'major' | 'minor';
  title: string;
  description: string;
  resolution_guidance: string;
  estimated_effort: 'low' | 'medium' | 'high';
  status: 'open' | 'in_progress' | 'resolved';
}

// ─── Scope 3 Screening ───────────────────────────────────────────────────────

export interface Scope3Screening {
  id: string;
  organization_id: string;
  screening_date: string;
  total_scope_1_2_emissions: number;
  total_scope_3_emissions: number;
  scope_3_as_pct_of_total: number;
  trigger_threshold_pct: number;
  trigger_exceeded: boolean;
  category_breakdown: CategoryBreakdown[];
  trigger_assessment: TriggerAssessment;
  recommended_categories: number[];
  coverage_pct: number;
  created_at: string;
}

export interface CategoryBreakdown {
  category_number: number;
  category_name: string;
  emissions_tco2e: number;
  percentage_of_scope3: number;
  percentage_of_total: number;
  cumulative_pct: number;
  included_in_target: boolean;
  data_quality: PCAFDataQuality;
  significance: 'high' | 'medium' | 'low';
  screening_method: string;
}

export interface TriggerAssessment {
  scope_3_exceeds_40_pct: boolean;
  scope_3_pct_of_total: number;
  categories_over_threshold: number[];
  minimum_coverage_required_pct: number;
  current_coverage_pct: number;
  coverage_sufficient: boolean;
  two_thirds_coverage_met: boolean;
}

// ─── FLAG Assessment ──────────────────────────────────────────────────────────

export interface FLAGAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  total_emissions: number;
  flag_emissions: number;
  non_flag_emissions: number;
  flag_pct_of_total: number;
  trigger_threshold_pct: number;
  flag_target_required: boolean;
  commodities: CommodityData[];
  deforestation_commitment: boolean;
  deforestation_commitment_date: string | null;
  deforestation_zero_by_year: number | null;
  flag_trigger_result: FLAGTriggerResult;
  created_at: string;
}

export interface CommodityData {
  commodity: FLAGCommodity;
  emissions_tco2e: number;
  percentage_of_flag: number;
  land_use_change_emissions: number;
  production_emissions: number;
  pathway_available: boolean;
  sector_pathway_id: string | null;
}

export interface FLAGTriggerResult {
  flag_exceeds_20_pct: boolean;
  flag_pct_of_total: number;
  separate_flag_target_required: boolean;
  recommended_flag_target_year: number;
  recommended_reduction_pct: number;
  deforestation_free_required: boolean;
}

// ─── Sector Pathways ──────────────────────────────────────────────────────────

export interface SectorPathway {
  id: string;
  sector: SBTiSector;
  sub_sector: string;
  pathway_name: string;
  temperature_alignment: PathwayAlignment;
  scenario_source: string;
  base_year: number;
  target_year: number;
  annual_reduction_rate: number;
  intensity_metric: IntensityMetric;
  intensity_unit: string;
  milestones: { year: number; intensity_value: number; absolute_reduction_pct: number }[];
  applicability_criteria: string[];
  last_updated: string;
}

export interface SectorBenchmark {
  sector: SBTiSector;
  metric_name: string;
  metric_unit: string;
  organization_value: number;
  sector_average: number;
  best_in_class: number;
  pathway_1_5c: number;
  pathway_wb2c: number;
  percentile_rank: number;
}

// ─── Progress Tracking ────────────────────────────────────────────────────────

export interface ProgressRecord {
  id: string;
  target_id: string;
  reporting_year: number;
  actual_emissions: number;
  expected_emissions: number;
  base_year_emissions: number;
  absolute_reduction_pct: number;
  intensity_value: number | null;
  rag_status: RAGStatus;
  variance_pct: number;
  scope_1_emissions: number;
  scope_2_emissions: number;
  scope_3_emissions: number;
  contributing_factors: string[];
  corrective_actions: string[];
  data_quality_score: number;
  verified: boolean;
  created_at: string;
}

export interface ProgressSummary {
  target_id: string;
  target_name: string;
  target_type: TargetType;
  base_year: number;
  target_year: number;
  years_remaining: number;
  base_emissions: number;
  target_emissions: number;
  current_emissions: number;
  progress_pct: number;
  annual_reduction_required: number;
  actual_annual_reduction: number;
  rag_status: RAGStatus;
  on_track: boolean;
  projected_target_year_emissions: number;
  gap_to_target: number;
  records: ProgressRecord[];
}

export interface VarianceAnalysis {
  target_id: string;
  reporting_year: number;
  expected_emissions: number;
  actual_emissions: number;
  variance_absolute: number;
  variance_pct: number;
  scope_variances: { scope: string; expected: number; actual: number; variance: number }[];
  contributing_factors: { factor: string; impact_tco2e: number; direction: 'increase' | 'decrease' }[];
  year_over_year_change: number;
  cumulative_variance: number;
}

// ─── Temperature Scoring ──────────────────────────────────────────────────────

export interface TemperatureScore {
  id: string;
  organization_id: string;
  target_id: string | null;
  calculation_date: string;
  scope: TargetScope;
  temperature_score: number;
  alignment: PathwayAlignment;
  methodology: string;
  scenario_source: string;
  time_horizon: TargetTimeframe;
  confidence_low: number;
  confidence_high: number;
  peer_average: number;
  sector_average: number;
  created_at: string;
}

export interface PortfolioTemperature {
  portfolio_id: string;
  calculation_date: string;
  weighted_temperature: number;
  alignment: PathwayAlignment;
  scope_scores: { scope: TargetScope; temperature: number }[];
  contributions: { company_id: string; company_name: string; weight_pct: number; temperature: number; contribution: number }[];
  methodology: 'WATS' | 'TETS' | 'MOTS' | 'EOTS' | 'AOTS' | 'ROTS';
  coverage_pct: number;
}

export interface TemperatureTimeSeries {
  year: number;
  temperature_score: number;
  target_temperature: number;
  peer_average: number;
}

export interface PeerTemperatureRanking {
  rank: number;
  company_name: string;
  sector: SBTiSector;
  temperature_score: number;
  alignment: PathwayAlignment;
  sbti_status: TargetStatus;
  is_current_org: boolean;
}

// ─── Recalculation ────────────────────────────────────────────────────────────

export interface Recalculation {
  id: string;
  target_id: string;
  trigger: RecalculationTrigger;
  trigger_date: string;
  description: string;
  impact_on_base_year_pct: number;
  original_base_year_emissions: number;
  recalculated_base_year_emissions: number;
  threshold_exceeded: boolean;
  threshold_pct: number;
  status: 'identified' | 'under_review' | 'approved' | 'completed' | 'rejected';
  approved_by: string | null;
  completed_date: string | null;
  audit_trail: RecalculationAuditEntry[];
  created_at: string;
  updated_at: string;
}

export interface RecalculationAuditEntry {
  timestamp: string;
  action: string;
  user: string;
  details: string;
  old_value: number | null;
  new_value: number | null;
}

export interface ThresholdCheck {
  trigger: RecalculationTrigger;
  change_pct: number;
  threshold_pct: number;
  exceeds_threshold: boolean;
  recommendation: string;
  affected_targets: string[];
}

// ─── Five-Year Review ─────────────────────────────────────────────────────────

export interface FiveYearReview {
  id: string;
  organization_id: string;
  review_cycle: number;
  review_start_date: string;
  review_due_date: string;
  review_completed_date: string | null;
  days_remaining: number;
  status: ReviewStatus;
  outcome: ReviewOutcome | null;
  scope_1_2_progress: number;
  scope_3_progress: number;
  pathway_alignment_check: PathwayAlignment;
  methodology_current: boolean;
  data_quality_sufficient: boolean;
  recalculation_needed: boolean;
  checklist_items: ReviewChecklistItem[];
  reviewer_notes: string;
  created_at: string;
  updated_at: string;
}

export type ReviewStatus =
  | 'upcoming'
  | 'in_progress'
  | 'review'
  | 'completed'
  | 'overdue';

export interface ReviewChecklistItem {
  id: string;
  category: string;
  requirement: string;
  status: 'not_started' | 'in_progress' | 'completed';
  notes: string;
  due_date: string;
  assigned_to: string;
}

// ─── Financial Institutions ───────────────────────────────────────────────────

export interface FIPortfolio {
  id: string;
  organization_id: string;
  portfolio_name: string;
  asset_class: FIAssetClass;
  total_value: number;
  currency: Currency;
  holdings_count: number;
  financed_emissions_tco2e: number;
  waci: number;
  coverage_pct: number;
  sbti_aligned_pct: number;
  data_quality_avg: PCAFDataQuality;
  temperature_score: number;
  target_coverage_2030: number;
  target_coverage_2040: number;
  holdings: PortfolioHolding[];
  engagement_records: EngagementRecord[];
  created_at: string;
  updated_at: string;
}

export interface PortfolioHolding {
  id: string;
  portfolio_id: string;
  company_name: string;
  company_id: string;
  sector: SBTiSector;
  asset_class: FIAssetClass;
  investment_value: number;
  ownership_pct: number;
  financed_emissions_tco2e: number;
  attribution_factor: number;
  data_quality: PCAFDataQuality;
  sbti_status: TargetStatus;
  sbti_committed: boolean;
  temperature_score: number;
  engagement_status: 'not_started' | 'engaged' | 'committed' | 'target_set' | 'validated';
}

export interface EngagementRecord {
  id: string;
  portfolio_id: string;
  company_id: string;
  company_name: string;
  engagement_date: string;
  engagement_type: 'letter' | 'meeting' | 'shareholder_resolution' | 'collaborative' | 'escalation';
  outcome: string;
  sbti_commitment_obtained: boolean;
  follow_up_date: string | null;
  notes: string;
}

// ─── Framework Mapping ────────────────────────────────────────────────────────

export interface FrameworkMapping {
  id: string;
  sbti_requirement: string;
  sbti_code: string;
  framework: ComplianceFramework;
  framework_requirement: string;
  framework_code: string;
  mapping_type: 'direct' | 'partial' | 'enhanced' | 'no_equivalent';
  gap_description: string | null;
  organization_status: 'met' | 'partial' | 'not_met' | 'not_applicable';
}

export interface AlignmentItem {
  framework: ComplianceFramework;
  total_requirements: number;
  met: number;
  partial: number;
  not_met: number;
  not_applicable: number;
  alignment_pct: number;
}

// ─── Reports ──────────────────────────────────────────────────────────────────

export interface Report {
  id: string;
  organization_id: string;
  report_type: 'target_summary' | 'progress_report' | 'validation_report' | 'submission_package' | 'annual_disclosure';
  title: string;
  description: string;
  reporting_year: number;
  generated_date: string;
  format: ExportFormat;
  file_url: string | null;
  includes_targets: string[];
  status: 'generating' | 'ready' | 'exported' | 'submitted';
  created_at: string;
}

export interface SubmissionForm {
  organization_id: string;
  target_ids: string[];
  contact_name: string;
  contact_email: string;
  contact_role: string;
  methodology_description: string;
  data_sources: string[];
  verification_statement: boolean;
  board_approval_date: string;
  additional_notes: string;
}

// ─── Gap Assessment ───────────────────────────────────────────────────────────

export interface GapAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  overall_readiness_pct: number;
  category_scores: { category: string; score: number; max_score: number; pct: number }[];
  gaps: GapItem[];
  action_plan: GapAction[];
  estimated_readiness_date: string;
  created_at: string;
}

export interface GapItem {
  id: string;
  category: string;
  requirement: string;
  current_state: string;
  desired_state: string;
  severity: 'critical' | 'major' | 'moderate' | 'minor';
  priority: number;
  effort_estimate: 'low' | 'medium' | 'high';
  responsible_team: string;
  target_completion: string;
}

export interface GapAction {
  id: string;
  gap_id: string;
  title: string;
  description: string;
  responsible: string;
  start_date: string;
  due_date: string;
  status: 'not_started' | 'in_progress' | 'completed' | 'blocked';
  progress_pct: number;
  dependencies: string[];
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export interface DashboardSummary {
  readiness_score: number;
  targets_summary: {
    total_targets: number;
    validated: number;
    submitted: number;
    draft: number;
    near_term_count: number;
    long_term_count: number;
  };
  pathway_alignment: PathwayAlignment;
  temperature_score: number;
  temperature_trend: { year: number; score: number }[];
  progress_overview: {
    target_id: string;
    target_name: string;
    rag_status: RAGStatus;
    progress_pct: number;
  }[];
  scope_3_trigger: boolean;
  flag_trigger: boolean;
  next_review_date: string;
  days_to_review: number;
  upcoming_milestones: { date: string; description: string; target_name: string }[];
  emissions_trend: { year: number; scope_1: number; scope_2: number; scope_3: number; total: number }[];
  recent_activity: ActivityItem[];
}

export interface ActivityItem {
  id: string;
  type: 'target_update' | 'validation_run' | 'progress_entry' | 'recalculation' | 'review_action' | 'submission';
  description: string;
  user: string;
  timestamp: string;
}

// ─── Settings ─────────────────────────────────────────────────────────────────

export interface OrganizationSettings {
  id: string;
  organization_name: string;
  industry_sector: SBTiSector;
  sub_sector: string;
  isic_code: string;
  country: string;
  reporting_currency: Currency;
  fiscal_year_end_month: number;
  base_year: number;
  sbti_commitment_letter_date: string | null;
  near_term_target_year: number;
  long_term_target_year: number;
  scope_3_threshold_pct: number;
  flag_threshold_pct: number;
  recalculation_threshold_pct: number;
  review_cycle_years: number;
  auto_validate: boolean;
  notification_preferences: {
    email_alerts: boolean;
    milestone_reminders: boolean;
    review_deadline_alerts: boolean;
    recalculation_triggers: boolean;
  };
  created_at: string;
  updated_at: string;
}

// ─── API Response Wrappers ────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface ApiError {
  code: string;
  message: string;
  details: Record<string, unknown>;
}

export interface ApiResponse<T> {
  data: T;
  meta?: {
    request_id: string;
    timestamp: string;
    processing_time_ms: number;
  };
}
