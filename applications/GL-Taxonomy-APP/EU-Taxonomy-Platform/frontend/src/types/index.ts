/**
 * GL-Taxonomy-APP TypeScript Type Definitions
 *
 * All interfaces and enums for the EU Taxonomy Alignment Platform,
 * matching the backend Pydantic models exactly.
 */

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

export enum EnvironmentalObjective {
  CLIMATE_MITIGATION = 'climate_mitigation',
  CLIMATE_ADAPTATION = 'climate_adaptation',
  WATER_MARINE = 'water_marine',
  CIRCULAR_ECONOMY = 'circular_economy',
  POLLUTION_PREVENTION = 'pollution_prevention',
  BIODIVERSITY = 'biodiversity',
}

export enum ActivityType {
  ENABLING = 'enabling',
  TRANSITIONAL = 'transitional',
  OWN_PERFORMANCE = 'own_performance',
}

export enum AlignmentStatus {
  NOT_STARTED = 'not_started',
  ELIGIBLE = 'eligible',
  SC_PASS = 'sc_pass',
  DNSH_PASS = 'dnsh_pass',
  MS_PASS = 'ms_pass',
  ALIGNED = 'aligned',
  NOT_ELIGIBLE = 'not_eligible',
  NOT_ALIGNED = 'not_aligned',
}

export enum DataQualityLevel {
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
  ESTIMATED = 'estimated',
}

export enum KPIType {
  TURNOVER = 'turnover',
  CAPEX = 'capex',
  OPEX = 'opex',
}

export enum ExposureType {
  GENERAL_LENDING = 'general_lending',
  SPECIALIZED_LENDING = 'specialized_lending',
  PROJECT_FINANCE = 'project_finance',
  EQUITY = 'equity',
  DEBT_SECURITIES = 'debt_securities',
  DERIVATIVES = 'derivatives',
  MORTGAGE = 'mortgage',
  AUTO_LOAN = 'auto_loan',
  INTERBANK = 'interbank',
  SOVEREIGN = 'sovereign',
}

export enum ReportTemplateType {
  ARTICLE_8_NFRD = 'article_8_nfrd',
  ARTICLE_8_CSRD = 'article_8_csrd',
  ARTICLE_8_SIMPLIFIED = 'article_8_simplified',
  EBA_PILLAR_3_GAR = 'eba_pillar_3_gar',
  EBA_PILLAR_3_BTAR = 'eba_pillar_3_btar',
  EBA_PILLAR_3_STOCK = 'eba_pillar_3_stock',
  EBA_PILLAR_3_FLOW = 'eba_pillar_3_flow',
  EBA_PILLAR_3_SUMMARY = 'eba_pillar_3_summary',
}

export enum GapSeverity {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
}

export enum GapCategory {
  SUBSTANTIAL_CONTRIBUTION = 'substantial_contribution',
  DNSH = 'dnsh',
  SAFEGUARDS = 'safeguards',
  DATA = 'data',
  REPORTING = 'reporting',
}

export enum SafeguardTopic {
  HUMAN_RIGHTS = 'human_rights',
  ANTI_CORRUPTION = 'anti_corruption',
  TAXATION = 'taxation',
  FAIR_COMPETITION = 'fair_competition',
}

export enum DNSHStatus {
  PASS = 'pass',
  FAIL = 'fail',
  NOT_APPLICABLE = 'not_applicable',
  PENDING = 'pending',
}

export enum ClimateRiskType {
  ACUTE_PHYSICAL = 'acute_physical',
  CHRONIC_PHYSICAL = 'chronic_physical',
  TRANSITION = 'transition',
}

export enum ExportFormat {
  PDF = 'pdf',
  EXCEL = 'excel',
  CSV = 'csv',
  XBRL = 'xbrl',
  JSON = 'json',
}

// ---------------------------------------------------------------------------
// Organization & Activities
// ---------------------------------------------------------------------------

export interface Organization {
  id: string;
  name: string;
  lei_code: string;
  sector: string;
  nace_codes: string[];
  country: string;
  reporting_standard: 'nfrd' | 'csrd';
  financial_institution: boolean;
  created_at: string;
  updated_at: string;
}

export interface NACEMapping {
  nace_code: string;
  nace_description: string;
  taxonomy_activity_id: string;
  taxonomy_activity_name: string;
  eligible_objectives: EnvironmentalObjective[];
  activity_type: ActivityType | null;
  sector: string;
  delegated_act_reference: string;
}

export interface EconomicActivity {
  id: string;
  organization_id: string;
  nace_code: string;
  activity_name: string;
  activity_description: string;
  taxonomy_activity_id: string;
  sector: string;
  eligible_objectives: EnvironmentalObjective[];
  activity_type: ActivityType | null;
  turnover_amount: number;
  capex_amount: number;
  opex_amount: number;
  currency: string;
  reporting_period: string;
  alignment_status: AlignmentStatus;
  created_at: string;
  updated_at: string;
}

export interface ActivityStatistics {
  total_activities: number;
  eligible_count: number;
  aligned_count: number;
  not_eligible_count: number;
  by_sector: Record<string, number>;
  by_objective: Record<string, number>;
  by_type: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Eligibility Screening
// ---------------------------------------------------------------------------

export interface EligibilityScreening {
  id: string;
  organization_id: string;
  screening_date: string;
  total_activities: number;
  eligible_count: number;
  not_eligible_count: number;
  eligibility_ratio: number;
  de_minimis_applied: boolean;
  de_minimis_threshold: number;
  results: ActivityEligibility[];
}

export interface ActivityEligibility {
  activity_id: string;
  activity_name: string;
  nace_code: string;
  is_eligible: boolean;
  eligible_objectives: EnvironmentalObjective[];
  delegated_act_reference: string;
  activity_type: ActivityType | null;
  turnover_share: number;
  capex_share: number;
  opex_share: number;
  rationale: string;
}

export interface BatchScreenRequest {
  organization_id: string;
  nace_codes: string[];
  include_enabling: boolean;
  include_transitional: boolean;
}

export interface ScreeningSummary {
  total_turnover: number;
  eligible_turnover: number;
  eligible_turnover_ratio: number;
  total_capex: number;
  eligible_capex: number;
  eligible_capex_ratio: number;
  total_opex: number;
  eligible_opex: number;
  eligible_opex_ratio: number;
  by_sector: Record<string, { count: number; turnover: number; capex: number; opex: number }>;
}

// ---------------------------------------------------------------------------
// Substantial Contribution Assessment
// ---------------------------------------------------------------------------

export interface SCAssessment {
  id: string;
  activity_id: string;
  objective: EnvironmentalObjective;
  assessment_date: string;
  meets_criteria: boolean;
  activity_type: ActivityType | null;
  tsc_evaluations: TSCEvaluation[];
  overall_score: number;
  evidence_items: EvidenceItem[];
  assessor: string;
  notes: string;
}

export interface TSCEvaluation {
  criterion_id: string;
  criterion_name: string;
  description: string;
  threshold_value: number | null;
  threshold_unit: string;
  actual_value: number | null;
  actual_unit: string;
  meets_threshold: boolean;
  evidence_ref: string;
  notes: string;
}

export interface ThresholdCheck {
  criterion_id: string;
  criterion_name: string;
  required_value: number;
  required_unit: string;
  actual_value: number;
  actual_unit: string;
  operator: 'gte' | 'lte' | 'eq' | 'between';
  passes: boolean;
  margin: number;
  margin_percent: number;
}

export interface SCCriteria {
  objective: EnvironmentalObjective;
  activity_id: string;
  criteria: TSCEvaluation[];
  delegated_act_version: string;
  effective_date: string;
}

export interface SCProfile {
  activity_id: string;
  activity_name: string;
  objectives_assessed: EnvironmentalObjective[];
  objectives_passed: EnvironmentalObjective[];
  activity_type: ActivityType | null;
  enabling_activities_supported: string[];
  transitional_plan: string | null;
}

export interface EvidenceItem {
  id: string;
  type: 'document' | 'certificate' | 'measurement' | 'audit_report' | 'third_party';
  name: string;
  description: string;
  file_url: string | null;
  uploaded_at: string;
  verified: boolean;
  verified_by: string | null;
}

// ---------------------------------------------------------------------------
// DNSH Assessment
// ---------------------------------------------------------------------------

export interface DNSHAssessment {
  id: string;
  activity_id: string;
  assessment_date: string;
  overall_pass: boolean;
  objective_assessments: ObjectiveDNSH[];
  climate_risk_assessment: ClimateRiskAssessment | null;
  evidence_items: EvidenceItem[];
  assessor: string;
  notes: string;
}

export interface ObjectiveDNSH {
  objective: EnvironmentalObjective;
  status: DNSHStatus;
  criteria: DNSHCriterion[];
  rationale: string;
  evidence_refs: string[];
}

export interface DNSHCriterion {
  criterion_id: string;
  criterion_name: string;
  description: string;
  requirement: string;
  met: boolean;
  evidence_ref: string;
  notes: string;
}

export interface ClimateRiskAssessment {
  id: string;
  activity_id: string;
  physical_risks: PhysicalRisk[];
  transition_risks: TransitionRisk[];
  adaptation_measures: AdaptationMeasure[];
  vulnerability_score: number;
  resilience_score: number;
  overall_pass: boolean;
}

export interface PhysicalRisk {
  hazard_type: string;
  category: 'acute' | 'chronic';
  likelihood: 'low' | 'medium' | 'high' | 'very_high';
  impact: 'low' | 'medium' | 'high' | 'very_high';
  time_horizon: 'short' | 'medium' | 'long';
  description: string;
  mitigation_measures: string[];
}

export interface TransitionRisk {
  risk_type: string;
  category: 'policy' | 'technology' | 'market' | 'reputation';
  likelihood: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  description: string;
}

export interface AdaptationMeasure {
  measure_id: string;
  description: string;
  implementation_status: 'planned' | 'in_progress' | 'implemented';
  effectiveness: 'low' | 'medium' | 'high';
  cost_estimate: number;
  timeline: string;
}

export interface DNSHMatrix {
  activity_id: string;
  activity_name: string;
  sc_objective: EnvironmentalObjective;
  dnsh_results: Record<EnvironmentalObjective, DNSHStatus>;
}

// ---------------------------------------------------------------------------
// Minimum Safeguards Assessment
// ---------------------------------------------------------------------------

export interface SafeguardAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  overall_pass: boolean;
  topic_assessments: TopicAssessment[];
  adverse_findings: AdverseFinding[];
  evidence_items: EvidenceItem[];
  assessor: string;
  next_review_date: string;
}

export interface TopicAssessment {
  topic: SafeguardTopic;
  procedural_score: number;
  outcome_score: number;
  overall_pass: boolean;
  frameworks_referenced: string[];
  checklist_items: ChecklistItem[];
  findings: string[];
  recommendations: string[];
}

export interface ChecklistItem {
  id: string;
  category: string;
  requirement: string;
  met: boolean;
  evidence_ref: string;
  notes: string;
}

export interface AdverseFinding {
  id: string;
  topic: SafeguardTopic;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  date_identified: string;
  remediation_status: 'open' | 'in_progress' | 'resolved';
  remediation_plan: string;
  resolution_date: string | null;
}

export interface DueDiligenceRecord {
  id: string;
  organization_id: string;
  topic: SafeguardTopic;
  framework: string;
  last_assessment_date: string;
  next_assessment_date: string;
  status: 'current' | 'due_soon' | 'overdue';
  responsible_party: string;
}

// ---------------------------------------------------------------------------
// KPI Calculation
// ---------------------------------------------------------------------------

export interface KPICalculation {
  id: string;
  organization_id: string;
  reporting_period: string;
  calculation_date: string;
  turnover_kpi: KPIDetail;
  capex_kpi: KPIDetail;
  opex_kpi: KPIDetail;
  double_counting_check: boolean;
  objective_breakdown: ObjectiveBreakdown[];
  notes: string;
}

export interface KPIDetail {
  total_amount: number;
  eligible_amount: number;
  eligible_ratio: number;
  aligned_amount: number;
  aligned_ratio: number;
  enabling_amount: number;
  enabling_ratio: number;
  transitional_amount: number;
  transitional_ratio: number;
  currency: string;
}

export interface ActivityFinancials {
  activity_id: string;
  activity_name: string;
  nace_code: string;
  turnover: number;
  capex: number;
  opex: number;
  alignment_status: AlignmentStatus;
  objective: EnvironmentalObjective | null;
  activity_type: ActivityType | null;
}

export interface CapExPlan {
  id: string;
  activity_id: string;
  plan_name: string;
  start_date: string;
  end_date: string;
  total_capex: number;
  annual_breakdown: { year: number; amount: number }[];
  target_objective: EnvironmentalObjective;
  current_progress: number;
  milestones: { name: string; date: string; completed: boolean }[];
  approved: boolean;
}

export interface ObjectiveBreakdown {
  objective: EnvironmentalObjective;
  turnover_amount: number;
  turnover_ratio: number;
  capex_amount: number;
  capex_ratio: number;
  opex_amount: number;
  opex_ratio: number;
  activity_count: number;
}

export interface KPISummary {
  organization_id: string;
  reporting_period: string;
  turnover_eligible_pct: number;
  turnover_aligned_pct: number;
  capex_eligible_pct: number;
  capex_aligned_pct: number;
  opex_eligible_pct: number;
  opex_aligned_pct: number;
  total_activities: number;
  aligned_activities: number;
  period_comparison: PeriodComparison | null;
}

export interface PeriodComparison {
  previous_period: string;
  turnover_change: number;
  capex_change: number;
  opex_change: number;
  alignment_change: number;
}

// ---------------------------------------------------------------------------
// GAR Calculation (Green Asset Ratio)
// ---------------------------------------------------------------------------

export interface GARCalculation {
  id: string;
  organization_id: string;
  reporting_date: string;
  gar_stock: GARDetail;
  gar_flow: GARDetail;
  exposure_breakdown: ExposureBreakdown[];
  sector_breakdown: SectorGAR[];
  total_assets: number;
  covered_assets: number;
  excluded_assets: number;
  notes: string;
}

export interface GARDetail {
  total_covered_assets: number;
  taxonomy_aligned_assets: number;
  gar_ratio: number;
  eligible_not_aligned: number;
  non_eligible: number;
  by_objective: Record<EnvironmentalObjective, number>;
  by_counterparty_type: Record<string, number>;
}

export interface ExposureBreakdown {
  exposure_type: ExposureType;
  total_amount: number;
  eligible_amount: number;
  aligned_amount: number;
  gar_ratio: number;
  counterparty_count: number;
}

export interface SectorGAR {
  sector: string;
  nace_code: string;
  total_exposure: number;
  aligned_exposure: number;
  gar_ratio: number;
  activity_count: number;
}

export interface BTARCalculation {
  id: string;
  organization_id: string;
  reporting_date: string;
  total_trading_book: number;
  taxonomy_aligned_trading: number;
  btar_ratio: number;
  by_instrument_type: Record<string, { total: number; aligned: number; ratio: number }>;
  notes: string;
}

export interface EBATemplateData {
  template_id: string;
  template_name: string;
  reporting_date: string;
  rows: EBATemplateRow[];
  totals: Record<string, number>;
  notes: string[];
}

export interface EBATemplateRow {
  row_id: string;
  label: string;
  total: number;
  taxonomy_eligible: number;
  taxonomy_aligned: number;
  of_which_enabling: number;
  of_which_transitional: number;
  non_eligible: number;
}

export interface MortgageAlignment {
  property_id: string;
  epc_rating: string;
  nzeb_threshold: boolean;
  top_15_percent: boolean;
  renovation_plan: boolean;
  aligned: boolean;
  rationale: string;
}

export interface AutoLoanAlignment {
  vehicle_id: string;
  co2_grams_per_km: number;
  threshold_2025: number;
  threshold_2026: number;
  aligned: boolean;
  rationale: string;
}

// ---------------------------------------------------------------------------
// Alignment Result
// ---------------------------------------------------------------------------

export interface AlignmentResult {
  id: string;
  activity_id: string;
  organization_id: string;
  assessment_date: string;
  status: AlignmentStatus;
  current_step: number;
  total_steps: number;
  eligibility_pass: boolean;
  sc_pass: boolean;
  sc_objective: EnvironmentalObjective | null;
  dnsh_pass: boolean;
  ms_pass: boolean;
  activity_type: ActivityType | null;
  data_quality: DataQualityLevel;
  notes: string;
}

export interface PortfolioAlignment {
  organization_id: string;
  assessment_date: string;
  total_activities: number;
  aligned_count: number;
  eligible_count: number;
  not_eligible_count: number;
  alignment_rate: number;
  by_objective: Record<EnvironmentalObjective, { aligned: number; eligible: number; total: number }>;
  by_sector: Record<string, { aligned: number; eligible: number; total: number }>;
  by_type: Record<ActivityType, number>;
  funnel: AlignmentFunnelData;
}

export interface AlignmentFunnelData {
  total: number;
  eligible: number;
  sc_pass: number;
  dnsh_pass: number;
  ms_pass: number;
  aligned: number;
}

export interface AlignmentProgress {
  activity_id: string;
  activity_name: string;
  current_step: 'eligibility' | 'sc' | 'dnsh' | 'safeguards' | 'aligned';
  step_results: {
    eligibility: boolean | null;
    sc: boolean | null;
    dnsh: boolean | null;
    safeguards: boolean | null;
  };
  last_updated: string;
}

export interface BatchAlignmentRequest {
  organization_id: string;
  activity_ids: string[];
  objective: EnvironmentalObjective;
  include_evidence: boolean;
}

// ---------------------------------------------------------------------------
// Disclosure & Reporting
// ---------------------------------------------------------------------------

export interface DisclosureReport {
  id: string;
  organization_id: string;
  report_type: ReportTemplateType;
  reporting_period: string;
  created_at: string;
  status: 'draft' | 'review' | 'approved' | 'published';
  qualitative_disclosures: QualitativeDisclosure[];
  data_tables: DataTable[];
  export_formats: ExportFormat[];
}

export interface QualitativeDisclosure {
  section_id: string;
  title: string;
  content: string;
  required: boolean;
  word_count: number;
  max_words: number;
}

export interface DataTable {
  table_id: string;
  title: string;
  headers: string[];
  rows: Record<string, string | number>[];
  footnotes: string[];
}

export interface Article8Data {
  reporting_period: string;
  turnover_kpi: KPIDetail;
  capex_kpi: KPIDetail;
  opex_kpi: KPIDetail;
  objective_breakdown: ObjectiveBreakdown[];
  contextual_information: string;
  accounting_policy: string;
  compliance_statement: string;
}

export interface ReportHistory {
  id: string;
  report_type: ReportTemplateType;
  reporting_period: string;
  created_at: string;
  created_by: string;
  status: string;
  file_url: string | null;
}

export interface ReportComparison {
  period_1: string;
  period_2: string;
  turnover_change: number;
  capex_change: number;
  opex_change: number;
  alignment_change: number;
  activity_changes: { activity: string; change: string; impact: number }[];
}

// ---------------------------------------------------------------------------
// Portfolio Management
// ---------------------------------------------------------------------------

export interface Portfolio {
  id: string;
  organization_id: string;
  name: string;
  description: string;
  total_value: number;
  currency: string;
  holdings_count: number;
  taxonomy_aligned_pct: number;
  created_at: string;
  updated_at: string;
}

export interface Holding {
  id: string;
  portfolio_id: string;
  counterparty_name: string;
  counterparty_lei: string;
  instrument_type: ExposureType;
  nominal_value: number;
  carrying_amount: number;
  nace_code: string;
  sector: string;
  taxonomy_eligible: boolean;
  taxonomy_aligned: boolean;
  alignment_source: 'reported' | 'estimated' | 'proxy';
  data_quality: DataQualityLevel;
}

export interface CounterpartySearchResult {
  lei: string;
  name: string;
  country: string;
  sector: string;
  nace_codes: string[];
  taxonomy_eligible_pct: number;
  taxonomy_aligned_pct: number;
  data_source: string;
}

// ---------------------------------------------------------------------------
// Data Quality
// ---------------------------------------------------------------------------

export interface DataQualityScore {
  organization_id: string;
  assessment_date: string;
  overall_score: number;
  overall_grade: 'A' | 'B' | 'C' | 'D' | 'F';
  dimensions: DimensionScore[];
  evidence_coverage: number;
  improvement_areas: string[];
}

export interface DimensionScore {
  dimension: 'completeness' | 'accuracy' | 'timeliness' | 'consistency' | 'verifiability';
  score: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
  issues: string[];
  recommendations: string[];
}

export interface EvidenceTracker {
  total_required: number;
  total_provided: number;
  coverage_pct: number;
  by_category: Record<string, { required: number; provided: number }>;
  missing_items: { criterion: string; evidence_type: string; priority: string }[];
}

export interface ImprovementPlan {
  id: string;
  organization_id: string;
  created_at: string;
  target_score: number;
  current_score: number;
  actions: ImprovementAction[];
}

export interface ImprovementAction {
  id: string;
  category: string;
  description: string;
  priority: 'high' | 'medium' | 'low';
  status: 'pending' | 'in_progress' | 'completed';
  responsible: string;
  due_date: string;
  impact_estimate: number;
}

// ---------------------------------------------------------------------------
// Gap Analysis
// ---------------------------------------------------------------------------

export interface GapAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  overall_readiness: number;
  total_gaps: number;
  critical_gaps: number;
  gap_items: GapItem[];
  action_plan: ActionPlanItem[];
}

export interface GapItem {
  id: string;
  category: GapCategory;
  area: string;
  description: string;
  severity: GapSeverity;
  current_state: string;
  required_state: string;
  remediation_effort: 'low' | 'medium' | 'high';
  estimated_cost: number;
  estimated_timeline_days: number;
}

export interface ActionPlanItem {
  id: string;
  gap_id: string;
  action: string;
  responsible: string;
  priority: 'high' | 'medium' | 'low';
  status: 'pending' | 'in_progress' | 'completed';
  start_date: string;
  due_date: string;
  completion_pct: number;
}

// ---------------------------------------------------------------------------
// Regulatory
// ---------------------------------------------------------------------------

export interface DelegatedActVersion {
  id: string;
  version: string;
  effective_date: string;
  objectives_covered: EnvironmentalObjective[];
  status: 'draft' | 'adopted' | 'in_force' | 'superseded';
  summary: string;
  key_changes: string[];
  document_url: string;
}

export interface RegulatoryUpdate {
  id: string;
  title: string;
  date: string;
  category: 'delegated_act' | 'faq' | 'guidance' | 'omnibus' | 'amendment';
  summary: string;
  impact: 'high' | 'medium' | 'low';
  affected_objectives: EnvironmentalObjective[];
  action_required: boolean;
  action_description: string;
}

export interface OmnibusImpact {
  id: string;
  change_description: string;
  affected_area: string;
  current_requirement: string;
  proposed_change: string;
  expected_effective_date: string;
  impact_on_organization: string;
  preparation_actions: string[];
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export interface TaxonomySettings {
  organization_id: string;
  reporting_periods: ReportingPeriod[];
  default_currency: string;
  delegated_act_version: string;
  de_minimis_threshold: number;
  custom_thresholds: Record<string, number>;
  mrv_mapping: MRVMapping[];
  notification_preferences: NotificationPreferences;
  auto_update_regulatory: boolean;
}

export interface ReportingPeriod {
  id: string;
  label: string;
  start_date: string;
  end_date: string;
  status: 'active' | 'closed' | 'draft';
  is_current: boolean;
}

export interface MRVMapping {
  mrv_agent: string;
  taxonomy_objective: EnvironmentalObjective;
  data_fields: string[];
  mapping_type: 'direct' | 'calculated' | 'proxy';
}

export interface NotificationPreferences {
  regulatory_updates: boolean;
  assessment_reminders: boolean;
  data_quality_alerts: boolean;
  reporting_deadlines: boolean;
  email_frequency: 'immediate' | 'daily' | 'weekly';
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

export interface DashboardOverview {
  organization_id: string;
  reporting_period: string;
  kpi_summary: KPISummary;
  alignment_summary: AlignmentSummaryCard;
  sector_breakdown: SectorBreakdownItem[];
  trend_data: TrendDataPoint[];
  objective_radar: ObjectiveRadarPoint[];
  recent_activities: RecentActivity[];
}

export interface AlignmentSummaryCard {
  total_activities: number;
  eligible: number;
  aligned: number;
  not_eligible: number;
  alignment_rate: number;
  eligible_rate: number;
  change_from_previous: number;
}

export interface SectorBreakdownItem {
  sector: string;
  activity_count: number;
  aligned_count: number;
  turnover: number;
  aligned_turnover: number;
  alignment_rate: number;
}

export interface TrendDataPoint {
  period: string;
  turnover_eligible: number;
  turnover_aligned: number;
  capex_eligible: number;
  capex_aligned: number;
  opex_eligible: number;
  opex_aligned: number;
}

export interface ObjectiveRadarPoint {
  objective: EnvironmentalObjective;
  label: string;
  eligible_count: number;
  aligned_count: number;
  sc_pass_rate: number;
  dnsh_pass_rate: number;
}

export interface RecentActivity {
  id: string;
  activity_name: string;
  action: string;
  timestamp: string;
  user: string;
  status: AlignmentStatus;
}

// ---------------------------------------------------------------------------
// API Request/Response Types
// ---------------------------------------------------------------------------

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  per_page: number;
  total_pages: number;
}

export interface ApiError {
  detail: string;
  status_code: number;
  error_type: string;
}

export interface ActivitySearchParams {
  query?: string;
  sector?: string;
  objective?: EnvironmentalObjective;
  activity_type?: ActivityType;
  alignment_status?: AlignmentStatus;
  page?: number;
  per_page?: number;
}

export interface KPIDashboardParams {
  organization_id: string;
  reporting_period: string;
  include_comparison?: boolean;
}

export interface GARParams {
  organization_id: string;
  reporting_date: string;
  include_btar?: boolean;
  include_sector_breakdown?: boolean;
}

export interface ReportExportParams {
  report_id: string;
  format: ExportFormat;
  include_annexes?: boolean;
  include_methodology?: boolean;
}

export interface PortfolioUploadParams {
  portfolio_id: string;
  file_format: 'csv' | 'excel';
  mapping: Record<string, string>;
}
