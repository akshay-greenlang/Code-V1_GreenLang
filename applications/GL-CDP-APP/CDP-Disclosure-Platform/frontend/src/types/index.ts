/**
 * GL-CDP-APP v1.0 - TypeScript Type Definitions
 *
 * Central type definitions for the CDP Climate Change Disclosure Platform.
 * Covers all 13 CDP modules, 17 scoring categories, questionnaire engine,
 * response management, scoring simulation, gap analysis, benchmarking,
 * supply chain, transition planning, verification, and dashboard metrics.
 *
 * All emissions are in metric tonnes CO2e unless otherwise noted.
 * Timestamps are UTC ISO 8601 strings.
 */

/* ------------------------------------------------------------------ */
/*  Enums                                                              */
/* ------------------------------------------------------------------ */

export enum CDPModule {
  M0_INTRODUCTION = 'M0',
  M1_GOVERNANCE = 'M1',
  M2_POLICIES = 'M2',
  M3_RISKS = 'M3',
  M4_STRATEGY = 'M4',
  M5_TRANSITION = 'M5',
  M6_IMPLEMENTATION = 'M6',
  M7_CLIMATE_PERFORMANCE = 'M7',
  M8_FORESTS = 'M8',
  M9_WATER = 'M9',
  M10_SUPPLY_CHAIN = 'M10',
  M11_ADDITIONAL = 'M11',
  M12_FINANCIAL_SERVICES = 'M12',
  M13_SIGN_OFF = 'M13',
}

export enum ScoringLevel {
  A = 'A',
  A_MINUS = 'A-',
  B = 'B',
  B_MINUS = 'B-',
  C = 'C',
  C_MINUS = 'C-',
  D = 'D',
  D_MINUS = 'D-',
}

export enum ScoringBand {
  LEADERSHIP = 'Leadership',
  MANAGEMENT = 'Management',
  AWARENESS = 'Awareness',
  DISCLOSURE = 'Disclosure',
}

export enum ScoringCategory {
  GOVERNANCE = 'governance',
  RISK_MANAGEMENT_PROCESSES = 'risk_management_processes',
  RISK_DISCLOSURE = 'risk_disclosure',
  OPPORTUNITY_DISCLOSURE = 'opportunity_disclosure',
  BUSINESS_STRATEGY = 'business_strategy',
  SCENARIO_ANALYSIS = 'scenario_analysis',
  TARGETS = 'targets',
  EMISSIONS_REDUCTION_INITIATIVES = 'emissions_reduction_initiatives',
  SCOPE_1_2_EMISSIONS = 'scope_1_2_emissions',
  SCOPE_3_EMISSIONS = 'scope_3_emissions',
  ENERGY = 'energy',
  CARBON_PRICING = 'carbon_pricing',
  VALUE_CHAIN_ENGAGEMENT = 'value_chain_engagement',
  PUBLIC_POLICY_ENGAGEMENT = 'public_policy_engagement',
  TRANSITION_PLAN = 'transition_plan',
  PORTFOLIO_CLIMATE_PERFORMANCE = 'portfolio_climate_performance',
  FINANCIAL_IMPACT_ASSESSMENT = 'financial_impact_assessment',
}

export enum QuestionType {
  TEXT = 'text',
  NUMERIC = 'numeric',
  PERCENTAGE = 'percentage',
  TABLE = 'table',
  MULTI_SELECT = 'multi_select',
  SINGLE_SELECT = 'single_select',
  YES_NO = 'yes_no',
}

export enum ResponseStatus {
  NOT_STARTED = 'not_started',
  DRAFT = 'draft',
  IN_REVIEW = 'in_review',
  APPROVED = 'approved',
  SUBMITTED = 'submitted',
}

export enum GapSeverity {
  CRITICAL = 'critical',
  HIGH = 'high',
  MEDIUM = 'medium',
  LOW = 'low',
}

export enum GapEffort {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
}

export enum VerificationLevel {
  LIMITED = 'limited',
  REASONABLE = 'reasonable',
  NOT_VERIFIED = 'not_verified',
}

export enum SupplierStatus {
  INVITED = 'invited',
  IN_PROGRESS = 'in_progress',
  SUBMITTED = 'submitted',
  SCORED = 'scored',
  DECLINED = 'declined',
}

export enum TransitionMilestoneStatus {
  NOT_STARTED = 'not_started',
  IN_PROGRESS = 'in_progress',
  COMPLETED = 'completed',
  DELAYED = 'delayed',
}

export enum ReportFormat {
  PDF = 'pdf',
  EXCEL = 'excel',
  XML = 'xml',
  JSON = 'json',
}

export enum ReportStatus {
  DRAFT = 'draft',
  REVIEW = 'review',
  FINAL = 'final',
  SUBMITTED = 'submitted',
}

export enum AlertSeverity {
  INFO = 'info',
  WARNING = 'warning',
  ERROR = 'error',
}

/* ------------------------------------------------------------------ */
/*  Organization & Questionnaire Models                                */
/* ------------------------------------------------------------------ */

export interface Organization {
  id: string;
  name: string;
  industry: string;
  gics_sector: string;
  gics_industry_group: string;
  country: string;
  region: string;
  employees: number | null;
  revenue_usd: number | null;
  reporting_year: number;
  is_financial_services: boolean;
  created_at: string;
  updated_at: string;
}

export interface Questionnaire {
  id: string;
  org_id: string;
  reporting_year: number;
  questionnaire_version: string;
  status: ResponseStatus;
  modules: Module[];
  total_questions: number;
  answered_questions: number;
  reviewed_questions: number;
  approved_questions: number;
  completion_pct: number;
  submission_deadline: string;
  created_at: string;
  updated_at: string;
}

export interface Module {
  id: string;
  questionnaire_id: string;
  module_code: CDPModule;
  name: string;
  description: string;
  question_count: number;
  answered_count: number;
  reviewed_count: number;
  approved_count: number;
  completion_pct: number;
  is_applicable: boolean;
  is_sector_specific: boolean;
  display_order: number;
}

export interface Question {
  id: string;
  module_id: string;
  module_code: CDPModule;
  question_number: string;
  question_text: string;
  guidance_text: string;
  question_type: QuestionType;
  scoring_category: ScoringCategory | null;
  scoring_weight: number;
  is_required: boolean;
  is_conditional: boolean;
  depends_on_question_id: string | null;
  depends_on_answer: string | null;
  options: string[] | null;
  table_columns: TableColumn[] | null;
  example_response: string | null;
  previous_year_response: string | null;
  auto_populated_data: AutoPopulatedField[] | null;
  assigned_to: string | null;
  display_order: number;
}

export interface TableColumn {
  key: string;
  label: string;
  type: 'text' | 'numeric' | 'percentage' | 'select';
  options?: string[];
  required: boolean;
}

export interface AutoPopulatedField {
  field_name: string;
  value: string | number;
  source_agent: string;
  source_description: string;
  confidence: number;
  last_updated: string;
}

/* ------------------------------------------------------------------ */
/*  Response Models                                                    */
/* ------------------------------------------------------------------ */

export interface Response {
  id: string;
  question_id: string;
  questionnaire_id: string;
  response_text: string;
  response_data: Record<string, unknown> | null;
  table_data: Record<string, unknown>[] | null;
  status: ResponseStatus;
  assigned_to: string | null;
  reviewer: string | null;
  review_comments: string | null;
  evidence_ids: string[];
  version: number;
  is_auto_populated: boolean;
  manual_override: boolean;
  override_justification: string | null;
  created_at: string;
  updated_at: string;
}

export interface ResponseVersion {
  id: string;
  response_id: string;
  version: number;
  response_text: string;
  response_data: Record<string, unknown> | null;
  status: ResponseStatus;
  edited_by: string;
  edited_at: string;
  change_summary: string;
}

export interface Evidence {
  id: string;
  response_id: string;
  file_name: string;
  file_type: string;
  file_size_bytes: number;
  description: string;
  uploaded_by: string;
  uploaded_at: string;
  url: string;
}

export interface ReviewComment {
  id: string;
  response_id: string;
  author: string;
  comment: string;
  created_at: string;
}

/* ------------------------------------------------------------------ */
/*  Scoring Models                                                     */
/* ------------------------------------------------------------------ */

export interface ScoringResult {
  id: string;
  questionnaire_id: string;
  overall_score: number;
  scoring_level: ScoringLevel;
  scoring_band: ScoringBand;
  category_scores: CategoryScore[];
  a_level_eligible: boolean;
  a_level_requirements: ARequirement[];
  confidence: number;
  previous_score: number | null;
  previous_level: ScoringLevel | null;
  simulated_at: string;
}

export interface CategoryScore {
  category: ScoringCategory;
  category_name: string;
  score: number;
  max_score: number;
  percentage: number;
  weight_management: number;
  weight_leadership: number;
  weighted_score: number;
  level: ScoringLevel;
  question_count: number;
  answered_count: number;
}

export interface ARequirement {
  id: string;
  description: string;
  met: boolean;
  details: string;
}

export interface WhatIfScenario {
  id: string;
  name: string;
  improvements: WhatIfImprovement[];
  projected_score: number;
  projected_level: ScoringLevel;
  score_delta: number;
}

export interface WhatIfImprovement {
  question_id: string;
  question_number: string;
  category: ScoringCategory;
  current_score: number;
  improved_score: number;
  effort: GapEffort;
}

/* ------------------------------------------------------------------ */
/*  Gap Analysis Models                                                */
/* ------------------------------------------------------------------ */

export interface GapAnalysis {
  id: string;
  questionnaire_id: string;
  total_gaps: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  total_uplift_potential: number;
  gaps: GapItem[];
  analyzed_at: string;
}

export interface GapItem {
  id: string;
  question_id: string;
  question_number: string;
  module_code: CDPModule;
  category: ScoringCategory;
  severity: GapSeverity;
  effort: GapEffort;
  current_level: string;
  target_level: string;
  gap_description: string;
  recommendation: string;
  example_response: string | null;
  uplift_points: number;
  is_resolved: boolean;
  resolved_at: string | null;
}

export interface GapRecommendation {
  id: string;
  gap_id: string;
  title: string;
  description: string;
  action_items: string[];
  effort: GapEffort;
  estimated_uplift: number;
  priority_rank: number;
}

/* ------------------------------------------------------------------ */
/*  Benchmarking Models                                                */
/* ------------------------------------------------------------------ */

export interface Benchmark {
  id: string;
  sector: string;
  region: string;
  year: number;
  sample_size: number;
  avg_score: number;
  median_score: number;
  a_list_count: number;
  a_list_rate: number;
  score_distribution: ScoreDistributionBucket[];
  category_averages: Record<ScoringCategory, number>;
}

export interface ScoreDistributionBucket {
  band: ScoringBand;
  level: ScoringLevel;
  count: number;
  percentage: number;
}

export interface PeerComparison {
  id: string;
  org_name: string;
  sector: string;
  region: string;
  score: number;
  level: ScoringLevel;
  rank: number;
  total_peers: number;
  percentile: number;
  category_scores: Record<ScoringCategory, number>;
}

/* ------------------------------------------------------------------ */
/*  Supply Chain Models                                                */
/* ------------------------------------------------------------------ */

export interface SupplierRequest {
  id: string;
  org_id: string;
  supplier_name: string;
  supplier_email: string;
  supplier_country: string;
  supplier_sector: string;
  status: SupplierStatus;
  invited_at: string;
  responded_at: string | null;
  score: number | null;
  scope_1_emissions: number | null;
  scope_2_emissions: number | null;
  scope_3_emissions: number | null;
  engagement_score: number | null;
  has_sbti_target: boolean;
  has_transition_plan: boolean;
}

export interface SupplierResponse {
  id: string;
  request_id: string;
  supplier_name: string;
  total_emissions: number;
  scope_1: number;
  scope_2: number;
  scope_3: number | null;
  data_quality_score: number;
  response_completeness: number;
  submitted_at: string;
}

export interface SupplyChainSummary {
  total_suppliers: number;
  invited_count: number;
  responded_count: number;
  scored_count: number;
  response_rate: number;
  avg_supplier_score: number;
  total_supplier_emissions: number;
  hotspot_categories: EmissionHotspot[];
}

export interface EmissionHotspot {
  category: string;
  supplier_count: number;
  total_emissions: number;
  percentage_of_total: number;
}

/* ------------------------------------------------------------------ */
/*  Transition Plan Models                                             */
/* ------------------------------------------------------------------ */

export interface TransitionPlan {
  id: string;
  org_id: string;
  title: string;
  target_year: number;
  pathway_type: string;
  base_year: number;
  base_year_emissions: number;
  target_emissions: number;
  reduction_target_pct: number;
  annual_reduction_rate: number;
  sbti_aligned: boolean;
  sbti_status: string;
  milestones: TransitionMilestone[];
  technology_levers: TechLever[];
  investment_total_usd: number;
  low_carbon_revenue_pct: number;
  board_oversight: boolean;
  publicly_disclosed: boolean;
  created_at: string;
  updated_at: string;
}

export interface TransitionMilestone {
  id: string;
  plan_id: string;
  title: string;
  description: string;
  target_year: number;
  target_reduction_pct: number;
  status: TransitionMilestoneStatus;
  progress_pct: number;
  responsible: string;
  is_short_term: boolean;
  is_medium_term: boolean;
  is_long_term: boolean;
}

export interface TechLever {
  id: string;
  name: string;
  description: string;
  reduction_potential_tco2e: number;
  investment_required_usd: number;
  timeline: string;
  maturity: string;
}

export interface PathwayPoint {
  year: number;
  target_emissions: number;
  actual_emissions: number | null;
  gap: number | null;
}

/* ------------------------------------------------------------------ */
/*  Verification Models                                                */
/* ------------------------------------------------------------------ */

export interface VerificationRecord {
  id: string;
  org_id: string;
  reporting_year: number;
  scope: string;
  verifier_name: string;
  verifier_accreditation: string;
  verification_level: VerificationLevel;
  coverage_pct: number;
  verified: boolean;
  verification_date: string | null;
  statement_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface VerificationSummary {
  scope_1_verified: boolean;
  scope_1_coverage: number;
  scope_1_level: VerificationLevel;
  scope_2_verified: boolean;
  scope_2_coverage: number;
  scope_2_level: VerificationLevel;
  scope_3_verified: boolean;
  scope_3_coverage: number;
  scope_3_level: VerificationLevel;
  scope_3_categories_verified: string[];
  meets_a_level_scope12: boolean;
  meets_a_level_scope3: boolean;
  overall_meets_a_level: boolean;
}

/* ------------------------------------------------------------------ */
/*  Historical Models                                                  */
/* ------------------------------------------------------------------ */

export interface HistoricalScore {
  year: number;
  score: number;
  level: ScoringLevel;
  band: ScoringBand;
  submitted: boolean;
  submission_date: string | null;
}

export interface YearComparison {
  year_a: number;
  year_b: number;
  score_a: number;
  score_b: number;
  level_a: ScoringLevel;
  level_b: ScoringLevel;
  category_comparison: CategoryComparisonItem[];
  changes: ChangeLogEntry[];
}

export interface CategoryComparisonItem {
  category: ScoringCategory;
  category_name: string;
  score_a: number;
  score_b: number;
  delta: number;
}

export interface ChangeLogEntry {
  id: string;
  question_number: string;
  module_code: CDPModule;
  change_type: 'added' | 'modified' | 'removed';
  description: string;
  impact_on_score: number;
}

/* ------------------------------------------------------------------ */
/*  Dashboard Models                                                   */
/* ------------------------------------------------------------------ */

export interface DashboardData {
  org_id: string;
  reporting_year: number;
  predicted_score: number;
  predicted_level: ScoringLevel;
  predicted_band: ScoringBand;
  previous_score: number | null;
  previous_level: ScoringLevel | null;
  score_delta: number | null;
  completion_pct: number;
  answered_questions: number;
  total_questions: number;
  reviewed_questions: number;
  approved_questions: number;
  gap_summary: GapSummaryData;
  module_progress: ModuleProgress[];
  category_scores: CategoryScore[];
  a_level_status: ARequirement[];
  submission_deadline: string;
  days_until_deadline: number;
  readiness_pct: number;
  recent_activity: TimelineEvent[];
}

export interface GapSummaryData {
  total: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  resolved: number;
}

export interface ModuleProgress {
  module_code: CDPModule;
  module_name: string;
  total_questions: number;
  answered: number;
  reviewed: number;
  approved: number;
  completion_pct: number;
  is_applicable: boolean;
}

export interface TimelineEvent {
  id: string;
  event_type: string;
  description: string;
  user: string;
  timestamp: string;
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
/*  Report Models                                                      */
/* ------------------------------------------------------------------ */

export interface CDPReport {
  id: string;
  questionnaire_id: string;
  title: string;
  format: ReportFormat;
  status: ReportStatus;
  generated_at: string;
  file_url: string | null;
  file_size_bytes: number | null;
}

export interface SubmissionChecklist {
  items: ChecklistItem[];
  total_items: number;
  completed_items: number;
  is_ready: boolean;
}

export interface ChecklistItem {
  id: string;
  description: string;
  completed: boolean;
  severity: 'required' | 'recommended' | 'optional';
  module_code: CDPModule | null;
}

/* ------------------------------------------------------------------ */
/*  Settings Models                                                    */
/* ------------------------------------------------------------------ */

export interface OrganizationSettings {
  id: string;
  org_id: string;
  reporting_year: number;
  reporting_boundary: string;
  gics_sector: string;
  gics_industry_group: string;
  notification_email: string;
  team_members: TeamMember[];
  mrv_connections: MRVConnection[];
  auto_populate_enabled: boolean;
  submission_deadline: string;
}

export interface TeamMember {
  id: string;
  name: string;
  email: string;
  role: string;
  modules_assigned: CDPModule[];
}

export interface MRVConnection {
  agent_name: string;
  agent_id: string;
  connected: boolean;
  last_sync: string | null;
  data_freshness: string;
}

/* ------------------------------------------------------------------ */
/*  Request Types                                                      */
/* ------------------------------------------------------------------ */

export interface CreateQuestionnaireRequest {
  org_id: string;
  reporting_year: number;
  questionnaire_version?: string;
}

export interface SaveResponseRequest {
  response_text: string;
  response_data?: Record<string, unknown> | null;
  table_data?: Record<string, unknown>[] | null;
  status?: ResponseStatus;
}

export interface BulkSaveResponsesRequest {
  responses: Array<{
    question_id: string;
    response_text: string;
    response_data?: Record<string, unknown> | null;
    status?: ResponseStatus;
  }>;
}

export interface SubmitForReviewRequest {
  response_ids: string[];
  reviewer: string;
}

export interface ApproveResponsesRequest {
  response_ids: string[];
  comments?: string;
}

export interface UploadEvidenceRequest {
  file: File;
  description: string;
}

export interface SimulateScoringRequest {
  questionnaire_id: string;
}

export interface WhatIfRequest {
  questionnaire_id: string;
  improvements: Array<{
    question_id: string;
    improved_score: number;
  }>;
}

export interface RunGapAnalysisRequest {
  questionnaire_id: string;
  target_level?: ScoringLevel;
}

export interface InviteSupplierRequest {
  supplier_name: string;
  supplier_email: string;
  supplier_country: string;
  supplier_sector: string;
  message?: string;
}

export interface CreateTransitionPlanRequest {
  title: string;
  target_year: number;
  pathway_type?: string;
  base_year: number;
  base_year_emissions: number;
  target_emissions: number;
}

export interface AddMilestoneRequest {
  title: string;
  description?: string;
  target_year: number;
  target_reduction_pct: number;
  responsible?: string;
}

export interface CreateVerificationRequest {
  scope: string;
  verifier_name: string;
  verifier_accreditation?: string;
  verification_level?: VerificationLevel;
}

export interface GenerateReportRequest {
  questionnaire_id: string;
  format: ReportFormat;
  title?: string;
}

export interface UpdateSettingsRequest {
  reporting_year?: number;
  reporting_boundary?: string;
  gics_sector?: string;
  notification_email?: string;
  auto_populate_enabled?: boolean;
  submission_deadline?: string;
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

export interface QuestionnaireState {
  currentQuestionnaire: Questionnaire | null;
  questionnaires: Questionnaire[];
  modules: Module[];
  currentModule: Module | null;
  loading: boolean;
  error: string | null;
}

export interface ResponseState {
  responses: Record<string, Response>;
  currentResponse: Response | null;
  versions: ResponseVersion[];
  evidence: Evidence[];
  comments: ReviewComment[];
  saving: boolean;
  loading: boolean;
  error: string | null;
}

export interface ScoringState {
  result: ScoringResult | null;
  whatIfScenarios: WhatIfScenario[];
  currentScenario: WhatIfScenario | null;
  simulating: boolean;
  loading: boolean;
  error: string | null;
}

export interface GapAnalysisState {
  analysis: GapAnalysis | null;
  recommendations: GapRecommendation[];
  loading: boolean;
  error: string | null;
}

export interface BenchmarkingState {
  benchmark: Benchmark | null;
  peerComparison: PeerComparison | null;
  loading: boolean;
  error: string | null;
}

export interface SupplyChainState {
  suppliers: SupplierRequest[];
  supplierResponses: SupplierResponse[];
  summary: SupplyChainSummary | null;
  loading: boolean;
  error: string | null;
}

export interface TransitionPlanState {
  plan: TransitionPlan | null;
  pathway: PathwayPoint[];
  loading: boolean;
  error: string | null;
}

export interface CDPVerificationState {
  records: VerificationRecord[];
  summary: VerificationSummary | null;
  loading: boolean;
  error: string | null;
}

export interface HistoricalState {
  scores: HistoricalScore[];
  comparison: YearComparison | null;
  loading: boolean;
  error: string | null;
}

export interface ReportsState {
  reports: CDPReport[];
  checklist: SubmissionChecklist | null;
  generating: boolean;
  loading: boolean;
  error: string | null;
}

export interface CDPDashboardState {
  data: DashboardData | null;
  alerts: DashboardAlert[];
  loading: boolean;
  error: string | null;
}

export interface SettingsState {
  settings: OrganizationSettings | null;
  loading: boolean;
  saving: boolean;
  error: string | null;
}

export interface RootState {
  questionnaire: QuestionnaireState;
  response: ResponseState;
  scoring: ScoringState;
  gapAnalysis: GapAnalysisState;
  benchmarking: BenchmarkingState;
  supplyChain: SupplyChainState;
  transitionPlan: TransitionPlanState;
  verification: CDPVerificationState;
  historical: HistoricalState;
  reports: ReportsState;
  dashboard: CDPDashboardState;
  settings: SettingsState;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

export const CDP_MODULE_NAMES: Record<CDPModule, string> = {
  [CDPModule.M0_INTRODUCTION]: 'Introduction',
  [CDPModule.M1_GOVERNANCE]: 'Governance',
  [CDPModule.M2_POLICIES]: 'Policies & Commitments',
  [CDPModule.M3_RISKS]: 'Risks & Opportunities',
  [CDPModule.M4_STRATEGY]: 'Strategy',
  [CDPModule.M5_TRANSITION]: 'Transition Plans',
  [CDPModule.M6_IMPLEMENTATION]: 'Implementation',
  [CDPModule.M7_CLIMATE_PERFORMANCE]: 'Climate Performance',
  [CDPModule.M8_FORESTS]: 'Forests',
  [CDPModule.M9_WATER]: 'Water Security',
  [CDPModule.M10_SUPPLY_CHAIN]: 'Supply Chain',
  [CDPModule.M11_ADDITIONAL]: 'Additional Metrics',
  [CDPModule.M12_FINANCIAL_SERVICES]: 'Financial Services',
  [CDPModule.M13_SIGN_OFF]: 'Sign Off',
};

export const CDP_MODULE_DESCRIPTIONS: Record<CDPModule, string> = {
  [CDPModule.M0_INTRODUCTION]: 'Organization profile, reporting boundary, base year',
  [CDPModule.M1_GOVERNANCE]: 'Board oversight, management responsibility, incentives',
  [CDPModule.M2_POLICIES]: 'Climate policies, commitments, deforestation-free',
  [CDPModule.M3_RISKS]: 'Climate risk assessment, physical/transition risks',
  [CDPModule.M4_STRATEGY]: 'Business strategy alignment, scenario analysis',
  [CDPModule.M5_TRANSITION]: '1.5C pathway, decarbonization roadmap, milestones',
  [CDPModule.M6_IMPLEMENTATION]: 'Emissions reduction initiatives, investments, R&D',
  [CDPModule.M7_CLIMATE_PERFORMANCE]: 'Scope 1/2/3 emissions, methodology, verification',
  [CDPModule.M8_FORESTS]: 'Commodity-driven deforestation (if applicable)',
  [CDPModule.M9_WATER]: 'Water dependencies (if applicable)',
  [CDPModule.M10_SUPPLY_CHAIN]: 'Supplier engagement, Scope 3 collaboration',
  [CDPModule.M11_ADDITIONAL]: 'Sector-specific metrics, energy mix',
  [CDPModule.M12_FINANCIAL_SERVICES]: 'Portfolio emissions, financed emissions (if FS)',
  [CDPModule.M13_SIGN_OFF]: 'Authorization, verification statement',
};

export const SCORING_CATEGORY_NAMES: Record<ScoringCategory, string> = {
  [ScoringCategory.GOVERNANCE]: 'Governance',
  [ScoringCategory.RISK_MANAGEMENT_PROCESSES]: 'Risk Management Processes',
  [ScoringCategory.RISK_DISCLOSURE]: 'Risk Disclosure',
  [ScoringCategory.OPPORTUNITY_DISCLOSURE]: 'Opportunity Disclosure',
  [ScoringCategory.BUSINESS_STRATEGY]: 'Business Strategy',
  [ScoringCategory.SCENARIO_ANALYSIS]: 'Scenario Analysis',
  [ScoringCategory.TARGETS]: 'Targets',
  [ScoringCategory.EMISSIONS_REDUCTION_INITIATIVES]: 'Emissions Reduction Initiatives',
  [ScoringCategory.SCOPE_1_2_EMISSIONS]: 'Scope 1 & 2 Emissions',
  [ScoringCategory.SCOPE_3_EMISSIONS]: 'Scope 3 Emissions',
  [ScoringCategory.ENERGY]: 'Energy',
  [ScoringCategory.CARBON_PRICING]: 'Carbon Pricing',
  [ScoringCategory.VALUE_CHAIN_ENGAGEMENT]: 'Value Chain Engagement',
  [ScoringCategory.PUBLIC_POLICY_ENGAGEMENT]: 'Public Policy Engagement',
  [ScoringCategory.TRANSITION_PLAN]: 'Transition Plan',
  [ScoringCategory.PORTFOLIO_CLIMATE_PERFORMANCE]: 'Portfolio Climate Performance',
  [ScoringCategory.FINANCIAL_IMPACT_ASSESSMENT]: 'Financial Impact Assessment',
};

export const SCORING_CATEGORY_WEIGHTS_MANAGEMENT: Record<ScoringCategory, number> = {
  [ScoringCategory.GOVERNANCE]: 0.07,
  [ScoringCategory.RISK_MANAGEMENT_PROCESSES]: 0.06,
  [ScoringCategory.RISK_DISCLOSURE]: 0.05,
  [ScoringCategory.OPPORTUNITY_DISCLOSURE]: 0.05,
  [ScoringCategory.BUSINESS_STRATEGY]: 0.06,
  [ScoringCategory.SCENARIO_ANALYSIS]: 0.05,
  [ScoringCategory.TARGETS]: 0.08,
  [ScoringCategory.EMISSIONS_REDUCTION_INITIATIVES]: 0.07,
  [ScoringCategory.SCOPE_1_2_EMISSIONS]: 0.10,
  [ScoringCategory.SCOPE_3_EMISSIONS]: 0.08,
  [ScoringCategory.ENERGY]: 0.06,
  [ScoringCategory.CARBON_PRICING]: 0.04,
  [ScoringCategory.VALUE_CHAIN_ENGAGEMENT]: 0.06,
  [ScoringCategory.PUBLIC_POLICY_ENGAGEMENT]: 0.03,
  [ScoringCategory.TRANSITION_PLAN]: 0.06,
  [ScoringCategory.PORTFOLIO_CLIMATE_PERFORMANCE]: 0.05,
  [ScoringCategory.FINANCIAL_IMPACT_ASSESSMENT]: 0.03,
};

export const SCORING_CATEGORY_WEIGHTS_LEADERSHIP: Record<ScoringCategory, number> = {
  [ScoringCategory.GOVERNANCE]: 0.07,
  [ScoringCategory.RISK_MANAGEMENT_PROCESSES]: 0.05,
  [ScoringCategory.RISK_DISCLOSURE]: 0.04,
  [ScoringCategory.OPPORTUNITY_DISCLOSURE]: 0.04,
  [ScoringCategory.BUSINESS_STRATEGY]: 0.05,
  [ScoringCategory.SCENARIO_ANALYSIS]: 0.05,
  [ScoringCategory.TARGETS]: 0.08,
  [ScoringCategory.EMISSIONS_REDUCTION_INITIATIVES]: 0.07,
  [ScoringCategory.SCOPE_1_2_EMISSIONS]: 0.10,
  [ScoringCategory.SCOPE_3_EMISSIONS]: 0.08,
  [ScoringCategory.ENERGY]: 0.06,
  [ScoringCategory.CARBON_PRICING]: 0.04,
  [ScoringCategory.VALUE_CHAIN_ENGAGEMENT]: 0.06,
  [ScoringCategory.PUBLIC_POLICY_ENGAGEMENT]: 0.03,
  [ScoringCategory.TRANSITION_PLAN]: 0.08,
  [ScoringCategory.PORTFOLIO_CLIMATE_PERFORMANCE]: 0.07,
  [ScoringCategory.FINANCIAL_IMPACT_ASSESSMENT]: 0.03,
};

export const SCORING_LEVEL_ORDER: ScoringLevel[] = [
  ScoringLevel.A,
  ScoringLevel.A_MINUS,
  ScoringLevel.B,
  ScoringLevel.B_MINUS,
  ScoringLevel.C,
  ScoringLevel.C_MINUS,
  ScoringLevel.D,
  ScoringLevel.D_MINUS,
];

export const SCORING_LEVEL_COLORS: Record<ScoringLevel, string> = {
  [ScoringLevel.A]: '#1b5e20',
  [ScoringLevel.A_MINUS]: '#2e7d32',
  [ScoringLevel.B]: '#1565c0',
  [ScoringLevel.B_MINUS]: '#1e88e5',
  [ScoringLevel.C]: '#e65100',
  [ScoringLevel.C_MINUS]: '#ef6c00',
  [ScoringLevel.D]: '#b71c1c',
  [ScoringLevel.D_MINUS]: '#c62828',
};

export const SCORING_BAND_COLORS: Record<ScoringBand, string> = {
  [ScoringBand.LEADERSHIP]: '#1b5e20',
  [ScoringBand.MANAGEMENT]: '#1565c0',
  [ScoringBand.AWARENESS]: '#e65100',
  [ScoringBand.DISCLOSURE]: '#b71c1c',
};

export const MODULE_COLORS: Record<CDPModule, string> = {
  [CDPModule.M0_INTRODUCTION]: '#546e7a',
  [CDPModule.M1_GOVERNANCE]: '#1b5e20',
  [CDPModule.M2_POLICIES]: '#2e7d32',
  [CDPModule.M3_RISKS]: '#e53935',
  [CDPModule.M4_STRATEGY]: '#1565c0',
  [CDPModule.M5_TRANSITION]: '#7b1fa2',
  [CDPModule.M6_IMPLEMENTATION]: '#00838f',
  [CDPModule.M7_CLIMATE_PERFORMANCE]: '#ef6c00',
  [CDPModule.M8_FORESTS]: '#33691e',
  [CDPModule.M9_WATER]: '#0277bd',
  [CDPModule.M10_SUPPLY_CHAIN]: '#4527a0',
  [CDPModule.M11_ADDITIONAL]: '#6d4c41',
  [CDPModule.M12_FINANCIAL_SERVICES]: '#37474f',
  [CDPModule.M13_SIGN_OFF]: '#455a64',
};

export const SCORING_CATEGORY_COLORS: Record<ScoringCategory, string> = {
  [ScoringCategory.GOVERNANCE]: '#1b5e20',
  [ScoringCategory.RISK_MANAGEMENT_PROCESSES]: '#e53935',
  [ScoringCategory.RISK_DISCLOSURE]: '#c62828',
  [ScoringCategory.OPPORTUNITY_DISCLOSURE]: '#2e7d32',
  [ScoringCategory.BUSINESS_STRATEGY]: '#1565c0',
  [ScoringCategory.SCENARIO_ANALYSIS]: '#7b1fa2',
  [ScoringCategory.TARGETS]: '#00838f',
  [ScoringCategory.EMISSIONS_REDUCTION_INITIATIVES]: '#ef6c00',
  [ScoringCategory.SCOPE_1_2_EMISSIONS]: '#d32f2f',
  [ScoringCategory.SCOPE_3_EMISSIONS]: '#f57c00',
  [ScoringCategory.ENERGY]: '#fbc02d',
  [ScoringCategory.CARBON_PRICING]: '#455a64',
  [ScoringCategory.VALUE_CHAIN_ENGAGEMENT]: '#4527a0',
  [ScoringCategory.PUBLIC_POLICY_ENGAGEMENT]: '#6d4c41',
  [ScoringCategory.TRANSITION_PLAN]: '#00695c',
  [ScoringCategory.PORTFOLIO_CLIMATE_PERFORMANCE]: '#37474f',
  [ScoringCategory.FINANCIAL_IMPACT_ASSESSMENT]: '#827717',
};
