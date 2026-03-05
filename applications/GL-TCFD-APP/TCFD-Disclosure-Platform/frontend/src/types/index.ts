/**
 * GL-TCFD-APP Type Definitions
 *
 * Comprehensive TypeScript interfaces for the TCFD Disclosure & Scenario Analysis Platform.
 * Covers all four TCFD pillars: Governance, Strategy, Risk Management, Metrics & Targets.
 */

// ─── Enums as String Unions ───────────────────────────────────────────────────

export type TCFDPillar = 'governance' | 'strategy' | 'risk_management' | 'metrics_targets';

export type RiskCategory = 'physical' | 'transition';

export type PhysicalRiskType = 'acute' | 'chronic';

export type TransitionRiskType = 'policy_legal' | 'technology' | 'market' | 'reputation';

export type OpportunityType =
  | 'resource_efficiency'
  | 'energy_source'
  | 'products_services'
  | 'markets'
  | 'resilience';

export type RiskLevel = 'critical' | 'high' | 'medium' | 'low' | 'negligible';

export type Likelihood =
  | 'almost_certain'
  | 'likely'
  | 'possible'
  | 'unlikely'
  | 'rare';

export type ImpactSeverity = 'catastrophic' | 'major' | 'moderate' | 'minor' | 'insignificant';

export type TimeHorizon = 'short_term' | 'medium_term' | 'long_term';

export type ScenarioType =
  | 'orderly_transition'
  | 'disorderly_transition'
  | 'hot_house'
  | 'net_zero_2050'
  | 'delayed_transition'
  | 'current_policies'
  | 'custom';

export type TemperatureTarget = '1.5C' | '2.0C' | '2.5C' | '3.0C' | '4.0C';

export type DisclosureStatus = 'not_started' | 'in_progress' | 'draft' | 'review' | 'final' | 'published';

export type GapSeverity = 'critical' | 'major' | 'moderate' | 'minor';

export type ComplianceFramework = 'tcfd' | 'issb_s2' | 'csrd' | 'sec_climate';

export type MetricCategory =
  | 'ghg_emissions'
  | 'transition_risks'
  | 'physical_risks'
  | 'opportunities'
  | 'capital_deployment'
  | 'internal_carbon_price'
  | 'remuneration';

export type TargetType = 'absolute' | 'intensity' | 'engagement' | 'portfolio';

export type GovernanceMaturity = 'initial' | 'developing' | 'defined' | 'managed' | 'optimizing';

export type FinancialStatementType = 'income_statement' | 'balance_sheet' | 'cash_flow';

export type Currency = 'USD' | 'EUR' | 'GBP' | 'JPY' | 'AUD' | 'CAD' | 'CHF';

export type HazardType =
  | 'tropical_cyclone'
  | 'flood'
  | 'drought'
  | 'wildfire'
  | 'sea_level_rise'
  | 'extreme_heat'
  | 'extreme_cold'
  | 'precipitation'
  | 'storm_surge'
  | 'permafrost_thaw';

export type AssetCategory =
  | 'real_estate'
  | 'infrastructure'
  | 'manufacturing'
  | 'supply_chain'
  | 'fleet'
  | 'data_center'
  | 'agricultural'
  | 'other';

export type ResponseStrategy = 'mitigate' | 'transfer' | 'accept' | 'avoid' | 'exploit';

export type SBTiStatus = 'committed' | 'targets_set' | 'validated' | 'not_committed';

export type ExportFormat = 'pdf' | 'docx' | 'xlsx' | 'json' | 'xbrl';

// ─── Governance ───────────────────────────────────────────────────────────────

export interface GovernanceRole {
  id: string;
  title: string;
  name: string;
  role_type: 'board' | 'committee' | 'executive' | 'management';
  responsibilities: string[];
  climate_competencies: string[];
  reporting_frequency: string;
  last_review_date: string;
  created_at: string;
  updated_at: string;
}

export interface GovernanceCommittee {
  id: string;
  name: string;
  type: 'board_committee' | 'management_committee' | 'working_group';
  chair: string;
  members: string[];
  mandate: string;
  meeting_frequency: string;
  last_meeting_date: string;
  next_meeting_date: string;
  climate_agenda_items: number;
  total_agenda_items: number;
}

export interface GovernanceAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  board_oversight_score: number;
  management_role_score: number;
  climate_competency_score: number;
  reporting_frequency_score: number;
  integration_score: number;
  incentive_alignment_score: number;
  training_score: number;
  stakeholder_engagement_score: number;
  overall_maturity: GovernanceMaturity;
  overall_score: number;
  recommendations: Recommendation[];
  roles: GovernanceRole[];
  committees: GovernanceCommittee[];
  created_at: string;
  updated_at: string;
}

export interface CompetencyEntry {
  id: string;
  person_name: string;
  role: string;
  climate_science: number;
  ghg_accounting: number;
  scenario_analysis: number;
  risk_management: number;
  regulatory_knowledge: number;
  financial_impact: number;
  strategy_development: number;
  stakeholder_engagement: number;
  overall_score: number;
  training_plan: string;
}

// ─── Strategy ─────────────────────────────────────────────────────────────────

export interface ClimateRisk {
  id: string;
  organization_id: string;
  category: RiskCategory;
  risk_type: PhysicalRiskType | TransitionRiskType;
  name: string;
  description: string;
  risk_level: RiskLevel;
  likelihood: Likelihood;
  impact_severity: ImpactSeverity;
  time_horizon: TimeHorizon;
  affected_areas: string[];
  financial_impact_low: number;
  financial_impact_mid: number;
  financial_impact_high: number;
  velocity: 'gradual' | 'rapid' | 'sudden';
  interconnections: string[];
  mitigation_actions: string[];
  residual_risk_level: RiskLevel;
  owner: string;
  last_assessed: string;
  next_review: string;
  created_at: string;
  updated_at: string;
}

export interface ClimateOpportunity {
  id: string;
  organization_id: string;
  opportunity_type: OpportunityType;
  name: string;
  description: string;
  time_horizon: TimeHorizon;
  strategic_priority: 'critical' | 'high' | 'medium' | 'low';
  revenue_potential_low: number;
  revenue_potential_mid: number;
  revenue_potential_high: number;
  cost_savings_low: number;
  cost_savings_mid: number;
  cost_savings_high: number;
  investment_required: number;
  payback_period_years: number;
  feasibility_score: number;
  impact_score: number;
  status: 'identified' | 'evaluating' | 'approved' | 'implementing' | 'realized';
  responsible_team: string;
  value_chain_position: 'upstream' | 'direct_operations' | 'downstream';
  related_risks: string[];
  kpis: string[];
  created_at: string;
  updated_at: string;
}

export interface BusinessModelImpact {
  id: string;
  area: string;
  description: string;
  risk_impacts: { risk_id: string; risk_name: string; impact_value: number }[];
  opportunity_impacts: { opportunity_id: string; opportunity_name: string; impact_value: number }[];
  net_impact: number;
  confidence: 'high' | 'medium' | 'low';
}

export interface ValueChainNode {
  id: string;
  name: string;
  position: 'upstream' | 'direct_operations' | 'downstream';
  risks: { id: string; name: string; level: RiskLevel }[];
  opportunities: { id: string; name: string; priority: string }[];
  emissions_scope: 'scope_1' | 'scope_2' | 'scope_3';
  financial_exposure: number;
}

// ─── Scenario Analysis ────────────────────────────────────────────────────────

export interface ScenarioParameter {
  id: string;
  name: string;
  unit: string;
  min_value: number;
  max_value: number;
  default_value: number;
  current_value: number;
  description: string;
  category: 'carbon_price' | 'energy_mix' | 'temperature' | 'policy' | 'technology' | 'market';
}

export interface ScenarioDefinition {
  id: string;
  organization_id: string;
  name: string;
  description: string;
  scenario_type: ScenarioType;
  temperature_target: TemperatureTarget;
  time_horizon_years: number;
  base_year: number;
  parameters: ScenarioParameter[];
  source: string;
  narrative: string;
  assumptions: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface ScenarioResult {
  id: string;
  scenario_id: string;
  scenario_name: string;
  year: number;
  revenue_impact: number;
  cost_impact: number;
  asset_impairment: number;
  capex_required: number;
  stranded_asset_value: number;
  carbon_cost: number;
  opportunity_value: number;
  net_financial_impact: number;
  emissions_pathway: number;
  energy_mix: { source: string; percentage: number }[];
  confidence_interval_low: number;
  confidence_interval_high: number;
}

export interface SensitivityResult {
  parameter_name: string;
  base_value: number;
  low_value: number;
  high_value: number;
  low_impact: number;
  high_impact: number;
  unit: string;
}

export interface StrandingDataPoint {
  year: number;
  scenario_name: string;
  percentage_at_risk: number;
  value_at_risk: number;
}

// ─── Physical Risk ────────────────────────────────────────────────────────────

export interface AssetLocation {
  id: string;
  name: string;
  asset_type: AssetCategory;
  latitude: number;
  longitude: number;
  country: string;
  region: string;
  book_value: number;
  replacement_cost: number;
  annual_revenue: number;
  employees: number;
  hazards: HazardExposure[];
  overall_risk_score: number;
  risk_level: RiskLevel;
  insurance_coverage: number;
  adaptation_measures: string[];
}

export interface HazardExposure {
  hazard_type: HazardType;
  exposure_score: number;
  vulnerability_score: number;
  risk_score: number;
  risk_level: RiskLevel;
  return_period_years: number;
  projected_change_pct: number;
  time_horizon: TimeHorizon;
}

export interface PhysicalRiskAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  assets: AssetLocation[];
  total_assets_at_risk: number;
  total_value_at_risk: number;
  hazard_summary: { hazard_type: HazardType; asset_count: number; total_exposure: number }[];
  insurance_gap: number;
  supply_chain_risks: SupplyChainRiskNode[];
  created_at: string;
  updated_at: string;
}

export interface SupplyChainRiskNode {
  id: string;
  supplier_name: string;
  tier: number;
  location: string;
  hazard_exposures: HazardType[];
  risk_level: RiskLevel;
  revenue_dependency: number;
  alternative_suppliers: number;
  lead_time_impact_days: number;
}

export interface InsuranceCostProjection {
  year: number;
  scenario: string;
  baseline_premium: number;
  projected_premium: number;
  increase_pct: number;
  coverage_gap: number;
}

// ─── Transition Risk ──────────────────────────────────────────────────────────

export interface TransitionRiskAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  policy_risks: PolicyRisk[];
  technology_risks: TechnologyRisk[];
  market_risks: MarketRisk[];
  reputation_risks: ReputationRisk[];
  total_exposure: number;
  stranded_asset_value: number;
  carbon_cost_exposure: number;
  created_at: string;
  updated_at: string;
}

export interface PolicyRisk {
  id: string;
  name: string;
  jurisdiction: string;
  effective_date: string;
  description: string;
  risk_level: RiskLevel;
  financial_impact: number;
  compliance_status: 'compliant' | 'partial' | 'non_compliant' | 'not_applicable';
  mitigation_plan: string;
}

export interface TechnologyRisk {
  id: string;
  technology: string;
  disruption_type: 'substitution' | 'efficiency' | 'process_change' | 'new_market';
  current_adoption_pct: number;
  projected_adoption_pct: number;
  adoption_timeline_years: number;
  impact_on_revenue: number;
  impact_on_costs: number;
  readiness_score: number;
  investment_required: number;
}

export interface MarketRisk {
  id: string;
  market_segment: string;
  shift_description: string;
  demand_change_pct: number;
  timeline_years: number;
  revenue_at_risk: number;
  market_opportunity: number;
  confidence: 'high' | 'medium' | 'low';
}

export interface ReputationRisk {
  id: string;
  factor: string;
  current_score: number;
  trend: 'improving' | 'stable' | 'declining';
  stakeholder_group: string;
  impact_description: string;
  mitigation_actions: string[];
}

// ─── Financial Impact ─────────────────────────────────────────────────────────

export interface FinancialImpact {
  id: string;
  organization_id: string;
  scenario_id: string;
  scenario_name: string;
  year: number;
  statement_type: FinancialStatementType;
  line_items: FinancialLineItem[];
  total_impact: number;
  currency: Currency;
  created_at: string;
}

export interface FinancialLineItem {
  id: string;
  category: string;
  line_item: string;
  baseline_value: number;
  climate_impact: number;
  adjusted_value: number;
  impact_pct: number;
  risk_drivers: string[];
  opportunity_drivers: string[];
  confidence: 'high' | 'medium' | 'low';
  notes: string;
}

export interface MACCDataPoint {
  id: string;
  measure: string;
  abatement_potential_tco2e: number;
  cost_per_tco2e: number;
  investment_required: number;
  payback_years: number;
  category: string;
  status: 'implemented' | 'approved' | 'evaluating' | 'identified';
}

export interface NPVResult {
  scenario_id: string;
  scenario_name: string;
  npv: number;
  irr: number;
  payback_years: number;
  discount_rate: number;
  cash_flows: { year: number; amount: number }[];
  sensitivity: { discount_rate: number; npv: number }[];
}

export interface MonteCarloResult {
  scenario_id: string;
  scenario_name: string;
  iterations: number;
  mean: number;
  median: number;
  std_dev: number;
  p5: number;
  p25: number;
  p75: number;
  p95: number;
  distribution: { bin_start: number; bin_end: number; frequency: number }[];
}

// ─── Risk Management ──────────────────────────────────────────────────────────

export interface RiskManagementRecord {
  id: string;
  risk_id: string;
  risk_name: string;
  risk_category: RiskCategory;
  risk_type: string;
  risk_level: RiskLevel;
  likelihood: Likelihood;
  impact_severity: ImpactSeverity;
  likelihood_score: number;
  impact_score: number;
  inherent_risk_score: number;
  response_strategy: ResponseStrategy;
  response_actions: RiskResponseAction[];
  residual_risk_score: number;
  residual_risk_level: RiskLevel;
  owner: string;
  status: 'open' | 'mitigating' | 'monitoring' | 'closed';
  erm_integrated: boolean;
  last_review: string;
  next_review: string;
  created_at: string;
  updated_at: string;
}

export interface RiskResponseAction {
  id: string;
  description: string;
  responsible: string;
  due_date: string;
  status: 'not_started' | 'in_progress' | 'completed' | 'overdue';
  effectiveness: number;
  cost: number;
}

export interface RiskIndicator {
  id: string;
  name: string;
  type: 'leading' | 'lagging';
  current_value: number;
  threshold_warning: number;
  threshold_critical: number;
  unit: string;
  trend: 'improving' | 'stable' | 'deteriorating';
  history: { date: string; value: number }[];
}

export interface HeatMapCell {
  likelihood_score: number;
  impact_score: number;
  risk_count: number;
  risks: { id: string; name: string; risk_level: RiskLevel }[];
}

// ─── Metrics & Targets ───────────────────────────────────────────────────────

export interface ClimateMetric {
  id: string;
  organization_id: string;
  category: MetricCategory;
  name: string;
  description: string;
  value: number;
  unit: string;
  reporting_year: number;
  previous_value: number;
  change_pct: number;
  methodology: string;
  data_quality_score: number;
  verified: boolean;
  scope: 'scope_1' | 'scope_2_location' | 'scope_2_market' | 'scope_3' | 'total' | 'other';
  industry_benchmark: number | null;
  peer_average: number | null;
  created_at: string;
  updated_at: string;
}

export interface ClimateTarget {
  id: string;
  organization_id: string;
  name: string;
  description: string;
  target_type: TargetType;
  metric_id: string;
  metric_name: string;
  base_year: number;
  base_value: number;
  target_year: number;
  target_value: number;
  interim_targets: { year: number; value: number }[];
  current_value: number;
  progress_pct: number;
  on_track: boolean;
  sbti_aligned: boolean;
  sbti_status: SBTiStatus;
  scope: string;
  methodology: string;
  created_at: string;
  updated_at: string;
}

export interface TargetProgress {
  target_id: string;
  target_name: string;
  years: number[];
  actual_values: number[];
  target_pathway: number[];
  sbti_pathway: number[];
  on_track: boolean;
  gap_to_target: number;
}

export interface EmissionsSummary {
  reporting_year: number;
  scope_1: number;
  scope_2_location: number;
  scope_2_market: number;
  scope_3_categories: { category: number; name: string; value: number }[];
  scope_3_total: number;
  total_emissions: number;
  previous_year_total: number;
  change_pct: number;
  intensity_revenue: number;
  intensity_employee: number;
  intensity_unit: number;
}

export interface PeerBenchmarkData {
  metric_name: string;
  unit: string;
  organization_value: number;
  peer_values: { name: string; value: number }[];
  industry_average: number;
  best_in_class: number;
}

// ─── Disclosure ───────────────────────────────────────────────────────────────

export interface TCFDDisclosure {
  id: string;
  organization_id: string;
  reporting_year: number;
  title: string;
  status: DisclosureStatus;
  pillar: TCFDPillar;
  sections: DisclosureSection[];
  overall_completeness: number;
  compliance_score: number;
  last_modified: string;
  published_date: string | null;
  created_at: string;
  updated_at: string;
}

export interface DisclosureSection {
  id: string;
  disclosure_id: string;
  tcfd_recommendation: string;
  tcfd_code: string;
  title: string;
  content: string;
  status: DisclosureStatus;
  completeness_score: number;
  evidence_ids: string[];
  reviewer_notes: string;
  pillar: TCFDPillar;
  issb_mapping: string | null;
  order: number;
}

export interface Evidence {
  id: string;
  title: string;
  type: 'document' | 'data' | 'meeting_minutes' | 'board_resolution' | 'policy' | 'report';
  source: string;
  date: string;
  url: string | null;
  linked_sections: string[];
  tags: string[];
}

export interface ComplianceCheck {
  id: string;
  framework: ComplianceFramework;
  requirement: string;
  requirement_code: string;
  status: 'met' | 'partial' | 'not_met' | 'not_applicable';
  evidence_quality: 'strong' | 'adequate' | 'weak' | 'missing';
  gap_description: string | null;
  recommendation: string | null;
}

// ─── Gap Analysis ─────────────────────────────────────────────────────────────

export interface GapAssessment {
  id: string;
  organization_id: string;
  assessment_date: string;
  framework: ComplianceFramework;
  pillar_scores: { pillar: TCFDPillar; score: number; max_score: number }[];
  overall_score: number;
  overall_max_score: number;
  maturity_level: GovernanceMaturity;
  gaps: GapItem[];
  actions: GapAction[];
  peer_comparison: { percentile: number; peer_group: string };
  created_at: string;
  updated_at: string;
}

export interface GapItem {
  id: string;
  pillar: TCFDPillar;
  requirement: string;
  requirement_code: string;
  current_state: string;
  desired_state: string;
  severity: GapSeverity;
  effort_estimate: 'low' | 'medium' | 'high';
  priority: number;
  responsible_team: string;
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

// ─── ISSB Cross-Walk ──────────────────────────────────────────────────────────

export interface ISSBMapping {
  id: string;
  tcfd_code: string;
  tcfd_requirement: string;
  tcfd_pillar: TCFDPillar;
  issb_code: string;
  issb_requirement: string;
  issb_standard: 'IFRS_S1' | 'IFRS_S2';
  mapping_type: 'direct' | 'partial' | 'enhanced' | 'new';
  gap_description: string | null;
  migration_effort: 'low' | 'medium' | 'high';
  notes: string;
}

export interface DualComplianceScore {
  framework: ComplianceFramework;
  total_requirements: number;
  met: number;
  partial: number;
  not_met: number;
  not_applicable: number;
  score_pct: number;
}

export interface MigrationChecklistItem {
  id: string;
  category: string;
  requirement: string;
  tcfd_status: 'met' | 'partial' | 'not_met';
  issb_status: 'met' | 'partial' | 'not_met';
  action_required: string;
  priority: 'high' | 'medium' | 'low';
  completed: boolean;
}

// ─── Dashboard ────────────────────────────────────────────────────────────────

export interface DashboardSummary {
  risk_exposure: {
    total_financial_impact: number;
    physical_risk_total: number;
    transition_risk_total: number;
    risk_count_by_level: Record<RiskLevel, number>;
    top_risks: { id: string; name: string; level: RiskLevel; impact: number }[];
  };
  opportunity_value: {
    total_opportunity_value: number;
    total_cost_savings: number;
    opportunity_count_by_type: Record<OpportunityType, number>;
    top_opportunities: { id: string; name: string; value: number; status: string }[];
  };
  disclosure_maturity: {
    overall_pct: number;
    pillar_scores: Record<TCFDPillar, number>;
    section_statuses: { code: string; title: string; status: DisclosureStatus }[];
  };
  scenario_summary: {
    scenarios_analyzed: number;
    net_impact_range: { low: number; high: number };
    key_driver: string;
    scenario_results: { name: string; net_impact: number }[];
  };
  emissions_summary: EmissionsSummary;
  recent_activity: ActivityItem[];
}

export interface ActivityItem {
  id: string;
  type: 'risk_update' | 'disclosure_edit' | 'scenario_run' | 'target_update' | 'assessment';
  description: string;
  user: string;
  timestamp: string;
}

// ─── Recommendations ──────────────────────────────────────────────────────────

export interface Recommendation {
  id: string;
  category: string;
  title: string;
  description: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  effort: 'low' | 'medium' | 'high';
  impact: 'high' | 'medium' | 'low';
  status: 'open' | 'accepted' | 'in_progress' | 'completed' | 'deferred';
}

// ─── Settings ─────────────────────────────────────────────────────────────────

export interface OrganizationSettings {
  id: string;
  organization_name: string;
  industry_sector: string;
  sub_sector: string;
  reporting_currency: Currency;
  fiscal_year_end_month: number;
  base_year: number;
  default_scenarios: string[];
  default_time_horizons: {
    short_term_years: number;
    medium_term_years: number;
    long_term_years: number;
  };
  disclosure_frameworks: ComplianceFramework[];
  auto_save: boolean;
  notification_preferences: {
    email_alerts: boolean;
    deadline_reminders: boolean;
    risk_threshold_alerts: boolean;
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
