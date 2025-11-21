/**
 * GL-007 Furnace Performance Monitor - Type Definitions
 *
 * Comprehensive TypeScript types for furnace monitoring system
 */

// ============================================================================
// CORE TYPES
// ============================================================================

export interface FurnaceConfig {
  id: string;
  name: string;
  type: FurnaceType;
  manufacturer: string;
  model: string;
  capacity: number; // tonnes/hour
  zones: ThermalZone[];
  sensors: Sensor[];
  fuelType: FuelType;
  installDate: string;
  location: Location;
  status: OperationalStatus;
  specifications: FurnaceSpecs;
}

export type FurnaceType =
  | 'blast_furnace'
  | 'arc_furnace'
  | 'induction_furnace'
  | 'reverberatory_furnace'
  | 'rotary_kiln'
  | 'tunnel_kiln';

export type FuelType = 'natural_gas' | 'coal' | 'electricity' | 'oil' | 'hydrogen' | 'mixed';

export type OperationalStatus = 'running' | 'idle' | 'maintenance' | 'fault' | 'startup' | 'shutdown';

export interface FurnaceSpecs {
  maxTemperature: number; // °C
  minTemperature: number;
  optimalTemperature: number;
  maxPressure: number; // bar
  maxFuelConsumption: number; // kg/hour or kWh
  thermalEfficiency: number; // %
  emissionLimits: EmissionLimits;
}

export interface EmissionLimits {
  co2: number; // kg/tonne
  nox: number;
  sox: number;
  particulates: number;
}

export interface Location {
  plant: string;
  site: string;
  building: string;
  coordinates?: {
    latitude: number;
    longitude: number;
  };
}

// ============================================================================
// SENSOR & MONITORING
// ============================================================================

export interface Sensor {
  id: string;
  type: SensorType;
  location: string;
  unit: string;
  minValue: number;
  maxValue: number;
  accuracy: number;
  calibrationDate: string;
  status: SensorStatus;
  zoneId?: string;
}

export type SensorType =
  | 'temperature'
  | 'pressure'
  | 'flow'
  | 'oxygen'
  | 'co'
  | 'nox'
  | 'thermal_imaging'
  | 'vibration'
  | 'power';

export type SensorStatus = 'active' | 'degraded' | 'fault' | 'calibration_required';

export interface SensorReading {
  sensorId: string;
  timestamp: string;
  value: number;
  quality: DataQuality;
  alarm?: AlarmLevel;
}

export type DataQuality = 'good' | 'uncertain' | 'bad' | 'estimated';
export type AlarmLevel = 'critical' | 'high' | 'medium' | 'low';

export interface ThermalZone {
  id: string;
  name: string;
  sequence: number;
  targetTemperature: number;
  temperatureRange: {
    min: number;
    max: number;
    optimal: number;
  };
  sensors: string[]; // sensor IDs
  controlMode: ControlMode;
}

export type ControlMode = 'manual' | 'automatic' | 'cascade' | 'feedforward';

// ============================================================================
// REAL-TIME PERFORMANCE
// ============================================================================

export interface FurnacePerformance {
  furnaceId: string;
  timestamp: string;
  kpis: PerformanceKPIs;
  thermal: ThermalPerformance;
  efficiency: EfficiencyMetrics;
  fuel: FuelMetrics;
  production: ProductionMetrics;
  emissions: EmissionsData;
  health: HealthScore;
}

export interface PerformanceKPIs {
  overallEfficiency: number; // %
  thermalEfficiency: number;
  fuelEfficiency: number;
  productionRate: number; // tonnes/hour
  specificEnergyConsumption: number; // MJ/tonne
  availabilityFactor: number; // %
  utilizationRate: number; // %
  qualityIndex: number; // %
  costPerTonne: number; // USD
  carbonIntensity: number; // kgCO2/tonne
}

export interface ThermalPerformance {
  zones: ZonePerformance[];
  averageTemperature: number;
  temperatureUniformity: number; // %
  hotSpots: HotSpot[];
  coldSpots: ColdSpot[];
  thermalProfile: ThermalProfile;
  heatLosses: HeatLoss[];
}

export interface ZonePerformance {
  zoneId: string;
  currentTemperature: number;
  targetTemperature: number;
  deviation: number;
  stability: number; // %
  controlPerformance: number; // %
  trend: Trend;
}

export type Trend = 'increasing' | 'decreasing' | 'stable' | 'fluctuating';

export interface HotSpot {
  location: string;
  temperature: number;
  severity: 'critical' | 'high' | 'medium';
  zoneId: string;
  recommendation: string;
}

export interface ColdSpot {
  location: string;
  temperature: number;
  impact: 'high' | 'medium' | 'low';
  zoneId: string;
  recommendation: string;
}

export interface ThermalProfile {
  data: ThermalDataPoint[];
  timestamp: string;
  uniformityIndex: number;
}

export interface ThermalDataPoint {
  x: number;
  y: number;
  temperature: number;
  zoneId: string;
}

export interface HeatLoss {
  location: string;
  type: HeatLossType;
  amount: number; // kW
  percentage: number; // % of total input
  cost: number; // USD/hour
}

export type HeatLossType =
  | 'wall_conduction'
  | 'opening_radiation'
  | 'cooling_water'
  | 'exhaust_gas'
  | 'product_heat';

export interface EfficiencyMetrics {
  thermal: {
    current: number;
    target: number;
    optimal: number;
    trend: TrendData[];
  };
  combustion: {
    efficiency: number;
    excessAir: number;
    o2Level: number;
  };
  heat: {
    input: number; // kW
    useful: number;
    losses: number;
    recovered: number;
  };
  overall: {
    oee: number; // Overall Equipment Effectiveness
    availability: number;
    performance: number;
    quality: number;
  };
}

export interface TrendData {
  timestamp: string;
  value: number;
}

export interface FuelMetrics {
  consumption: {
    current: number; // kg/hour or kWh
    average24h: number;
    specific: number; // per tonne product
    trend: TrendData[];
  };
  cost: {
    current: number; // USD/hour
    daily: number;
    monthly: number;
    perTonne: number;
  };
  pressure: number;
  flowRate: number;
  temperature: number;
  quality: FuelQuality;
}

export interface FuelQuality {
  heatingValue: number; // MJ/kg
  composition: {
    [component: string]: number; // percentage
  };
  moisture?: number;
  ash?: number;
}

export interface ProductionMetrics {
  rate: number; // tonnes/hour
  target: number;
  achievement: number; // %
  quality: QualityMetrics;
  downtime: DowntimeData;
  throughput: {
    hourly: number;
    daily: number;
    weekly: number;
    monthly: number;
  };
}

export interface QualityMetrics {
  conformance: number; // %
  defectRate: number;
  rework: number;
  yield: number;
  specifications: SpecificationCompliance[];
}

export interface SpecificationCompliance {
  parameter: string;
  target: number;
  actual: number;
  tolerance: number;
  status: 'pass' | 'fail' | 'warning';
}

export interface DowntimeData {
  planned: number; // minutes
  unplanned: number;
  reasons: DowntimeReason[];
  mtbf: number; // Mean Time Between Failures (hours)
  mttr: number; // Mean Time To Repair (hours)
}

export interface DowntimeReason {
  reason: string;
  duration: number;
  category: DowntimeCategory;
  cost: number;
}

export type DowntimeCategory =
  | 'equipment_failure'
  | 'maintenance'
  | 'material_shortage'
  | 'quality_issue'
  | 'operator_error'
  | 'power_outage';

export interface EmissionsData {
  co2: EmissionMetric;
  nox: EmissionMetric;
  sox: EmissionMetric;
  co: EmissionMetric;
  particulates: EmissionMetric;
  total: {
    mass: number; // kg/hour
    specific: number; // kg/tonne product
    cost: number; // carbon cost USD/hour
  };
}

export interface EmissionMetric {
  current: number;
  limit: number;
  compliance: number; // %
  trend: TrendData[];
}

export interface HealthScore {
  overall: number; // 0-100
  components: ComponentHealth[];
  predicted: PredictedHealth[];
  recommendations: HealthRecommendation[];
}

export interface ComponentHealth {
  component: string;
  score: number; // 0-100
  status: ComponentStatus;
  remainingLife: number; // hours
  nextMaintenance: string;
  criticalityLevel: 'critical' | 'high' | 'medium' | 'low';
}

export type ComponentStatus = 'healthy' | 'degrading' | 'critical' | 'failed';

export interface PredictedHealth {
  component: string;
  currentScore: number;
  predicted7d: number;
  predicted30d: number;
  failureProbability: number; // %
  confidenceLevel: number; // %
}

export interface HealthRecommendation {
  id: string;
  priority: Priority;
  component: string;
  issue: string;
  recommendation: string;
  estimatedCost: number;
  estimatedSavings: number;
  roi: number;
  timeline: string;
}

export type Priority = 'critical' | 'high' | 'medium' | 'low';

// ============================================================================
// ALERTS & ALARMS
// ============================================================================

export interface Alert {
  id: string;
  timestamp: string;
  furnaceId: string;
  type: AlertType;
  severity: AlertSeverity;
  category: AlertCategory;
  title: string;
  message: string;
  source: string;
  value?: number;
  threshold?: number;
  status: AlertStatus;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  resolvedAt?: string;
  actions: AlertAction[];
  rootCause?: RootCause;
}

export type AlertType =
  | 'threshold_exceeded'
  | 'trend_deviation'
  | 'equipment_fault'
  | 'sensor_fault'
  | 'efficiency_drop'
  | 'emission_violation'
  | 'predictive_warning';

export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

export type AlertCategory =
  | 'safety'
  | 'operational'
  | 'maintenance'
  | 'quality'
  | 'environmental'
  | 'efficiency';

export type AlertStatus = 'active' | 'acknowledged' | 'resolved' | 'suppressed';

export interface AlertAction {
  id: string;
  action: string;
  takenBy?: string;
  takenAt?: string;
  result?: string;
}

export interface RootCause {
  identified: boolean;
  cause: string;
  category: string;
  contributingFactors: string[];
  correctiveActions: string[];
  preventiveMeasures: string[];
}

export interface AlertConfiguration {
  furnaceId: string;
  alerts: AlertConfig[];
  notifications: NotificationConfig;
}

export interface AlertConfig {
  id: string;
  name: string;
  type: AlertType;
  enabled: boolean;
  thresholds: {
    critical?: number;
    high?: number;
    medium?: number;
    low?: number;
  };
  hysteresis: number;
  delay: number; // seconds
  escalation: EscalationRule[];
}

export interface EscalationRule {
  level: number;
  delay: number; // minutes
  recipients: string[];
  method: NotificationMethod[];
}

export type NotificationMethod = 'email' | 'sms' | 'push' | 'webhook';

export interface NotificationConfig {
  email: {
    enabled: boolean;
    recipients: string[];
    includeCharts: boolean;
  };
  sms: {
    enabled: boolean;
    recipients: string[];
    criticalOnly: boolean;
  };
  push: {
    enabled: boolean;
    devices: string[];
  };
  webhook: {
    enabled: boolean;
    url: string;
    headers?: Record<string, string>;
  };
}

// ============================================================================
// MAINTENANCE
// ============================================================================

export interface MaintenanceSchedule {
  furnaceId: string;
  tasks: MaintenanceTask[];
  nextScheduled: string;
  upcomingCount: number;
  overdueCount: number;
}

export interface MaintenanceTask {
  id: string;
  type: MaintenanceType;
  component: string;
  description: string;
  scheduledDate: string;
  dueDate: string;
  priority: Priority;
  status: MaintenanceStatus;
  estimatedDuration: number; // hours
  estimatedCost: number;
  assignedTo?: string;
  completedAt?: string;
  completedBy?: string;
  notes?: string;
  parts: SparePart[];
  procedures: Procedure[];
}

export type MaintenanceType =
  | 'preventive'
  | 'predictive'
  | 'corrective'
  | 'condition_based'
  | 'inspection';

export type MaintenanceStatus =
  | 'scheduled'
  | 'due'
  | 'overdue'
  | 'in_progress'
  | 'completed'
  | 'cancelled'
  | 'deferred';

export interface SparePart {
  id: string;
  name: string;
  quantity: number;
  inStock: number;
  cost: number;
  leadTime: number; // days
  criticality: 'critical' | 'high' | 'medium' | 'low';
}

export interface Procedure {
  id: string;
  name: string;
  steps: ProcedureStep[];
  safetyRequirements: string[];
  estimatedTime: number; // minutes
}

export interface ProcedureStep {
  sequence: number;
  description: string;
  completed: boolean;
  verification?: string;
}

export interface RefractoryCondition {
  furnaceId: string;
  zones: RefractoryZone[];
  overallCondition: number; // %
  remainingLife: number; // hours
  nextInspection: string;
  recommendations: RefractoryRecommendation[];
}

export interface RefractoryZone {
  zoneId: string;
  thickness: number; // mm
  originalThickness: number;
  wearRate: number; // mm/day
  condition: number; // %
  hotSpotCount: number;
  crackCount: number;
  estimatedLife: number; // days
}

export interface RefractoryRecommendation {
  zone: string;
  issue: string;
  severity: AlertSeverity;
  action: string;
  urgency: 'immediate' | 'short_term' | 'medium_term' | 'long_term';
  cost: number;
}

// ============================================================================
// ANALYTICS
// ============================================================================

export interface AnalyticsData {
  furnaceId: string;
  period: TimePeriod;
  trends: TrendAnalysis;
  benchmarking: BenchmarkData;
  optimization: OptimizationOpportunity[];
  rootCauseAnalysis: RootCauseAnalysis[];
  whatIfScenarios?: WhatIfScenario[];
}

export interface TimePeriod {
  start: string;
  end: string;
  granularity: 'minute' | 'hour' | 'day' | 'week' | 'month';
}

export interface TrendAnalysis {
  efficiency: TimeSeriesData;
  fuelConsumption: TimeSeriesData;
  production: TimeSeriesData;
  emissions: TimeSeriesData;
  costs: TimeSeriesData;
  quality: TimeSeriesData;
}

export interface TimeSeriesData {
  data: TimeSeriesPoint[];
  statistics: Statistics;
  forecast?: TimeSeriesPoint[];
}

export interface TimeSeriesPoint {
  timestamp: string;
  value: number;
  confidence?: number; // for forecasts
}

export interface Statistics {
  mean: number;
  median: number;
  stdDev: number;
  min: number;
  max: number;
  percentile95: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  changeRate: number; // % change
}

export interface BenchmarkData {
  furnacePerformance: BenchmarkMetric;
  industryAverage: BenchmarkMetric;
  bestInClass: BenchmarkMetric;
  gap: BenchmarkGap;
  ranking: number; // percentile
}

export interface BenchmarkMetric {
  efficiency: number;
  fuelConsumption: number;
  emissions: number;
  availability: number;
  quality: number;
}

export interface BenchmarkGap {
  efficiency: number; // % difference from best
  potentialSavings: number; // USD/year
  improvementAreas: ImprovementArea[];
}

export interface ImprovementArea {
  area: string;
  currentValue: number;
  targetValue: number;
  gap: number;
  priority: Priority;
  estimatedSavings: number;
}

export interface OptimizationOpportunity {
  id: string;
  category: OptimizationCategory;
  title: string;
  description: string;
  impact: Impact;
  implementation: Implementation;
  roi: ROIAnalysis;
  priority: Priority;
}

export type OptimizationCategory =
  | 'fuel_efficiency'
  | 'thermal_efficiency'
  | 'process_control'
  | 'maintenance'
  | 'emissions_reduction'
  | 'waste_heat_recovery';

export interface Impact {
  energySavings: number; // MJ/day
  costSavings: number; // USD/year
  emissionsReduction: number; // tCO2/year
  productionIncrease: number; // %
  qualityImprovement: number; // %
}

export interface Implementation {
  difficulty: 'easy' | 'moderate' | 'complex';
  timeframe: string;
  resources: Resource[];
  risks: Risk[];
  dependencies: string[];
}

export interface Resource {
  type: 'equipment' | 'labor' | 'material' | 'expertise';
  description: string;
  cost: number;
  availability: 'available' | 'limited' | 'unavailable';
}

export interface Risk {
  description: string;
  probability: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  mitigation: string;
}

export interface ROIAnalysis {
  investmentCost: number;
  annualSavings: number;
  paybackPeriod: number; // months
  npv: number; // Net Present Value
  irr: number; // Internal Rate of Return %
}

export interface RootCauseAnalysis {
  id: string;
  issue: string;
  occurredAt: string;
  impact: {
    production: number; // tonnes lost
    cost: number; // USD
    duration: number; // hours
  };
  rootCauses: Cause[];
  correctiveActions: CorrectiveAction[];
  preventiveMeasures: PreventiveMeasure[];
}

export interface Cause {
  category: string;
  description: string;
  contributionFactor: number; // %
  evidence: string[];
}

export interface CorrectiveAction {
  action: string;
  responsible: string;
  deadline: string;
  status: 'pending' | 'in_progress' | 'completed';
  effectiveness?: number; // %
}

export interface PreventiveMeasure {
  measure: string;
  implementation: string;
  expectedImpact: string;
  cost: number;
}

export interface WhatIfScenario {
  id: string;
  name: string;
  description: string;
  parameters: ScenarioParameter[];
  results: ScenarioResults;
}

export interface ScenarioParameter {
  name: string;
  currentValue: number;
  proposedValue: number;
  unit: string;
}

export interface ScenarioResults {
  efficiency: number;
  fuelConsumption: number;
  production: number;
  emissions: number;
  cost: number;
  quality: number;
  feasibility: 'high' | 'medium' | 'low';
  risks: string[];
}

// ============================================================================
// REPORTS
// ============================================================================

export interface Report {
  id: string;
  type: ReportType;
  title: string;
  description: string;
  furnaceId: string;
  period: TimePeriod;
  generatedAt: string;
  generatedBy: string;
  format: ReportFormat;
  sections: ReportSection[];
  metadata: ReportMetadata;
}

export type ReportType =
  | 'performance'
  | 'efficiency'
  | 'maintenance'
  | 'emissions'
  | 'compliance'
  | 'cost'
  | 'custom';

export type ReportFormat = 'pdf' | 'excel' | 'csv' | 'json' | 'html';

export interface ReportSection {
  id: string;
  title: string;
  type: SectionType;
  content: any;
  charts?: ChartConfig[];
}

export type SectionType =
  | 'summary'
  | 'kpi'
  | 'trend'
  | 'comparison'
  | 'table'
  | 'chart'
  | 'text';

export interface ChartConfig {
  type: ChartType;
  title: string;
  data: any;
  options: any;
}

export type ChartType =
  | 'line'
  | 'bar'
  | 'pie'
  | 'scatter'
  | 'heatmap'
  | 'gauge'
  | 'sankey';

export interface ReportMetadata {
  version: string;
  tags: string[];
  recipients: string[];
  schedule?: ReportSchedule;
  filters: ReportFilter[];
}

export interface ReportSchedule {
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  day?: number;
  time: string;
  recipients: string[];
  autoSend: boolean;
}

export interface ReportFilter {
  field: string;
  operator: 'equals' | 'greater' | 'less' | 'between' | 'contains';
  value: any;
}

// ============================================================================
// UI STATE & CONFIGURATION
// ============================================================================

export interface DashboardConfig {
  layout: LayoutConfig;
  theme: ThemeConfig;
  preferences: UserPreferences;
}

export interface LayoutConfig {
  widgets: WidgetConfig[];
  columns: number;
  rowHeight: number;
}

export interface WidgetConfig {
  id: string;
  type: WidgetType;
  title: string;
  position: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  config: any;
  refreshInterval?: number; // seconds
}

export type WidgetType =
  | 'kpi'
  | 'chart'
  | 'gauge'
  | 'alert_feed'
  | 'thermal_map'
  | 'table'
  | 'text';

export interface ThemeConfig {
  mode: 'light' | 'dark';
  primaryColor: string;
  accentColor: string;
  fontSize: 'small' | 'medium' | 'large';
}

export interface UserPreferences {
  language: string;
  timezone: string;
  dateFormat: string;
  numberFormat: string;
  units: UnitPreferences;
  notifications: NotificationPreferences;
}

export interface UnitPreferences {
  temperature: 'celsius' | 'fahrenheit';
  pressure: 'bar' | 'psi' | 'kPa';
  flow: 'm3/h' | 'cfm' | 'l/min';
  energy: 'MJ' | 'kWh' | 'BTU';
  mass: 'kg' | 'lb' | 'tonne';
}

export interface NotificationPreferences {
  enabled: boolean;
  sound: boolean;
  desktop: boolean;
  severityFilter: AlertSeverity[];
  categoryFilter: AlertCategory[];
}

// ============================================================================
// API RESPONSES
// ============================================================================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp: string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
}

export interface PaginatedResponse<T> {
  items: T[];
  pagination: PaginationInfo;
}

export interface PaginationInfo {
  page: number;
  perPage: number;
  total: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// ============================================================================
// WEBSOCKET EVENTS
// ============================================================================

export interface WebSocketMessage {
  type: WebSocketMessageType;
  furnaceId: string;
  timestamp: string;
  data: any;
}

export type WebSocketMessageType =
  | 'performance_update'
  | 'alert'
  | 'sensor_reading'
  | 'status_change'
  | 'maintenance_update'
  | 'configuration_change';

export interface PerformanceUpdate {
  kpis: Partial<PerformanceKPIs>;
  thermal?: Partial<ThermalPerformance>;
  efficiency?: Partial<EfficiencyMetrics>;
}
