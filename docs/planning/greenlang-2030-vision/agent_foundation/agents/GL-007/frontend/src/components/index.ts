/**
 * GL-007 Frontend Components Index
 *
 * Central export point for all components
 */

// Chart Components
export { default as KPICard } from './charts/KPICard';
export { default as GaugeChart } from './charts/GaugeChart';

// Dashboard Components
export { default as ExecutiveDashboard } from './dashboards/ExecutiveDashboard';
export { default as OperationsDashboard } from './dashboards/OperationsDashboard';
export { default as ThermalProfilingView } from './dashboards/ThermalProfilingView';

// Re-export types
export type { KPICardProps } from './charts/KPICard';
export type { GaugeChartProps } from './charts/GaugeChart';
