/**
 * Scenario Helpers - Calculation and transformation utilities for scenario analysis.
 */

import type { ScenarioResult, ScenarioDefinition, SensitivityResult, StrandingDataPoint } from '../types';

export function calculateNetImpact(result: ScenarioResult): number {
  return result.revenue_impact + result.cost_impact + result.asset_impairment + result.opportunity_value;
}

export function calculateImpactRange(results: ScenarioResult[]): { low: number; high: number; median: number } {
  if (results.length === 0) return { low: 0, high: 0, median: 0 };
  const impacts = results.map(calculateNetImpact).sort((a, b) => a - b);
  return {
    low: impacts[0],
    high: impacts[impacts.length - 1],
    median: impacts[Math.floor(impacts.length / 2)],
  };
}

export function getScenarioColor(scenarioType: string): string {
  const colors: Record<string, string> = {
    orderly_transition: '#2E7D32',
    net_zero_2050: '#1B5E20',
    disorderly_transition: '#E65100',
    delayed_transition: '#F57F17',
    hot_house: '#B71C1C',
    current_policies: '#757575',
    custom: '#0D47A1',
  };
  return colors[scenarioType] || '#9E9E9E';
}

export function getTemperatureColor(target: string): string {
  const colors: Record<string, string> = {
    '1.5C': '#1B5E20',
    '2.0C': '#F57F17',
    '2.5C': '#E65100',
    '3.0C': '#C62828',
    '4.0C': '#B71C1C',
  };
  return colors[target] || '#9E9E9E';
}

export function interpolateLinear(
  x: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  if (x2 === x1) return y1;
  return y1 + ((y2 - y1) * (x - x1)) / (x2 - x1);
}

export function buildWaterfallData(result: ScenarioResult): { name: string; value: number; cumulative: number }[] {
  const items = [
    { name: 'Revenue Impact', value: result.revenue_impact },
    { name: 'Cost Impact', value: result.cost_impact },
    { name: 'Asset Impairment', value: result.asset_impairment },
    { name: 'Carbon Cost', value: -result.carbon_cost },
    { name: 'CapEx Required', value: -result.capex_required },
    { name: 'Opportunity Value', value: result.opportunity_value },
  ];

  let cumulative = 0;
  return items.map((item) => {
    cumulative += item.value;
    return { ...item, cumulative };
  });
}

export function groupResultsByScenario(
  results: ScenarioResult[]
): Record<string, ScenarioResult[]> {
  const grouped: Record<string, ScenarioResult[]> = {};
  for (const result of results) {
    if (!grouped[result.scenario_name]) {
      grouped[result.scenario_name] = [];
    }
    grouped[result.scenario_name].push(result);
  }
  return grouped;
}

export function getStrandingYearThreshold(
  data: StrandingDataPoint[],
  scenarioName: string,
  threshold: number
): number | null {
  const scenarioData = data
    .filter((d) => d.scenario_name === scenarioName)
    .sort((a, b) => a.year - b.year);

  for (const point of scenarioData) {
    if (point.percentage_at_risk >= threshold) return point.year;
  }
  return null;
}

export function buildSensitivityBars(results: SensitivityResult[]): {
  name: string;
  lowDelta: number;
  highDelta: number;
  range: number;
}[] {
  return results
    .map((r) => ({
      name: r.parameter_name,
      lowDelta: r.low_impact,
      highDelta: r.high_impact,
      range: Math.abs(r.high_impact - r.low_impact),
    }))
    .sort((a, b) => b.range - a.range);
}

export function carbonPriceImpact(
  emissionsTonnes: number,
  carbonPrice: number,
  abatementFraction: number = 0
): number {
  const effectiveEmissions = emissionsTonnes * (1 - abatementFraction);
  return effectiveEmissions * carbonPrice;
}

export function scenarioToCSV(scenarios: ScenarioDefinition[], results: ScenarioResult[]): string {
  const headers = [
    'Scenario', 'Year', 'Revenue Impact', 'Cost Impact', 'Asset Impairment',
    'CapEx Required', 'Carbon Cost', 'Opportunity Value', 'Net Impact',
  ];
  const rows = results.map((r) => [
    r.scenario_name, r.year, r.revenue_impact, r.cost_impact, r.asset_impairment,
    r.capex_required, r.carbon_cost, r.opportunity_value, r.net_financial_impact,
  ]);
  return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
}
