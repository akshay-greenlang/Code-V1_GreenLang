/**
 * Pathway Helpers - Calculation helpers and chart data transformers for SBTi pathways.
 */

import type { PathwayMilestone, PathwayComparison, ProgressRecord, PathwayAlignment } from '../types';

/**
 * Calculate annual linear reduction rate between two emission points.
 */
export function calculateAnnualReductionRate(
  baseEmissions: number,
  targetEmissions: number,
  years: number
): number {
  if (years <= 0 || baseEmissions <= 0) return 0;
  const totalReduction = (baseEmissions - targetEmissions) / baseEmissions;
  return (totalReduction / years) * 100;
}

/**
 * Generate a linear pathway from base to target year.
 */
export function generateLinearPathway(
  baseYear: number,
  targetYear: number,
  baseEmissions: number,
  targetEmissions: number
): { year: number; emissions: number }[] {
  const points: { year: number; emissions: number }[] = [];
  const years = targetYear - baseYear;
  if (years <= 0) return points;

  const annualReduction = (baseEmissions - targetEmissions) / years;
  for (let i = 0; i <= years; i++) {
    points.push({
      year: baseYear + i,
      emissions: baseEmissions - annualReduction * i,
    });
  }
  return points;
}

/**
 * Calculate the SBTi minimum ambition for near-term targets (4.2% per year for 1.5C).
 */
export function getMinimumAmbitionRate(alignment: PathwayAlignment): number {
  switch (alignment) {
    case '1.5C':
      return 4.2;
    case 'well_below_2C':
      return 2.5;
    case '2C':
      return 1.23;
    default:
      return 0;
  }
}

/**
 * Check if a reduction rate meets the minimum ambition level.
 */
export function meetsMinimumAmbition(
  annualReductionRate: number,
  alignment: PathwayAlignment
): boolean {
  return annualReductionRate >= getMinimumAmbitionRate(alignment);
}

/**
 * Transform pathway milestones and actual progress records into chart-ready data.
 */
export function buildPathwayChartData(
  milestones: PathwayMilestone[],
  progressRecords: ProgressRecord[],
  baseYear: number,
  baseEmissions: number
): { year: number; pathway: number; actual: number | null; target: number | null }[] {
  const dataMap = new Map<number, { pathway: number; actual: number | null }>();

  for (const ms of milestones) {
    dataMap.set(ms.year, {
      pathway: ms.expected_emissions,
      actual: ms.actual_emissions,
    });
  }

  for (const rec of progressRecords) {
    const existing = dataMap.get(rec.reporting_year);
    if (existing) {
      existing.actual = rec.actual_emissions;
    } else {
      dataMap.set(rec.reporting_year, {
        pathway: rec.expected_emissions,
        actual: rec.actual_emissions,
      });
    }
  }

  const years = Array.from(dataMap.keys()).sort((a, b) => a - b);
  return years.map((year) => {
    const entry = dataMap.get(year)!;
    return {
      year,
      pathway: entry.pathway,
      actual: entry.actual,
      target: milestones.find((m) => m.year === year)?.expected_emissions ?? null,
    };
  });
}

/**
 * Transform pathway comparisons into multi-line chart data.
 */
export function buildComparisonChartData(
  comparisons: PathwayComparison[]
): { year: number; [key: string]: number | null }[] {
  if (comparisons.length === 0) return [];

  const allYears = new Set<number>();
  for (const comp of comparisons) {
    for (const ms of comp.milestones) {
      allYears.add(ms.year);
    }
  }

  const sortedYears = Array.from(allYears).sort((a, b) => a - b);
  return sortedYears.map((year) => {
    const point: { year: number; [key: string]: number | null } = { year };
    for (const comp of comparisons) {
      const ms = comp.milestones.find((m) => m.year === year);
      point[comp.pathway_name] = ms ? ms.emissions : null;
    }
    return point;
  });
}

/**
 * Calculate cumulative emissions budget remaining.
 */
export function calculateBudgetRemaining(
  milestones: PathwayMilestone[],
  currentYear: number
): number {
  const futureMilestones = milestones.filter((m) => m.year > currentYear);
  return futureMilestones.reduce((sum, m) => sum + m.cumulative_budget, 0);
}

/**
 * Determine the projected year to reach target based on current trend.
 */
export function projectTargetYear(
  progressRecords: ProgressRecord[],
  targetEmissions: number
): number | null {
  if (progressRecords.length < 2) return null;

  const sorted = [...progressRecords].sort((a, b) => a.reporting_year - b.reporting_year);
  const lastTwo = sorted.slice(-2);
  const annualChange = lastTwo[1].actual_emissions - lastTwo[0].actual_emissions;

  if (annualChange >= 0) return null; // Emissions not decreasing

  const currentEmissions = lastTwo[1].actual_emissions;
  const yearsNeeded = (currentEmissions - targetEmissions) / Math.abs(annualChange);
  return Math.ceil(lastTwo[1].reporting_year + yearsNeeded);
}

/**
 * Calculate the temperature alignment color band.
 */
export function getTemperatureColor(temp: number): string {
  if (temp <= 1.5) return '#1B5E20';
  if (temp <= 2.0) return '#2E7D32';
  if (temp <= 2.5) return '#EF6C00';
  if (temp <= 3.0) return '#E65100';
  return '#B71C1C';
}

/**
 * Get RAG status color.
 */
export function getRAGColor(status: string): string {
  switch (status) {
    case 'on_track': return '#2E7D32';
    case 'at_risk': return '#EF6C00';
    case 'off_track': return '#B71C1C';
    default: return '#9E9E9E';
  }
}
