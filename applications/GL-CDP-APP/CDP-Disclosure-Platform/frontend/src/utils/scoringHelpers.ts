/**
 * GL-CDP-APP v1.0 - Scoring Helper Utilities
 *
 * Helper functions for CDP scoring level display, color coding,
 * band mapping, and scoring calculations.
 */

import {
  ScoringLevel,
  ScoringBand,
  ScoringCategory,
  SCORING_LEVEL_COLORS,
  SCORING_BAND_COLORS,
  SCORING_CATEGORY_NAMES,
  SCORING_CATEGORY_COLORS,
  SCORING_CATEGORY_WEIGHTS_MANAGEMENT,
  SCORING_CATEGORY_WEIGHTS_LEADERSHIP,
} from '../types';

/** Get display color for a scoring level. */
export function getLevelColor(level: ScoringLevel): string {
  return SCORING_LEVEL_COLORS[level] || '#9e9e9e';
}

/** Get display color for a scoring band. */
export function getBandColor(band: ScoringBand): string {
  return SCORING_BAND_COLORS[band] || '#9e9e9e';
}

/** Get display color for a scoring category. */
export function getCategoryColor(category: ScoringCategory): string {
  return SCORING_CATEGORY_COLORS[category] || '#9e9e9e';
}

/** Get display name for a scoring category. */
export function getCategoryName(category: ScoringCategory): string {
  return SCORING_CATEGORY_NAMES[category] || category;
}

/** Get MUI chip color for a scoring level. */
export function getLevelChipColor(
  level: ScoringLevel,
): 'success' | 'primary' | 'warning' | 'error' {
  switch (level) {
    case ScoringLevel.A:
    case ScoringLevel.A_MINUS:
      return 'success';
    case ScoringLevel.B:
    case ScoringLevel.B_MINUS:
      return 'primary';
    case ScoringLevel.C:
    case ScoringLevel.C_MINUS:
      return 'warning';
    case ScoringLevel.D:
    case ScoringLevel.D_MINUS:
      return 'error';
  }
}

/** Get band label from scoring level. */
export function getBandLabel(level: ScoringLevel): string {
  switch (level) {
    case ScoringLevel.A:
    case ScoringLevel.A_MINUS:
      return 'Leadership';
    case ScoringLevel.B:
    case ScoringLevel.B_MINUS:
      return 'Management';
    case ScoringLevel.C:
    case ScoringLevel.C_MINUS:
      return 'Awareness';
    case ScoringLevel.D:
    case ScoringLevel.D_MINUS:
      return 'Disclosure';
  }
}

/** Get scoring level from numeric score. */
export function scoreToLevel(score: number): ScoringLevel {
  if (score >= 80) return ScoringLevel.A;
  if (score >= 70) return ScoringLevel.A_MINUS;
  if (score >= 60) return ScoringLevel.B;
  if (score >= 50) return ScoringLevel.B_MINUS;
  if (score >= 40) return ScoringLevel.C;
  if (score >= 30) return ScoringLevel.C_MINUS;
  if (score >= 20) return ScoringLevel.D;
  return ScoringLevel.D_MINUS;
}

/** Get numeric score range for a level. */
export function getLevelRange(level: ScoringLevel): { min: number; max: number } {
  switch (level) {
    case ScoringLevel.A: return { min: 80, max: 100 };
    case ScoringLevel.A_MINUS: return { min: 70, max: 79 };
    case ScoringLevel.B: return { min: 60, max: 69 };
    case ScoringLevel.B_MINUS: return { min: 50, max: 59 };
    case ScoringLevel.C: return { min: 40, max: 49 };
    case ScoringLevel.C_MINUS: return { min: 30, max: 39 };
    case ScoringLevel.D: return { min: 20, max: 29 };
    case ScoringLevel.D_MINUS: return { min: 0, max: 19 };
  }
}

/** Get management weight for a category. */
export function getManagementWeight(category: ScoringCategory): number {
  return SCORING_CATEGORY_WEIGHTS_MANAGEMENT[category] || 0;
}

/** Get leadership weight for a category. */
export function getLeadershipWeight(category: ScoringCategory): number {
  return SCORING_CATEGORY_WEIGHTS_LEADERSHIP[category] || 0;
}

/** Format score delta with +/- prefix and color hint. */
export function formatScoreDelta(delta: number): { text: string; isPositive: boolean } {
  const sign = delta > 0 ? '+' : '';
  return {
    text: `${sign}${delta.toFixed(1)} pts`,
    isPositive: delta > 0,
  };
}

/** All 8 scoring levels ordered from best to worst. */
export const LEVEL_ORDER: ScoringLevel[] = [
  ScoringLevel.A,
  ScoringLevel.A_MINUS,
  ScoringLevel.B,
  ScoringLevel.B_MINUS,
  ScoringLevel.C,
  ScoringLevel.C_MINUS,
  ScoringLevel.D,
  ScoringLevel.D_MINUS,
];

/** Gauge segment data for rendering the D- to A score gauge. */
export const GAUGE_SEGMENTS = [
  { level: ScoringLevel.D_MINUS, label: 'D-', min: 0, max: 20, color: '#c62828' },
  { level: ScoringLevel.D, label: 'D', min: 20, max: 30, color: '#b71c1c' },
  { level: ScoringLevel.C_MINUS, label: 'C-', min: 30, max: 40, color: '#ef6c00' },
  { level: ScoringLevel.C, label: 'C', min: 40, max: 50, color: '#e65100' },
  { level: ScoringLevel.B_MINUS, label: 'B-', min: 50, max: 60, color: '#1e88e5' },
  { level: ScoringLevel.B, label: 'B', min: 60, max: 70, color: '#1565c0' },
  { level: ScoringLevel.A_MINUS, label: 'A-', min: 70, max: 80, color: '#2e7d32' },
  { level: ScoringLevel.A, label: 'A', min: 80, max: 100, color: '#1b5e20' },
];
