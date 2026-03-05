/**
 * Formatters - Number, currency, percentage, date, emissions, and SBTi-specific formatting utilities.
 */

import { format, parseISO, isValid, differenceInDays } from 'date-fns';
import type { Currency, RAGStatus, PathwayAlignment, TargetStatus, TargetMethod } from '../types';

const CURRENCY_SYMBOLS: Record<Currency, string> = {
  USD: '$',
  EUR: '\u20AC',
  GBP: '\u00A3',
  JPY: '\u00A5',
  AUD: 'A$',
  CAD: 'C$',
  CHF: 'CHF',
};

export function formatCurrency(value: number, currency: Currency = 'USD', compact = false): string {
  if (compact) {
    return `${CURRENCY_SYMBOLS[currency]}${formatCompactNumber(value)}`;
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatNumber(value: number, decimals = 0): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatCompactNumber(value: number): string {
  const absValue = Math.abs(value);
  const sign = value < 0 ? '-' : '';
  if (absValue >= 1_000_000_000) return `${sign}${(absValue / 1_000_000_000).toFixed(1)}B`;
  if (absValue >= 1_000_000) return `${sign}${(absValue / 1_000_000).toFixed(1)}M`;
  if (absValue >= 1_000) return `${sign}${(absValue / 1_000).toFixed(1)}K`;
  return `${sign}${absValue.toFixed(0)}`;
}

export function formatPercentage(value: number, decimals = 1): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
}

export function formatPercentageAbs(value: number, decimals = 1): string {
  return `${value.toFixed(decimals)}%`;
}

export function formatDate(dateStr: string, formatStr = 'MMM dd, yyyy'): string {
  if (!dateStr) return '-';
  const parsed = parseISO(dateStr);
  return isValid(parsed) ? format(parsed, formatStr) : dateStr;
}

export function formatDateTime(dateStr: string): string {
  return formatDate(dateStr, 'MMM dd, yyyy HH:mm');
}

export function formatEmissions(value: number, unit = 'tCO2e'): string {
  return `${formatCompactNumber(value)} ${unit}`;
}

export function formatTemperature(value: number): string {
  return `${value.toFixed(2)}\u00B0C`;
}

export function formatReductionRate(value: number): string {
  return `${value.toFixed(2)}% per year`;
}

export function formatDaysRemaining(dateStr: string): string {
  if (!dateStr) return '-';
  const parsed = parseISO(dateStr);
  if (!isValid(parsed)) return dateStr;
  const days = differenceInDays(parsed, new Date());
  if (days < 0) return `${Math.abs(days)} days overdue`;
  if (days === 0) return 'Due today';
  if (days === 1) return '1 day remaining';
  return `${days} days remaining`;
}

export function formatRAGStatus(status: RAGStatus): string {
  const map: Record<RAGStatus, string> = {
    on_track: 'On Track',
    at_risk: 'At Risk',
    off_track: 'Off Track',
  };
  return map[status] || status;
}

export function formatPathwayAlignment(alignment: PathwayAlignment): string {
  const map: Record<PathwayAlignment, string> = {
    '1.5C': '1.5\u00B0C Aligned',
    well_below_2C: 'Well Below 2\u00B0C',
    '2C': '2\u00B0C Aligned',
    above_2C: 'Above 2\u00B0C',
    not_aligned: 'Not Aligned',
  };
  return map[alignment] || alignment;
}

export function formatTargetStatus(status: TargetStatus): string {
  return status.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatTargetMethod(method: TargetMethod): string {
  const map: Record<TargetMethod, string> = {
    cross_sector_aca: 'Cross-Sector (ACA)',
    sector_specific_sda: 'Sector-Specific (SDA)',
    portfolio_coverage: 'Portfolio Coverage',
    sectoral_decarbonization: 'Sectoral Decarbonization',
    temperature_rating: 'Temperature Rating',
    engagement_threshold: 'Engagement Threshold',
  };
  return map[method] || method;
}

export function formatSector(sector: string): string {
  return sector.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trimEnd() + '...';
}
