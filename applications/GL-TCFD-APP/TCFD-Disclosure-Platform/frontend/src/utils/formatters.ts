/**
 * Formatters - Number, currency, percentage, and date formatting utilities.
 */

import { format, parseISO, isValid } from 'date-fns';
import type { Currency } from '../types';

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

export function formatRiskLevel(level: string): string {
  return level.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatTimeHorizon(horizon: string): string {
  const map: Record<string, string> = {
    short_term: 'Short-term (0-3 years)',
    medium_term: 'Medium-term (3-10 years)',
    long_term: 'Long-term (10+ years)',
  };
  return map[horizon] || horizon;
}

export function formatScenarioType(type: string): string {
  return type.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trimEnd() + '...';
}
