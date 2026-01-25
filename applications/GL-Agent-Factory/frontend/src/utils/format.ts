import { format, formatDistanceToNow, parseISO } from 'date-fns';

/**
 * Format a number with thousand separators and optional decimal places
 */
export function formatNumber(value: number, decimals = 0): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

/**
 * Format a number as currency
 */
export function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(value);
}

/**
 * Format a number as percentage
 */
export function formatPercentage(value: number, decimals = 1): string {
  return `${formatNumber(value, decimals)}%`;
}

/**
 * Format emissions value with unit (tCO2e)
 */
export function formatEmissions(value: number, decimals = 2): string {
  if (value >= 1000000) {
    return `${formatNumber(value / 1000000, decimals)} MtCO2e`;
  }
  if (value >= 1000) {
    return `${formatNumber(value / 1000, decimals)} ktCO2e`;
  }
  return `${formatNumber(value, decimals)} tCO2e`;
}

/**
 * Format weight in metric tons
 */
export function formatWeight(value: number, decimals = 2): string {
  if (value >= 1000000) {
    return `${formatNumber(value / 1000000, decimals)} Mt`;
  }
  if (value >= 1000) {
    return `${formatNumber(value / 1000, decimals)} kt`;
  }
  return `${formatNumber(value, decimals)} t`;
}

/**
 * Format date to human readable format
 */
export function formatDate(date: string | Date, formatStr = 'MMM d, yyyy'): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, formatStr);
}

/**
 * Format date with time
 */
export function formatDateTime(date: string | Date): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return format(dateObj, 'MMM d, yyyy HH:mm');
}

/**
 * Format relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(date: string | Date): string {
  const dateObj = typeof date === 'string' ? parseISO(date) : date;
  return formatDistanceToNow(dateObj, { addSuffix: true });
}

/**
 * Format file size in bytes to human readable format
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Truncate text with ellipsis
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength - 3) + '...';
}

/**
 * Format country code to flag emoji
 */
export function countryCodeToFlag(countryCode: string): string {
  const codePoints = countryCode
    .toUpperCase()
    .split('')
    .map((char) => 127397 + char.charCodeAt(0));
  return String.fromCodePoint(...codePoints);
}
