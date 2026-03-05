/**
 * Validation utilities for EU Taxonomy data inputs.
 */

/**
 * Validate NACE code format (letter + 2-4 digits with optional dots).
 * Examples: A01, C23.1, D35.11
 */
export const validateNACECode = (code: string): { valid: boolean; error?: string } => {
  if (!code || code.trim().length === 0) {
    return { valid: false, error: 'NACE code is required' };
  }
  const naceRegex = /^[A-U]\d{2}(\.\d{1,2})?$/;
  if (!naceRegex.test(code.trim().toUpperCase())) {
    return { valid: false, error: 'Invalid NACE code format (e.g., C23.1, D35.11)' };
  }
  return { valid: true };
};

/**
 * Validate KPI financial data.
 */
export const validateKPIData = (data: {
  turnover?: number;
  capex?: number;
  opex?: number;
}): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  if (data.turnover !== undefined && data.turnover < 0) {
    errors.push('Turnover must be non-negative');
  }
  if (data.capex !== undefined && data.capex < 0) {
    errors.push('CapEx must be non-negative');
  }
  if (data.opex !== undefined && data.opex < 0) {
    errors.push('OpEx must be non-negative');
  }
  return { valid: errors.length === 0, errors };
};

/**
 * Validate exposure amount and type for GAR calculation.
 */
export const validateExposure = (data: {
  amount?: number;
  type?: string;
  counterparty_lei?: string;
}): { valid: boolean; errors: string[] } => {
  const errors: string[] = [];
  if (data.amount !== undefined && data.amount < 0) {
    errors.push('Exposure amount must be non-negative');
  }
  if (!data.type) {
    errors.push('Exposure type is required');
  }
  if (data.counterparty_lei && !/^[A-Z0-9]{20}$/.test(data.counterparty_lei)) {
    errors.push('Invalid LEI format (must be 20 alphanumeric characters)');
  }
  return { valid: errors.length === 0, errors };
};

/**
 * Validate LEI code format.
 */
export const validateLEI = (lei: string): { valid: boolean; error?: string } => {
  if (!lei || lei.trim().length === 0) {
    return { valid: false, error: 'LEI code is required' };
  }
  if (!/^[A-Z0-9]{20}$/.test(lei.trim())) {
    return { valid: false, error: 'LEI must be exactly 20 alphanumeric characters' };
  }
  return { valid: true };
};

/**
 * Validate percentage value (0-100 or 0-1 depending on format).
 */
export const validatePercentage = (
  value: number,
  format: 'ratio' | 'percent' = 'percent'
): { valid: boolean; error?: string } => {
  const max = format === 'ratio' ? 1 : 100;
  if (value < 0 || value > max) {
    return { valid: false, error: `Value must be between 0 and ${max}` };
  }
  return { valid: true };
};

/**
 * Validate reporting period format (YYYY or YYYY-MM).
 */
export const validateReportingPeriod = (period: string): { valid: boolean; error?: string } => {
  if (!period) {
    return { valid: false, error: 'Reporting period is required' };
  }
  const yearRegex = /^\d{4}$/;
  const monthRegex = /^\d{4}-(0[1-9]|1[0-2])$/;
  if (!yearRegex.test(period) && !monthRegex.test(period)) {
    return { valid: false, error: 'Invalid period format (use YYYY or YYYY-MM)' };
  }
  return { valid: true };
};

/**
 * Validate EPC rating (A-G scale).
 */
export const validateEPCRating = (rating: string): { valid: boolean; error?: string } => {
  const validRatings = ['A', 'B', 'C', 'D', 'E', 'F', 'G'];
  if (!validRatings.includes(rating.toUpperCase())) {
    return { valid: false, error: 'EPC rating must be A through G' };
  }
  return { valid: true };
};

/**
 * Validate CO2 emissions value for auto loan alignment.
 */
export const validateCO2Emissions = (
  gramsPerKm: number
): { valid: boolean; error?: string } => {
  if (gramsPerKm < 0) {
    return { valid: false, error: 'CO2 emissions must be non-negative' };
  }
  if (gramsPerKm > 500) {
    return { valid: false, error: 'CO2 emissions value seems unreasonably high (>500 g/km)' };
  }
  return { valid: true };
};
