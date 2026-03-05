/**
 * Validators - Form validation helper functions for SBTi target setting.
 */

export function isRequired(value: unknown): string | null {
  if (value === null || value === undefined || value === '') return 'This field is required';
  if (typeof value === 'string' && value.trim() === '') return 'This field is required';
  return null;
}

export function isPositiveNumber(value: number | string): string | null {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(num)) return 'Must be a valid number';
  if (num < 0) return 'Must be a positive number';
  return null;
}

export function isPercentage(value: number | string): string | null {
  const num = typeof value === 'string' ? parseFloat(value) : value;
  if (isNaN(num)) return 'Must be a valid number';
  if (num < 0 || num > 100) return 'Must be between 0 and 100';
  return null;
}

export function isInRange(value: number, min: number, max: number): string | null {
  if (value < min || value > max) return `Must be between ${min} and ${max}`;
  return null;
}

export function isValidYear(value: number | string): string | null {
  const num = typeof value === 'string' ? parseInt(value, 10) : value;
  if (isNaN(num)) return 'Must be a valid year';
  if (num < 1990 || num > 2100) return 'Year must be between 1990 and 2100';
  return null;
}

export function isValidBaseYear(value: number | string): string | null {
  const yearError = isValidYear(value);
  if (yearError) return yearError;
  const num = typeof value === 'string' ? parseInt(value, 10) : value;
  if (num > new Date().getFullYear()) return 'Base year cannot be in the future';
  if (num < 2015) return 'SBTi requires base year no earlier than 2015';
  return null;
}

export function isValidTargetYear(value: number | string, baseYear: number): string | null {
  const yearError = isValidYear(value);
  if (yearError) return yearError;
  const num = typeof value === 'string' ? parseInt(value, 10) : value;
  if (num <= baseYear) return 'Target year must be after base year';
  return null;
}

export function isValidNearTermTargetYear(value: number | string, baseYear: number): string | null {
  const yearError = isValidTargetYear(value, baseYear);
  if (yearError) return yearError;
  const num = typeof value === 'string' ? parseInt(value, 10) : value;
  const yearsFromBase = num - baseYear;
  if (yearsFromBase < 5) return 'Near-term targets must be at least 5 years from base year';
  if (yearsFromBase > 10) return 'Near-term targets must be no more than 10 years from base year';
  return null;
}

export function isValidLongTermTargetYear(value: number | string): string | null {
  const yearError = isValidYear(value);
  if (yearError) return yearError;
  const num = typeof value === 'string' ? parseInt(value, 10) : value;
  if (num > 2050) return 'Long-term targets should be by 2050 at the latest';
  return null;
}

export function isValidCoverage(value: number, scope: string): string | null {
  const pctError = isPercentage(value);
  if (pctError) return pctError;
  if (scope === 'scope_1_2' && value < 95) {
    return 'Scope 1+2 targets must cover at least 95% of emissions';
  }
  if (scope === 'scope_3' && value < 67) {
    return 'Scope 3 targets must cover at least 67% of emissions';
  }
  return null;
}

export function isValidReductionRate(rate: number, alignment: string): string | null {
  if (rate <= 0) return 'Reduction rate must be positive';
  if (alignment === '1.5C' && rate < 4.2) {
    return 'For 1.5C alignment, annual reduction must be at least 4.2%';
  }
  if (alignment === 'well_below_2C' && rate < 2.5) {
    return 'For well-below 2C, annual reduction must be at least 2.5%';
  }
  return null;
}

export function isValidDate(value: string): string | null {
  if (!value) return 'Date is required';
  const date = new Date(value);
  if (isNaN(date.getTime())) return 'Must be a valid date';
  return null;
}

export function isNotEmpty<T>(arr: T[]): string | null {
  if (!arr || arr.length === 0) return 'At least one item is required';
  return null;
}

export function maxLength(value: string, max: number): string | null {
  if (value && value.length > max) return `Must be ${max} characters or less`;
  return null;
}

export function minLength(value: string, min: number): string | null {
  if (value && value.length < min) return `Must be at least ${min} characters`;
  return null;
}

export function composeValidators(...validators: ((value: unknown) => string | null)[]): (value: unknown) => string | null {
  return (value: unknown) => {
    for (const validator of validators) {
      const error = validator(value);
      if (error) return error;
    }
    return null;
  };
}

export function validateTargetForm(data: {
  name: string;
  base_year: number;
  target_year: number;
  target_reduction_pct: number;
  scope_coverage_pct: number;
  target_scope: string;
  target_timeframe: string;
}): Record<string, string> {
  const errors: Record<string, string> = {};

  const nameErr = isRequired(data.name);
  if (nameErr) errors.name = nameErr;

  const baseErr = isValidBaseYear(data.base_year);
  if (baseErr) errors.base_year = baseErr;

  if (data.target_timeframe === 'near_term') {
    const targetErr = isValidNearTermTargetYear(data.target_year, data.base_year);
    if (targetErr) errors.target_year = targetErr;
  } else {
    const targetErr = isValidLongTermTargetYear(data.target_year);
    if (targetErr) errors.target_year = targetErr;
  }

  const reductionErr = isPercentage(data.target_reduction_pct);
  if (reductionErr) errors.target_reduction_pct = reductionErr;

  const coverageErr = isValidCoverage(data.scope_coverage_pct, data.target_scope);
  if (coverageErr) errors.scope_coverage_pct = coverageErr;

  return errors;
}
