/**
 * Validators - Form validation helper functions.
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

export function validateScenarioParameters(params: { name: string; value: number; min: number; max: number }[]): Record<string, string> {
  const errors: Record<string, string> = {};
  for (const param of params) {
    if (param.value < param.min || param.value > param.max) {
      errors[param.name] = `${param.name} must be between ${param.min} and ${param.max}`;
    }
  }
  return errors;
}
