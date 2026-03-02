/**
 * GL-EUDR-APP Validation Utilities
 *
 * Client-side validators for coordinates, polygons, country codes,
 * commodities, and tax identifiers.
 */

import { EUDRCommodity } from '../types';
import type { GeoCoordinate } from '../types';

// ---------------------------------------------------------------------------
// Coordinate Validation
// ---------------------------------------------------------------------------

export interface ValidationResult {
  valid: boolean;
  message: string;
}

export function validateLatitude(lat: number): ValidationResult {
  if (typeof lat !== 'number' || isNaN(lat)) {
    return { valid: false, message: 'Latitude must be a number' };
  }
  if (lat < -90 || lat > 90) {
    return { valid: false, message: 'Latitude must be between -90 and 90' };
  }
  return { valid: true, message: '' };
}

export function validateLongitude(lon: number): ValidationResult {
  if (typeof lon !== 'number' || isNaN(lon)) {
    return { valid: false, message: 'Longitude must be a number' };
  }
  if (lon < -180 || lon > 180) {
    return { valid: false, message: 'Longitude must be between -180 and 180' };
  }
  return { valid: true, message: '' };
}

export function validateCoordinates(
  lat: number,
  lon: number
): ValidationResult {
  const latResult = validateLatitude(lat);
  if (!latResult.valid) return latResult;

  const lonResult = validateLongitude(lon);
  if (!lonResult.valid) return lonResult;

  return { valid: true, message: '' };
}

// ---------------------------------------------------------------------------
// Polygon Validation
// ---------------------------------------------------------------------------

export function validatePolygon(
  coordinates: GeoCoordinate[]
): ValidationResult {
  if (!Array.isArray(coordinates)) {
    return { valid: false, message: 'Coordinates must be an array' };
  }

  if (coordinates.length < 3) {
    return {
      valid: false,
      message: 'A polygon requires at least 3 coordinate points',
    };
  }

  if (coordinates.length > 10000) {
    return {
      valid: false,
      message: 'Polygon exceeds maximum of 10,000 coordinate points',
    };
  }

  for (let i = 0; i < coordinates.length; i++) {
    const coord = coordinates[i];
    const result = validateCoordinates(coord.latitude, coord.longitude);
    if (!result.valid) {
      return {
        valid: false,
        message: `Point ${i + 1}: ${result.message}`,
      };
    }
  }

  // Check ring closure (first and last point should match for a closed polygon)
  const first = coordinates[0];
  const last = coordinates[coordinates.length - 1];
  if (first.latitude !== last.latitude || first.longitude !== last.longitude) {
    return {
      valid: false,
      message:
        'Polygon ring is not closed. First and last coordinates must match.',
    };
  }

  return { valid: true, message: '' };
}

// ---------------------------------------------------------------------------
// Country Code Validation (ISO 3166-1 alpha-2)
// ---------------------------------------------------------------------------

const VALID_COUNTRY_CODES = new Set([
  'AF','AL','DZ','AS','AD','AO','AG','AR','AM','AU','AT','AZ','BS','BH','BD',
  'BB','BY','BE','BZ','BJ','BT','BO','BA','BW','BR','BN','BG','BF','BI','CV',
  'KH','CM','CA','CF','TD','CL','CN','CO','KM','CG','CD','CR','CI','HR','CU',
  'CY','CZ','DK','DJ','DM','DO','EC','EG','SV','GQ','ER','EE','SZ','ET','FJ',
  'FI','FR','GA','GM','GE','DE','GH','GR','GD','GT','GN','GW','GY','HT','HN',
  'HU','IS','IN','ID','IR','IQ','IE','IL','IT','JM','JP','JO','KZ','KE','KI',
  'KP','KR','KW','KG','LA','LV','LB','LS','LR','LY','LI','LT','LU','MG','MW',
  'MY','MV','ML','MT','MH','MR','MU','MX','FM','MD','MC','MN','ME','MA','MZ',
  'MM','NA','NR','NP','NL','NZ','NI','NE','NG','MK','NO','OM','PK','PW','PA',
  'PG','PY','PE','PH','PL','PT','QA','RO','RU','RW','KN','LC','VC','WS','SM',
  'ST','SA','SN','RS','SC','SL','SG','SK','SI','SB','SO','ZA','SS','ES','LK',
  'SD','SR','SE','CH','SY','TW','TJ','TZ','TH','TL','TG','TO','TT','TN','TR',
  'TM','TV','UG','UA','AE','GB','US','UY','UZ','VU','VE','VN','YE','ZM','ZW',
]);

export function validateCountryCode(code: string): ValidationResult {
  if (!code || typeof code !== 'string') {
    return { valid: false, message: 'Country code is required' };
  }

  const normalized = code.trim().toUpperCase();

  if (normalized.length !== 2) {
    return {
      valid: false,
      message: 'Country code must be a 2-letter ISO 3166-1 alpha-2 code',
    };
  }

  if (!VALID_COUNTRY_CODES.has(normalized)) {
    return {
      valid: false,
      message: `"${normalized}" is not a recognized ISO 3166-1 country code`,
    };
  }

  return { valid: true, message: '' };
}

// ---------------------------------------------------------------------------
// Commodity Validation
// ---------------------------------------------------------------------------

const VALID_COMMODITIES = new Set(Object.values(EUDRCommodity));

export function validateCommodity(commodity: string): ValidationResult {
  if (!commodity) {
    return { valid: false, message: 'Commodity is required' };
  }

  if (!VALID_COMMODITIES.has(commodity as EUDRCommodity)) {
    return {
      valid: false,
      message: `"${commodity}" is not a valid EUDR commodity. Must be one of: ${Array.from(VALID_COMMODITIES).join(', ')}`,
    };
  }

  return { valid: true, message: '' };
}

// ---------------------------------------------------------------------------
// Tax ID / EORI Validation
// ---------------------------------------------------------------------------

export function validateTaxId(taxId: string): ValidationResult {
  if (!taxId || typeof taxId !== 'string') {
    return { valid: false, message: 'Tax ID is required' };
  }

  const trimmed = taxId.trim();

  if (trimmed.length < 5) {
    return { valid: false, message: 'Tax ID must be at least 5 characters' };
  }

  if (trimmed.length > 30) {
    return { valid: false, message: 'Tax ID must not exceed 30 characters' };
  }

  // Basic alphanumeric + common separators
  if (!/^[A-Za-z0-9\-./]+$/.test(trimmed)) {
    return {
      valid: false,
      message: 'Tax ID contains invalid characters',
    };
  }

  return { valid: true, message: '' };
}

export function validateEORI(eori: string): ValidationResult {
  if (!eori || typeof eori !== 'string') {
    return { valid: false, message: 'EORI number is required' };
  }

  const trimmed = eori.trim().toUpperCase();

  // EORI format: 2-letter country code + up to 15 alphanumeric characters
  if (!/^[A-Z]{2}[A-Z0-9]{1,15}$/.test(trimmed)) {
    return {
      valid: false,
      message:
        'EORI must start with a 2-letter country code followed by up to 15 alphanumeric characters',
    };
  }

  const countryPart = trimmed.substring(0, 2);
  if (!VALID_COUNTRY_CODES.has(countryPart)) {
    return {
      valid: false,
      message: `EORI country prefix "${countryPart}" is not a valid country code`,
    };
  }

  return { valid: true, message: '' };
}
