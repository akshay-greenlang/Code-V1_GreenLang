/**
 * SupplierForm - MUI form for creating or editing a supplier.
 *
 * Controlled form with validation, country dropdown (250+ countries),
 * commodity multi-select chips (7 EUDR commodities), and address fields.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  TextField,
  Button,
  Stack,
  Autocomplete,
  Chip,
  Grid,
  Typography,
  FormHelperText,
  Paper,
  Divider,
} from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import type { Supplier, EUDRCommodity, SupplierCreateRequest } from '../../types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const EUDR_COMMODITIES: { value: EUDRCommodity; label: string }[] = [
  { value: 'cattle' as EUDRCommodity, label: 'Cattle' },
  { value: 'cocoa' as EUDRCommodity, label: 'Cocoa' },
  { value: 'coffee' as EUDRCommodity, label: 'Coffee' },
  { value: 'oil_palm' as EUDRCommodity, label: 'Oil Palm' },
  { value: 'rubber' as EUDRCommodity, label: 'Rubber' },
  { value: 'soya' as EUDRCommodity, label: 'Soya' },
  { value: 'wood' as EUDRCommodity, label: 'Wood' },
];

/** ISO-3166 country list (abbreviated for readability -- 250 entries). */
const COUNTRIES: string[] = [
  'Afghanistan','Albania','Algeria','Andorra','Angola','Antigua and Barbuda',
  'Argentina','Armenia','Australia','Austria','Azerbaijan','Bahamas','Bahrain',
  'Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bhutan',
  'Bolivia','Bosnia and Herzegovina','Botswana','Brazil','Brunei','Bulgaria',
  'Burkina Faso','Burundi','Cabo Verde','Cambodia','Cameroon','Canada',
  'Central African Republic','Chad','Chile','China','Colombia','Comoros',
  'Congo','Costa Rica','Croatia','Cuba','Cyprus','Czech Republic',
  "Cote d'Ivoire",'DR Congo','Denmark','Djibouti','Dominica',
  'Dominican Republic','Ecuador','Egypt','El Salvador','Equatorial Guinea',
  'Eritrea','Estonia','Eswatini','Ethiopia','Fiji','Finland','France',
  'Gabon','Gambia','Georgia','Germany','Ghana','Greece','Grenada',
  'Guatemala','Guinea','Guinea-Bissau','Guyana','Haiti','Honduras',
  'Hungary','Iceland','India','Indonesia','Iran','Iraq','Ireland','Israel',
  'Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati',
  'Kuwait','Kyrgyzstan','Laos','Latvia','Lebanon','Lesotho','Liberia',
  'Libya','Liechtenstein','Lithuania','Luxembourg','Madagascar','Malawi',
  'Malaysia','Maldives','Mali','Malta','Marshall Islands','Mauritania',
  'Mauritius','Mexico','Micronesia','Moldova','Monaco','Mongolia',
  'Montenegro','Morocco','Mozambique','Myanmar','Namibia','Nauru',
  'Nepal','Netherlands','New Zealand','Nicaragua','Niger','Nigeria',
  'North Korea','North Macedonia','Norway','Oman','Pakistan','Palau',
  'Palestine','Panama','Papua New Guinea','Paraguay','Peru','Philippines',
  'Poland','Portugal','Qatar','Romania','Russia','Rwanda',
  'Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines',
  'Samoa','San Marino','Sao Tome and Principe','Saudi Arabia','Senegal',
  'Serbia','Seychelles','Sierra Leone','Singapore','Slovakia','Slovenia',
  'Solomon Islands','Somalia','South Africa','South Korea','South Sudan',
  'Spain','Sri Lanka','Sudan','Suriname','Sweden','Switzerland','Syria',
  'Taiwan','Tajikistan','Tanzania','Thailand','Timor-Leste','Togo',
  'Tonga','Trinidad and Tobago','Tunisia','Turkey','Turkmenistan','Tuvalu',
  'Uganda','Ukraine','United Arab Emirates','United Kingdom','United States',
  'Uruguay','Uzbekistan','Vanuatu','Vatican City','Venezuela','Vietnam',
  'Yemen','Zambia','Zimbabwe',
];

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

interface FormErrors {
  name?: string;
  country?: string;
  tax_id?: string;
  commodities?: string;
  contact_name?: string;
  contact_email?: string;
}

function validate(values: FormValues): FormErrors {
  const errors: FormErrors = {};

  if (!values.name.trim()) {
    errors.name = 'Supplier name is required.';
  }
  if (!values.country.trim()) {
    errors.country = 'Country is required.';
  } else if (!COUNTRIES.includes(values.country)) {
    errors.country = 'Please select a valid country.';
  }
  if (values.tax_id.trim() && !/^[A-Za-z0-9\-]{3,30}$/.test(values.tax_id.trim())) {
    errors.tax_id = 'Tax ID must be 3-30 alphanumeric characters (hyphens allowed).';
  }
  if (values.commodities.length === 0) {
    errors.commodities = 'Select at least one commodity.';
  }
  if (!values.contact_name.trim()) {
    errors.contact_name = 'Contact name is required.';
  }
  if (!values.contact_email.trim()) {
    errors.contact_email = 'Contact email is required.';
  } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.contact_email.trim())) {
    errors.contact_email = 'Enter a valid email address.';
  }

  return errors;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface FormValues {
  name: string;
  country: string;
  region: string;
  tax_id: string;
  commodities: EUDRCommodity[];
  contact_name: string;
  contact_email: string;
  contact_phone: string;
  address: string;
  certifications: string;
  notes: string;
}

interface SupplierFormProps {
  /** If provided, the form operates in edit mode with pre-filled values. */
  supplier?: Supplier;
  onSubmit: (data: SupplierCreateRequest) => void;
  onCancel: () => void;
  loading?: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const SupplierForm: React.FC<SupplierFormProps> = ({
  supplier,
  onSubmit,
  onCancel,
  loading = false,
}) => {
  const isEdit = Boolean(supplier);

  const [values, setValues] = useState<FormValues>({
    name: supplier?.name ?? '',
    country: supplier?.country ?? '',
    region: supplier?.region ?? '',
    tax_id: supplier?.tax_id ?? '',
    commodities: supplier?.commodities ?? [],
    contact_name: supplier?.contact_name ?? '',
    contact_email: supplier?.contact_email ?? '',
    contact_phone: supplier?.contact_phone ?? '',
    address: supplier?.address ?? '',
    certifications: supplier?.certifications?.join(', ') ?? '',
    notes: supplier?.notes ?? '',
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Set<string>>(new Set());

  // Update form when supplier prop changes (edit mode)
  useEffect(() => {
    if (supplier) {
      setValues({
        name: supplier.name,
        country: supplier.country,
        region: supplier.region,
        tax_id: supplier.tax_id,
        commodities: supplier.commodities,
        contact_name: supplier.contact_name,
        contact_email: supplier.contact_email,
        contact_phone: supplier.contact_phone ?? '',
        address: supplier.address ?? '',
        certifications: supplier.certifications?.join(', ') ?? '',
        notes: supplier.notes ?? '',
      });
    }
  }, [supplier]);

  const handleChange = (field: keyof FormValues) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setValues((prev) => ({ ...prev, [field]: e.target.value }));
    setTouched((prev) => new Set(prev).add(field));
  };

  const handleCommodityChange = (
    _event: React.SyntheticEvent,
    newValue: { value: EUDRCommodity; label: string }[]
  ) => {
    setValues((prev) => ({
      ...prev,
      commodities: newValue.map((v) => v.value),
    }));
    setTouched((prev) => new Set(prev).add('commodities'));
  };

  const handleCountryChange = (
    _event: React.SyntheticEvent,
    newValue: string | null
  ) => {
    setValues((prev) => ({ ...prev, country: newValue ?? '' }));
    setTouched((prev) => new Set(prev).add('country'));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const validationErrors = validate(values);
    setErrors(validationErrors);

    // Mark all fields as touched
    setTouched(new Set(Object.keys(values)));

    if (Object.keys(validationErrors).length > 0) return;

    const certArray = values.certifications
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);

    onSubmit({
      name: values.name.trim(),
      country: values.country.trim(),
      region: values.region.trim(),
      tax_id: values.tax_id.trim(),
      commodities: values.commodities,
      contact_name: values.contact_name.trim(),
      contact_email: values.contact_email.trim(),
      contact_phone: values.contact_phone.trim() || undefined,
      address: values.address.trim() || undefined,
      certifications: certArray.length > 0 ? certArray : undefined,
      notes: values.notes.trim() || undefined,
    });
  };

  const selectedCommodities = useMemo(
    () => EUDR_COMMODITIES.filter((c) => values.commodities.includes(c.value)),
    [values.commodities]
  );

  const showError = (field: keyof FormErrors) =>
    touched.has(field) && errors[field];

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        {isEdit ? 'Edit Supplier' : 'Add New Supplier'}
      </Typography>
      <Divider sx={{ mb: 2 }} />

      <Box component="form" onSubmit={handleSubmit} noValidate>
        <Grid container spacing={2}>
          {/* Company Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Company Information
            </Typography>
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              required
              label="Supplier Name"
              value={values.name}
              onChange={handleChange('name')}
              error={Boolean(showError('name'))}
              helperText={showError('name') || ''}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <Autocomplete
              options={COUNTRIES}
              value={values.country || null}
              onChange={handleCountryChange}
              renderInput={(params) => (
                <TextField
                  {...params}
                  required
                  label="Country"
                  error={Boolean(showError('country'))}
                  helperText={showError('country') || ''}
                />
              )}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Region / State"
              value={values.region}
              onChange={handleChange('region')}
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Tax ID"
              value={values.tax_id}
              onChange={handleChange('tax_id')}
              error={Boolean(showError('tax_id'))}
              helperText={showError('tax_id') || 'Alphanumeric, 3-30 characters'}
            />
          </Grid>

          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Address"
              value={values.address}
              onChange={handleChange('address')}
              multiline
              rows={2}
            />
          </Grid>

          {/* Commodities */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom sx={{ mt: 1 }}>
              EUDR Commodities
            </Typography>
          </Grid>

          <Grid item xs={12}>
            <Autocomplete
              multiple
              options={EUDR_COMMODITIES}
              getOptionLabel={(opt) => opt.label}
              value={selectedCommodities}
              onChange={handleCommodityChange}
              isOptionEqualToValue={(opt, val) => opt.value === val.value}
              renderTags={(value, getTagProps) =>
                value.map((option, index) => (
                  <Chip
                    label={option.label}
                    size="small"
                    color="primary"
                    variant="outlined"
                    {...getTagProps({ index })}
                    key={option.value}
                  />
                ))
              }
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Commodities"
                  required
                  error={Boolean(showError('commodities'))}
                  helperText={showError('commodities') || 'Select one or more EUDR commodities'}
                />
              )}
            />
          </Grid>

          {/* Contact Information */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom sx={{ mt: 1 }}>
              Contact Information
            </Typography>
          </Grid>

          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              required
              label="Contact Name"
              value={values.contact_name}
              onChange={handleChange('contact_name')}
              error={Boolean(showError('contact_name'))}
              helperText={showError('contact_name') || ''}
            />
          </Grid>

          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              required
              label="Contact Email"
              type="email"
              value={values.contact_email}
              onChange={handleChange('contact_email')}
              error={Boolean(showError('contact_email'))}
              helperText={showError('contact_email') || ''}
            />
          </Grid>

          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              label="Contact Phone"
              value={values.contact_phone}
              onChange={handleChange('contact_phone')}
            />
          </Grid>

          {/* Additional */}
          <Grid item xs={12}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom sx={{ mt: 1 }}>
              Additional Information
            </Typography>
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Certifications"
              value={values.certifications}
              onChange={handleChange('certifications')}
              helperText="Comma-separated list (e.g., Rainforest Alliance, UTZ)"
            />
          </Grid>

          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Notes"
              value={values.notes}
              onChange={handleChange('notes')}
              multiline
              rows={2}
            />
          </Grid>

          {/* Actions */}
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
            <Stack direction="row" justifyContent="flex-end" spacing={1} mt={1}>
              <Button
                variant="outlined"
                startIcon={<CancelIcon />}
                onClick={onCancel}
                disabled={loading}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="contained"
                startIcon={<SaveIcon />}
                disabled={loading}
              >
                {loading ? 'Saving...' : isEdit ? 'Update Supplier' : 'Create Supplier'}
              </Button>
            </Stack>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default SupplierForm;
