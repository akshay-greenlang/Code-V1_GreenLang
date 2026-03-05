/**
 * EmissionSourceForm - Add/edit emission source form
 *
 * Provides a form for creating or editing an emission source with
 * category, gas, quantification method, activity data, emission factor,
 * and data quality tier fields.
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Grid,
  MenuItem,
  Typography,
} from '@mui/material';
import { Add, Save } from '@mui/icons-material';
import {
  ISOCategory,
  ISO_CATEGORY_SHORT_NAMES,
  GHGGas,
  QuantificationMethod,
  DataQualityTier,
} from '../../types';
import type { AddEmissionSourceRequest } from '../../types';

interface EmissionSourceFormProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: AddEmissionSourceRequest) => void;
  initialCategory?: ISOCategory;
  loading?: boolean;
}

const EmissionSourceForm: React.FC<EmissionSourceFormProps> = ({
  open,
  onClose,
  onSubmit,
  initialCategory,
  loading = false,
}) => {
  const [form, setForm] = useState<AddEmissionSourceRequest>({
    category: initialCategory || ISOCategory.CATEGORY_1_DIRECT,
    source_name: '',
    gas: GHGGas.CO2,
    method: QuantificationMethod.CALCULATION_BASED,
    activity_data: 0,
    activity_unit: '',
    emission_factor: 0,
    ef_unit: 'kg CO2e / unit',
    ef_source: '',
    data_quality_tier: DataQualityTier.TIER_2,
  });

  const handleChange = (field: keyof AddEmissionSourceRequest, value: unknown) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = () => {
    onSubmit(form);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Add Emission Source</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 0.5 }}>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              required
              label="Source Name"
              value={form.source_name}
              onChange={(e) => handleChange('source_name', e.target.value)}
              placeholder="e.g. Natural gas boiler - Building A"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              required
              select
              label="ISO Category"
              value={form.category}
              onChange={(e) => handleChange('category', e.target.value)}
            >
              {Object.values(ISOCategory).map((cat) => (
                <MenuItem key={cat} value={cat}>
                  {ISO_CATEGORY_SHORT_NAMES[cat]}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              select
              label="GHG Gas"
              value={form.gas}
              onChange={(e) => handleChange('gas', e.target.value)}
            >
              {Object.values(GHGGas).map((gas) => (
                <MenuItem key={gas} value={gas}>
                  {gas}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              select
              label="Quantification Method"
              value={form.method}
              onChange={(e) => handleChange('method', e.target.value)}
            >
              <MenuItem value={QuantificationMethod.CALCULATION_BASED}>
                Calculation-Based
              </MenuItem>
              <MenuItem value={QuantificationMethod.DIRECT_MEASUREMENT}>
                Direct Measurement
              </MenuItem>
              <MenuItem value={QuantificationMethod.MASS_BALANCE}>
                Mass Balance
              </MenuItem>
            </TextField>
          </Grid>
          <Grid item xs={12} sm={4}>
            <TextField
              fullWidth
              select
              label="Data Quality Tier"
              value={form.data_quality_tier}
              onChange={(e) => handleChange('data_quality_tier', e.target.value)}
            >
              {Object.values(DataQualityTier).map((tier) => (
                <MenuItem key={tier} value={tier}>
                  {tier.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </MenuItem>
              ))}
            </TextField>
          </Grid>

          <Grid item xs={12}>
            <Typography variant="subtitle2" sx={{ mt: 1, mb: 0.5 }}>
              Activity Data and Emission Factor
            </Typography>
          </Grid>

          <Grid item xs={12} sm={3}>
            <TextField
              fullWidth
              required
              type="number"
              label="Activity Data"
              value={form.activity_data}
              onChange={(e) => handleChange('activity_data', Number(e.target.value))}
              inputProps={{ step: 'any' }}
            />
          </Grid>
          <Grid item xs={12} sm={3}>
            <TextField
              fullWidth
              label="Activity Unit"
              value={form.activity_unit}
              onChange={(e) => handleChange('activity_unit', e.target.value)}
              placeholder="e.g. m3, kWh, litres"
            />
          </Grid>
          <Grid item xs={12} sm={3}>
            <TextField
              fullWidth
              required
              type="number"
              label="Emission Factor"
              value={form.emission_factor}
              onChange={(e) => handleChange('emission_factor', Number(e.target.value))}
              inputProps={{ step: 'any' }}
            />
          </Grid>
          <Grid item xs={12} sm={3}>
            <TextField
              fullWidth
              label="EF Unit"
              value={form.ef_unit}
              onChange={(e) => handleChange('ef_unit', e.target.value)}
              placeholder="e.g. kg CO2e / kWh"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="EF Source"
              value={form.ef_source}
              onChange={(e) => handleChange('ef_source', e.target.value)}
              placeholder="e.g. IPCC 2006, EPA 2024, DEFRA 2025"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              label="Facility ID (optional)"
              value={form.facility_id || ''}
              onChange={(e) =>
                handleChange('facility_id', e.target.value || null)
              }
              placeholder="Optional facility identifier"
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={
            loading || !form.source_name || form.activity_data === 0
          }
          startIcon={<Add />}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          Add Source
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default EmissionSourceForm;
