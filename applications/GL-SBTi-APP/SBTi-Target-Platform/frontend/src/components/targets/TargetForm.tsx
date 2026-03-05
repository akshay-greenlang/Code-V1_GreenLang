/**
 * TargetForm - Form for creating/editing SBTi targets with all fields.
 */

import React, { useState } from 'react';
import {
  Card, CardContent, Typography, Grid, TextField, Select, MenuItem, FormControl,
  InputLabel, Button, Box, Alert, SelectChangeEvent,
} from '@mui/material';
import { Save, Send } from '@mui/icons-material';
import type { Target, TargetType, TargetMethod, TargetScope, TargetTimeframe } from '../../types';
import { validateTargetForm } from '../../utils/validators';

interface TargetFormProps {
  target?: Partial<Target>;
  onSave: (data: Partial<Target>) => void;
  onSubmit?: (data: Partial<Target>) => void;
  saving?: boolean;
}

const TargetForm: React.FC<TargetFormProps> = ({ target, onSave, onSubmit, saving }) => {
  const [form, setForm] = useState<Partial<Target>>({
    name: '', description: '', target_type: 'absolute', target_method: 'cross_sector_aca',
    target_scope: 'scope_1_2', target_timeframe: 'near_term', base_year: 2020,
    target_year: 2030, target_reduction_pct: 42, scope_coverage_pct: 95,
    boundary_description: '', ...target,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
    setErrors((prev) => { const next = { ...prev }; delete next[field]; return next; });
  };

  const handleSelect = (field: string) => (e: SelectChangeEvent<string>) => {
    setForm((prev) => ({ ...prev, [field]: e.target.value }));
  };

  const handleSave = () => {
    const validation = validateTargetForm({
      name: form.name || '',
      base_year: Number(form.base_year),
      target_year: Number(form.target_year),
      target_reduction_pct: Number(form.target_reduction_pct),
      scope_coverage_pct: Number(form.scope_coverage_pct),
      target_scope: form.target_scope || 'scope_1_2',
      target_timeframe: form.target_timeframe || 'near_term',
    });
    if (Object.keys(validation).length > 0) { setErrors(validation); return; }
    onSave(form);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
          {target?.id ? 'Edit Target' : 'Create New Target'}
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <TextField fullWidth label="Target Name" value={form.name || ''} onChange={handleChange('name')}
              error={!!errors.name} helperText={errors.name} required />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField fullWidth label="Description" value={form.description || ''} onChange={handleChange('description')} multiline rows={1} />
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Target Type</InputLabel>
              <Select value={form.target_type || 'absolute'} label="Target Type" onChange={handleSelect('target_type')}>
                <MenuItem value="absolute">Absolute</MenuItem>
                <MenuItem value="intensity">Intensity</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Method</InputLabel>
              <Select value={form.target_method || 'cross_sector_aca'} label="Method" onChange={handleSelect('target_method')}>
                <MenuItem value="cross_sector_aca">Cross-Sector (ACA)</MenuItem>
                <MenuItem value="sector_specific_sda">Sector-Specific (SDA)</MenuItem>
                <MenuItem value="portfolio_coverage">Portfolio Coverage</MenuItem>
                <MenuItem value="temperature_rating">Temperature Rating</MenuItem>
                <MenuItem value="engagement_threshold">Engagement Threshold</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Scope</InputLabel>
              <Select value={form.target_scope || 'scope_1_2'} label="Scope" onChange={handleSelect('target_scope')}>
                <MenuItem value="scope_1">Scope 1</MenuItem>
                <MenuItem value="scope_2">Scope 2</MenuItem>
                <MenuItem value="scope_1_2">Scope 1+2</MenuItem>
                <MenuItem value="scope_3">Scope 3</MenuItem>
                <MenuItem value="scope_1_2_3">Scope 1+2+3</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <FormControl fullWidth>
              <InputLabel>Timeframe</InputLabel>
              <Select value={form.target_timeframe || 'near_term'} label="Timeframe" onChange={handleSelect('target_timeframe')}>
                <MenuItem value="near_term">Near-Term (5-10 years)</MenuItem>
                <MenuItem value="long_term">Long-Term (to 2050)</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField fullWidth type="number" label="Base Year" value={form.base_year || ''} onChange={handleChange('base_year')}
              error={!!errors.base_year} helperText={errors.base_year} />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField fullWidth type="number" label="Target Year" value={form.target_year || ''} onChange={handleChange('target_year')}
              error={!!errors.target_year} helperText={errors.target_year} />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField fullWidth type="number" label="Reduction %" value={form.target_reduction_pct || ''} onChange={handleChange('target_reduction_pct')}
              error={!!errors.target_reduction_pct} helperText={errors.target_reduction_pct} />
          </Grid>
          <Grid item xs={12} md={3}>
            <TextField fullWidth type="number" label="Coverage %" value={form.scope_coverage_pct || ''} onChange={handleChange('scope_coverage_pct')}
              error={!!errors.scope_coverage_pct} helperText={errors.scope_coverage_pct} />
          </Grid>
          <Grid item xs={12}>
            <TextField fullWidth label="Boundary Description" value={form.boundary_description || ''} onChange={handleChange('boundary_description')} multiline rows={2} />
          </Grid>
        </Grid>
        {Object.keys(errors).length > 0 && (
          <Alert severity="error" sx={{ mt: 2 }}>Please fix the validation errors above before saving.</Alert>
        )}
        <Box sx={{ display: 'flex', gap: 2, mt: 3, justifyContent: 'flex-end' }}>
          <Button variant="outlined" startIcon={<Save />} onClick={handleSave} disabled={saving}>
            Save Draft
          </Button>
          {onSubmit && (
            <Button variant="contained" startIcon={<Send />} onClick={() => onSubmit(form)} disabled={saving}>
              Submit for Validation
            </Button>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

export default TargetForm;
