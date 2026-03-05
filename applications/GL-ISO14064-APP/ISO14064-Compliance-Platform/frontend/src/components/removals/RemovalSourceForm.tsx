/**
 * RemovalSourceForm - Add/edit removal source dialog
 *
 * Provides a form for creating or editing a GHG removal source
 * with type, permanence level, gross removals, monitoring plan,
 * and data quality tier fields.
 */

import React, { useState, useEffect } from 'react';
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
  RemovalType,
  PermanenceLevel,
  DataQualityTier,
} from '../../types';
import type { AddRemovalSourceRequest, RemovalSource } from '../../types';

interface RemovalSourceFormProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: AddRemovalSourceRequest) => void;
  editSource?: RemovalSource | null;
  loading?: boolean;
}

const REMOVAL_TYPE_LABELS: Record<RemovalType, string> = {
  [RemovalType.FORESTRY]: 'Forestry / Afforestation',
  [RemovalType.SOIL_CARBON]: 'Soil Carbon Sequestration',
  [RemovalType.CCS]: 'Carbon Capture & Storage',
  [RemovalType.DIRECT_AIR_CAPTURE]: 'Direct Air Capture',
  [RemovalType.BECCS]: 'BECCS',
  [RemovalType.WETLAND_RESTORATION]: 'Wetland Restoration',
  [RemovalType.OCEAN_BASED]: 'Ocean-Based Removal',
  [RemovalType.OTHER]: 'Other',
};

const PERMANENCE_LABELS: Record<PermanenceLevel, string> = {
  [PermanenceLevel.PERMANENT]: 'Permanent (>1000 years)',
  [PermanenceLevel.LONG_TERM]: 'Long Term (100-1000 years)',
  [PermanenceLevel.MEDIUM_TERM]: 'Medium Term (25-100 years)',
  [PermanenceLevel.SHORT_TERM]: 'Short Term (10-25 years)',
  [PermanenceLevel.REVERSIBLE]: 'Reversible (<10 years)',
};

const RemovalSourceForm: React.FC<RemovalSourceFormProps> = ({
  open,
  onClose,
  onSubmit,
  editSource,
  loading = false,
}) => {
  const [form, setForm] = useState<AddRemovalSourceRequest>({
    removal_type: RemovalType.FORESTRY,
    source_name: '',
    gross_removals_tco2e: 0,
    permanence_level: PermanenceLevel.LONG_TERM,
    monitoring_plan: null,
    data_quality_tier: DataQualityTier.TIER_2,
  });

  useEffect(() => {
    if (editSource) {
      setForm({
        removal_type: editSource.removal_type,
        source_name: editSource.source_name,
        gross_removals_tco2e: editSource.gross_removals_tco2e,
        permanence_level: editSource.permanence_level,
        monitoring_plan: editSource.monitoring_plan,
        data_quality_tier: editSource.data_quality_tier,
      });
    } else {
      setForm({
        removal_type: RemovalType.FORESTRY,
        source_name: '',
        gross_removals_tco2e: 0,
        permanence_level: PermanenceLevel.LONG_TERM,
        monitoring_plan: null,
        data_quality_tier: DataQualityTier.TIER_2,
      });
    }
  }, [editSource, open]);

  const handleChange = (field: keyof AddRemovalSourceRequest, value: unknown) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = () => {
    onSubmit(form);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>{editSource ? 'Edit Removal Source' : 'Add Removal Source'}</DialogTitle>
      <DialogContent>
        <Grid container spacing={2} sx={{ mt: 0.5 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              required
              label="Source Name"
              value={form.source_name}
              onChange={(e) => handleChange('source_name', e.target.value)}
              placeholder="e.g. Reforestation Project - North Region"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              select
              label="Removal Type"
              value={form.removal_type}
              onChange={(e) => handleChange('removal_type', e.target.value)}
            >
              {Object.entries(REMOVAL_TYPE_LABELS).map(([val, label]) => (
                <MenuItem key={val} value={val}>
                  {label}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              select
              label="Permanence Level"
              value={form.permanence_level}
              onChange={(e) => handleChange('permanence_level', e.target.value)}
            >
              {Object.entries(PERMANENCE_LABELS).map(([val, label]) => (
                <MenuItem key={val} value={val}>
                  {label}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              fullWidth
              required
              type="number"
              label="Gross Removals (tCO2e)"
              value={form.gross_removals_tco2e}
              onChange={(e) =>
                handleChange('gross_removals_tco2e', Number(e.target.value))
              }
              inputProps={{ step: 'any', min: 0 }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
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
            <TextField
              fullWidth
              multiline
              rows={3}
              label="Monitoring Plan"
              value={form.monitoring_plan || ''}
              onChange={(e) =>
                handleChange('monitoring_plan', e.target.value || null)
              }
              placeholder="Describe the monitoring and verification plan for this removal activity..."
            />
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={loading || !form.source_name || form.gross_removals_tco2e <= 0}
          startIcon={editSource ? <Save /> : <Add />}
          sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
        >
          {editSource ? 'Save Changes' : 'Add Removal'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RemovalSourceForm;
