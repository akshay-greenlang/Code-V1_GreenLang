/**
 * SupplierTracker - Supplier progress tracker with SBTi and transition plan flags
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip, LinearProgress } from '@mui/material';
import { CheckCircle, Cancel } from '@mui/icons-material';
import type { SupplierRequest } from '../../types';

interface SupplierTrackerProps { supplier: SupplierRequest; }

const SupplierTracker: React.FC<SupplierTrackerProps> = ({ supplier }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" gutterBottom>{supplier.supplier_name}</Typography>
      <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
        <Chip label={supplier.supplier_country} size="small" />
        <Chip label={supplier.supplier_sector} size="small" variant="outlined" />
        {supplier.has_sbti_target ? <Chip icon={<CheckCircle sx={{ fontSize: 14 }} />} label="SBTi Target" color="success" size="small" /> : <Chip icon={<Cancel sx={{ fontSize: 14 }} />} label="No SBTi" size="small" variant="outlined" />}
        {supplier.has_transition_plan ? <Chip icon={<CheckCircle sx={{ fontSize: 14 }} />} label="Transition Plan" color="success" size="small" /> : <Chip icon={<Cancel sx={{ fontSize: 14 }} />} label="No Plan" size="small" variant="outlined" />}
      </Box>
      {supplier.scope_1_emissions != null && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="caption" color="text.secondary">Scope 1: {supplier.scope_1_emissions.toFixed(0)} tCO2e</Typography>
        </Box>
      )}
      {supplier.scope_2_emissions != null && (
        <Box sx={{ mb: 1 }}>
          <Typography variant="caption" color="text.secondary">Scope 2: {supplier.scope_2_emissions.toFixed(0)} tCO2e</Typography>
        </Box>
      )}
      {supplier.engagement_score != null && (
        <Box>
          <Typography variant="caption" color="text.secondary">Engagement Score</Typography>
          <LinearProgress variant="determinate" value={supplier.engagement_score} sx={{ height: 6, borderRadius: 3 }} />
        </Box>
      )}
    </CardContent>
  </Card>
);

export default SupplierTracker;
