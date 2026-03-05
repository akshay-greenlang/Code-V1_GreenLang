/**
 * MethodPicker - Target method selector (ACA/SDA/intensity/engagement).
 */

import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Radio, RadioGroup, FormControlLabel } from '@mui/material';
import { Timeline, DeviceHub, TrendingUp, People } from '@mui/icons-material';
import type { TargetMethod } from '../../types';

interface MethodPickerProps {
  selectedMethod: TargetMethod;
  onMethodChange: (method: TargetMethod) => void;
  sector?: string;
}

const METHODS: { value: TargetMethod; label: string; description: string; icon: React.ReactNode }[] = [
  { value: 'cross_sector_aca', label: 'Absolute Contraction (ACA)', description: 'Cross-sector linear reduction. Minimum 4.2%/yr for 1.5C, 2.5%/yr for WB2C.', icon: <TrendingUp /> },
  { value: 'sector_specific_sda', label: 'Sector Decarbonization (SDA)', description: 'Sector-specific intensity pathway based on IEA/IPCC scenarios.', icon: <DeviceHub /> },
  { value: 'portfolio_coverage', label: 'Portfolio Coverage', description: 'For FIs: percentage of portfolio with validated SBTi targets.', icon: <Timeline /> },
  { value: 'engagement_threshold', label: 'Engagement Threshold', description: 'For Scope 3: engage suppliers/investees to set SBTi targets.', icon: <People /> },
];

const MethodPicker: React.FC<MethodPickerProps> = ({ selectedMethod, onMethodChange }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Target Setting Method</Typography>
        <RadioGroup value={selectedMethod} onChange={(e) => onMethodChange(e.target.value as TargetMethod)}>
          <Grid container spacing={2}>
            {METHODS.map((method) => (
              <Grid item xs={12} md={6} key={method.value}>
                <Box
                  sx={{
                    border: selectedMethod === method.value ? '2px solid' : '1px solid',
                    borderColor: selectedMethod === method.value ? 'primary.main' : '#E0E0E0',
                    borderRadius: 1, p: 2, cursor: 'pointer',
                    backgroundColor: selectedMethod === method.value ? 'primary.main' + '08' : 'transparent',
                    '&:hover': { borderColor: 'primary.main' },
                  }}
                  onClick={() => onMethodChange(method.value)}
                >
                  <FormControlLabel
                    value={method.value}
                    control={<Radio size="small" />}
                    label={
                      <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          {method.icon}
                          <Typography variant="subtitle2" fontWeight={600}>{method.label}</Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary">{method.description}</Typography>
                      </Box>
                    }
                    sx={{ m: 0, width: '100%' }}
                  />
                </Box>
              </Grid>
            ))}
          </Grid>
        </RadioGroup>
      </CardContent>
    </Card>
  );
};

export default MethodPicker;
