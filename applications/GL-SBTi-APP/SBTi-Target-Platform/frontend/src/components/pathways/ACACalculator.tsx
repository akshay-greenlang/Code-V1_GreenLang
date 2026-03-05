/**
 * ACACalculator - Interactive Absolute Contraction Approach parameter inputs.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Grid, TextField, Button, Select, MenuItem, FormControl, InputLabel, Box, Alert, SelectChangeEvent } from '@mui/material';
import { Calculate } from '@mui/icons-material';
import { getMinimumAmbitionRate } from '../../utils/pathwayHelpers';
import type { PathwayAlignment } from '../../types';

interface ACACalculatorProps {
  onCalculate: (params: { base_year: number; target_year: number; base_emissions: number; alignment: string }) => void;
  calculating?: boolean;
}

const ACACalculator: React.FC<ACACalculatorProps> = ({ onCalculate, calculating }) => {
  const [baseYear, setBaseYear] = useState(2020);
  const [targetYear, setTargetYear] = useState(2030);
  const [baseEmissions, setBaseEmissions] = useState(100000);
  const [alignment, setAlignment] = useState<PathwayAlignment>('1.5C');

  const years = targetYear - baseYear;
  const minRate = getMinimumAmbitionRate(alignment);
  const totalReduction = minRate * years;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>ACA Pathway Calculator</Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} md={3}>
            <TextField fullWidth type="number" label="Base Year" value={baseYear} onChange={(e) => setBaseYear(Number(e.target.value))} size="small" />
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField fullWidth type="number" label="Target Year" value={targetYear} onChange={(e) => setTargetYear(Number(e.target.value))} size="small" />
          </Grid>
          <Grid item xs={6} md={3}>
            <TextField fullWidth type="number" label="Base Emissions (tCO2e)" value={baseEmissions} onChange={(e) => setBaseEmissions(Number(e.target.value))} size="small" />
          </Grid>
          <Grid item xs={6} md={3}>
            <FormControl fullWidth size="small">
              <InputLabel>Alignment</InputLabel>
              <Select value={alignment} label="Alignment" onChange={(e: SelectChangeEvent) => setAlignment(e.target.value as PathwayAlignment)}>
                <MenuItem value="1.5C">1.5{'\u00B0'}C</MenuItem>
                <MenuItem value="well_below_2C">Well Below 2{'\u00B0'}C</MenuItem>
                <MenuItem value="2C">2{'\u00B0'}C</MenuItem>
              </Select>
            </FormControl>
          </Grid>
        </Grid>
        <Box sx={{ mt: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
          <Alert severity="info" sx={{ flex: 1 }}>
            Min annual rate: {minRate}%/yr | Target period: {years} years | Total reduction: {totalReduction.toFixed(1)}%
          </Alert>
          <Button variant="contained" startIcon={<Calculate />} onClick={() => onCalculate({ base_year: baseYear, target_year: targetYear, base_emissions: baseEmissions, alignment })} disabled={calculating}>
            Calculate
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ACACalculator;
