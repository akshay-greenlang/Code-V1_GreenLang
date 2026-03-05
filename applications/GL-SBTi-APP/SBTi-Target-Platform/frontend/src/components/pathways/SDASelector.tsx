/**
 * SDASelector - Sector-specific SDA configuration.
 */

import React from 'react';
import { Card, CardContent, Typography, Grid, FormControl, InputLabel, Select, MenuItem, TextField, Box, Chip, SelectChangeEvent } from '@mui/material';
import type { SBTiSector } from '../../types';
import { formatSector } from '../../utils/formatters';

interface SDASelectorProps {
  sector: SBTiSector;
  onSectorChange: (sector: SBTiSector) => void;
  baseIntensity: number;
  onBaseIntensityChange: (value: number) => void;
  intensityUnit: string;
  onIntensityUnitChange: (unit: string) => void;
}

const SECTORS: SBTiSector[] = ['power_generation', 'steel', 'cement', 'aluminum', 'chemicals', 'pulp_and_paper', 'transport', 'buildings', 'aviation', 'shipping'];
const INTENSITY_UNITS = ['tCO2e/MWh', 'tCO2e/tonne', 'tCO2e/m2', 'tCO2e/pkm', 'tCO2e/tkm', 'tCO2e/revenue'];

const SDASelector: React.FC<SDASelectorProps> = ({ sector, onSectorChange, baseIntensity, onBaseIntensityChange, intensityUnit, onIntensityUnitChange }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>SDA Configuration</Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Sector</InputLabel>
              <Select value={sector} label="Sector" onChange={(e: SelectChangeEvent) => onSectorChange(e.target.value as SBTiSector)}>
                {SECTORS.map((s) => <MenuItem key={s} value={s}>{formatSector(s)}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={4}>
            <TextField fullWidth type="number" label="Base Year Intensity" value={baseIntensity} onChange={(e) => onBaseIntensityChange(Number(e.target.value))} size="small" />
          </Grid>
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Intensity Unit</InputLabel>
              <Select value={intensityUnit} label="Intensity Unit" onChange={(e: SelectChangeEvent) => onIntensityUnitChange(e.target.value)}>
                {INTENSITY_UNITS.map((u) => <MenuItem key={u} value={u}>{u}</MenuItem>)}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            SDA pathways use sector-specific IEA/IPCC scenario data to calculate the intensity reduction needed.
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SDASelector;
