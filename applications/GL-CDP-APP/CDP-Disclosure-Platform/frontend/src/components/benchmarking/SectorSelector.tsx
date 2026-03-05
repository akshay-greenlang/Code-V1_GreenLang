/**
 * SectorSelector - GICS sector dropdown for benchmarking
 */
import React from 'react';
import { TextField, MenuItem } from '@mui/material';

const GICS_SECTORS = [
  'Energy', 'Materials', 'Industrials', 'Consumer Discretionary', 'Consumer Staples',
  'Health Care', 'Financials', 'Information Technology', 'Communication Services',
  'Utilities', 'Real Estate',
];

interface SectorSelectorProps { value: string; onChange: (sector: string) => void; }

const SectorSelector: React.FC<SectorSelectorProps> = ({ value, onChange }) => (
  <TextField select label="GICS Sector" value={value} onChange={(e) => onChange(e.target.value)} size="small" sx={{ minWidth: 250 }}>
    {GICS_SECTORS.map((s) => (<MenuItem key={s} value={s}>{s}</MenuItem>))}
  </TextField>
);

export default SectorSelector;
