/**
 * CoverageCalculator - Interactive category selector with coverage percentage.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Checkbox, FormControlLabel, LinearProgress, Alert } from '@mui/material';
import type { CategoryBreakdown } from '../../types';

interface CoverageCalculatorProps { categories: CategoryBreakdown[]; onToggleCategory: (categoryNumber: number, included: boolean) => void; }

const CoverageCalculator: React.FC<CoverageCalculatorProps> = ({ categories, onToggleCategory }) => {
  const totalScope3 = categories.reduce((sum, c) => sum + c.emissions_tco2e, 0);
  const includedEmissions = categories.filter((c) => c.included_in_target).reduce((sum, c) => sum + c.emissions_tco2e, 0);
  const coveragePct = totalScope3 > 0 ? (includedEmissions / totalScope3) * 100 : 0;
  const meetsThreshold = coveragePct >= 67;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Coverage Calculator</Typography>
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
            <Typography variant="body2" fontWeight={500}>Scope 3 Coverage</Typography>
            <Typography variant="body2" fontWeight={700} color={meetsThreshold ? 'success.main' : 'error.main'}>{coveragePct.toFixed(1)}%</Typography>
          </Box>
          <LinearProgress variant="determinate" value={coveragePct} color={meetsThreshold ? 'success' : 'error'} sx={{ height: 10, borderRadius: 5 }} />
        </Box>
        <Alert severity={meetsThreshold ? 'success' : 'warning'} sx={{ mb: 2 }}>
          {meetsThreshold ? 'Coverage meets the minimum 2/3 (67%) requirement.' : `Coverage is ${coveragePct.toFixed(1)}%. At least 67% is required for SBTi.`}
        </Alert>
        {categories.sort((a, b) => b.emissions_tco2e - a.emissions_tco2e).map((cat) => (
          <FormControlLabel key={cat.category_number} sx={{ display: 'flex', mb: 0.5 }}
            control={<Checkbox checked={cat.included_in_target} onChange={(e) => onToggleCategory(cat.category_number, e.target.checked)} size="small" />}
            label={<Typography variant="body2">Cat {cat.category_number}: {cat.category_name} ({cat.percentage_of_scope3.toFixed(1)}%)</Typography>}
          />
        ))}
      </CardContent>
    </Card>
  );
};

export default CoverageCalculator;
