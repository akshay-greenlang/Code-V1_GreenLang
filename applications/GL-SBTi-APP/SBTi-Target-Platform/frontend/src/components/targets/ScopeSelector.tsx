/**
 * ScopeSelector - Multi-scope selector with coverage calculator.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, LinearProgress } from '@mui/material';
import type { TargetScopeDetail } from '../../types';

interface ScopeSelectorProps {
  scopes: TargetScopeDetail[];
  selectedScope: string;
  onScopeChange: (scope: string) => void;
}

const SCOPE_LABELS: Record<string, string> = {
  scope_1: 'Scope 1', scope_2: 'Scope 2', scope_1_2: 'Scope 1+2', scope_3: 'Scope 3', scope_1_2_3: 'Scope 1+2+3',
};

const ScopeSelector: React.FC<ScopeSelectorProps> = ({ scopes, selectedScope, onScopeChange }) => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Scope Coverage</Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 3, flexWrap: 'wrap' }}>
          {['scope_1_2', 'scope_3', 'scope_1_2_3'].map((scope) => (
            <Chip
              key={scope} label={SCOPE_LABELS[scope]} clickable
              color={selectedScope === scope ? 'primary' : 'default'}
              variant={selectedScope === scope ? 'filled' : 'outlined'}
              onClick={() => onScopeChange(scope)}
            />
          ))}
        </Box>
        {scopes.map((s) => (
          <Box key={s.scope} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="body2" fontWeight={500}>{SCOPE_LABELS[s.scope] || s.scope}</Typography>
              <Typography variant="body2" fontWeight={600}>{s.coverage_pct.toFixed(0)}% coverage</Typography>
            </Box>
            <LinearProgress
              variant="determinate" value={s.coverage_pct}
              color={s.coverage_pct >= 95 ? 'success' : s.coverage_pct >= 67 ? 'warning' : 'error'}
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                Base: {s.base_year_emissions.toLocaleString()} tCO2e
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Reduction: {s.reduction_pct.toFixed(1)}%
              </Typography>
            </Box>
          </Box>
        ))}
      </CardContent>
    </Card>
  );
};

export default ScopeSelector;
