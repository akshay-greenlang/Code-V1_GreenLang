/**
 * ReadinessBar - Overall readiness progress bar.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress, Chip } from '@mui/material';

interface ReadinessBarProps { score: number; blockers?: string[]; }

const ReadinessBar: React.FC<ReadinessBarProps> = ({ score, blockers = [] }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Validation Readiness</Typography>
        <Typography variant="h5" sx={{ fontWeight: 700, color: score >= 80 ? '#2E7D32' : score >= 50 ? '#EF6C00' : '#C62828' }}>{score}%</Typography>
      </Box>
      <LinearProgress variant="determinate" value={score}
        color={score >= 80 ? 'success' : score >= 50 ? 'warning' : 'error'}
        sx={{ height: 12, borderRadius: 6, mb: 2 }} />
      {blockers.length > 0 && (
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>Blockers</Typography>
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
            {blockers.map((b, i) => <Chip key={i} label={b} size="small" color="error" variant="outlined" />)}
          </Box>
        </Box>
      )}
    </CardContent>
  </Card>
);

export default ReadinessBar;
