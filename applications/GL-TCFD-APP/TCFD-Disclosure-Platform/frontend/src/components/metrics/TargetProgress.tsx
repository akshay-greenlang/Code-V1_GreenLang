import React from 'react';
import { Card, CardContent, Typography, Grid, Box, LinearProgress, Chip } from '@mui/material';
import type { ClimateTarget } from '../../types';

interface TargetProgressProps { targets: ClimateTarget[]; }

const TargetProgress: React.FC<TargetProgressProps> = ({ targets }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Target Progress</Typography>
    <Grid container spacing={2}>
      {targets.map((t) => (
        <Grid item xs={12} sm={6} key={t.id}>
          <Box sx={{ p: 2, border: '1px solid #E0E0E0', borderRadius: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{t.name}</Typography>
              <Chip label={t.on_track ? 'On Track' : 'Off Track'} size="small" color={t.on_track ? 'success' : 'error'} />
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption">Base: {t.base_value.toLocaleString()} ({t.base_year})</Typography>
              <Typography variant="caption">Target: {t.target_value.toLocaleString()} ({t.target_year})</Typography>
            </Box>
            <LinearProgress variant="determinate" value={Math.min(100, t.progress_pct)} sx={{ height: 10, borderRadius: 5, mb: 0.5 }}
              color={t.progress_pct >= 80 ? 'success' : t.progress_pct >= 50 ? 'primary' : 'warning'} />
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>{t.progress_pct.toFixed(1)}% complete</Typography>
              <Typography variant="caption">Current: {t.current_value.toLocaleString()}</Typography>
            </Box>
            {t.sbti_aligned && <Chip label={`SBTi: ${t.sbti_status}`} size="small" sx={{ mt: 1, fontSize: 10 }} color="primary" variant="outlined" />}
          </Box>
        </Grid>
      ))}
    </Grid>
  </CardContent></Card>
);

export default TargetProgress;
