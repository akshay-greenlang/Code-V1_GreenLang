import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip, LinearProgress } from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';
import type { ReputationRisk } from '../../types';

interface ReputationDashboardProps { data: ReputationRisk[]; }

const ReputationDashboard: React.FC<ReputationDashboardProps> = ({ data }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>ESG Reputation & Sentiment</Typography>
    <Grid container spacing={2}>
      {data.map((r) => (
        <Grid item xs={12} sm={6} md={4} key={r.id}>
          <Box sx={{ p: 2, border: '1px solid #E0E0E0', borderRadius: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{r.factor}</Typography>
              {r.trend === 'improving' ? <TrendingUp color="success" fontSize="small" /> : r.trend === 'declining' ? <TrendingDown color="error" fontSize="small" /> : <TrendingFlat color="action" fontSize="small" />}
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <LinearProgress variant="determinate" value={r.current_score} sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                color={r.current_score >= 70 ? 'success' : r.current_score >= 40 ? 'warning' : 'error'} />
              <Typography variant="body2" sx={{ fontWeight: 700 }}>{r.current_score}</Typography>
            </Box>
            <Chip label={r.stakeholder_group} size="small" variant="outlined" sx={{ fontSize: 11 }} />
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>{r.impact_description}</Typography>
          </Box>
        </Grid>
      ))}
    </Grid>
    {data.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No reputation data available</Typography>}
  </CardContent></Card>
);

export default ReputationDashboard;
