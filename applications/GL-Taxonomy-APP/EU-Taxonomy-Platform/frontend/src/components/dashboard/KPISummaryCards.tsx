/**
 * KPISummaryCards - Three KPI cards for turnover, capex, opex percentages.
 */

import React from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import ScoreGauge from '../common/ScoreGauge';

interface KPICardData {
  label: string;
  eligible: number;
  aligned: number;
  change?: number;
}

const DEMO_KPIS: KPICardData[] = [
  { label: 'Turnover', eligible: 68.2, aligned: 42.5, change: 3.2 },
  { label: 'CapEx', eligible: 72.1, aligned: 51.3, change: 5.8 },
  { label: 'OpEx', eligible: 55.4, aligned: 38.7, change: -1.4 },
];

interface KPISummaryCardsProps {
  kpis?: KPICardData[];
}

const KPISummaryCards: React.FC<KPISummaryCardsProps> = ({ kpis = DEMO_KPIS }) => (
  <Grid container spacing={2}>
    {kpis.map(kpi => (
      <Grid item xs={12} md={4} key={kpi.label}>
        <Card>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2 }}>
              {kpi.label} KPI
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: 2 }}>
              <ScoreGauge value={kpi.eligible} label="Eligible" size={80} thickness={5} color="#0277BD" />
              <ScoreGauge value={kpi.aligned} label="Aligned" size={80} thickness={5} color="#1B5E20" />
            </Box>
            {kpi.change !== undefined && (
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5 }}>
                {kpi.change >= 0 ? (
                  <TrendingUp fontSize="small" color="success" />
                ) : (
                  <TrendingDown fontSize="small" color="error" />
                )}
                <Typography variant="body2" color={kpi.change >= 0 ? 'success.main' : 'error.main'}>
                  {kpi.change >= 0 ? '+' : ''}{kpi.change.toFixed(1)}% vs prior period
                </Typography>
              </Box>
            )}
          </CardContent>
        </Card>
      </Grid>
    ))}
  </Grid>
);

export default KPISummaryCards;
