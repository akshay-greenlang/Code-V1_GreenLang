import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip } from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';

interface MetricsSummaryProps {
  metrics: { name: string; value: number; unit: string; change_pct: number; trend: string }[];
}

const MetricsSummary: React.FC<MetricsSummaryProps> = ({ metrics }) => {
  const displayMetrics = metrics.length > 0 ? metrics : [
    { name: 'Scope 1 Emissions', value: 12450, unit: 'tCO2e', change_pct: -8.2, trend: 'improving' },
    { name: 'Scope 2 Emissions', value: 8320, unit: 'tCO2e', change_pct: -12.5, trend: 'improving' },
    { name: 'Carbon Intensity', value: 45.2, unit: 'tCO2e/$M', change_pct: -5.1, trend: 'improving' },
    { name: 'Renewable Energy %', value: 42, unit: '%', change_pct: 8.0, trend: 'improving' },
    { name: 'Internal Carbon Price', value: 85, unit: '$/tCO2e', change_pct: 12.5, trend: 'stable' },
    { name: 'Climate CapEx', value: 25.4, unit: '$M', change_pct: 18.0, trend: 'improving' },
    { name: 'Remuneration Linked', value: 15, unit: '%', change_pct: 5.0, trend: 'stable' },
  ];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Cross-Industry Climate Metrics
        </Typography>
        <Grid container spacing={1.5}>
          {displayMetrics.map((metric) => (
            <Grid item xs={12} sm={6} key={metric.name}>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: 1,
                  border: '1px solid #E0E0E0',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    {metric.name}
                  </Typography>
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {typeof metric.value === 'number' ? metric.value.toLocaleString() : metric.value} {metric.unit}
                  </Typography>
                </Box>
                <Chip
                  icon={
                    metric.change_pct > 0
                      ? <TrendingUp sx={{ fontSize: 16 }} />
                      : metric.change_pct < 0
                      ? <TrendingDown sx={{ fontSize: 16 }} />
                      : <TrendingFlat sx={{ fontSize: 16 }} />
                  }
                  label={`${metric.change_pct > 0 ? '+' : ''}${metric.change_pct.toFixed(1)}%`}
                  size="small"
                  color={metric.trend === 'improving' ? 'success' : metric.trend === 'deteriorating' ? 'error' : 'default'}
                  variant="outlined"
                />
              </Box>
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default MetricsSummary;
