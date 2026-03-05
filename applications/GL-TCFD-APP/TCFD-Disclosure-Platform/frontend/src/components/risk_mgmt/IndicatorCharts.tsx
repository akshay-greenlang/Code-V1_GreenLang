import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { RiskIndicator } from '../../types';

interface IndicatorChartsProps { data: RiskIndicator[]; }

const IndicatorCharts: React.FC<IndicatorChartsProps> = ({ data }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Leading & Lagging Indicators</Typography>
    <Grid container spacing={2}>
      {data.map((indicator) => {
        const chartData = indicator.history.map((h) => ({ date: h.date.slice(5), value: h.value }));
        return (
          <Grid item xs={12} sm={6} md={4} key={indicator.id}>
            <Box sx={{ p: 1.5, border: '1px solid #E0E0E0', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>{indicator.name}</Typography>
                <Chip label={indicator.type} size="small" variant="outlined" color={indicator.type === 'leading' ? 'primary' : 'default'} />
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 0.5 }}>
                <Typography variant="h5" sx={{ fontWeight: 700 }}>{indicator.current_value}</Typography>
                <Typography variant="caption" color="text.secondary">{indicator.unit}</Typography>
                <Chip label={indicator.trend} size="small" color={indicator.trend === 'improving' ? 'success' : indicator.trend === 'deteriorating' ? 'error' : 'default'} sx={{ ml: 'auto' }} />
              </Box>
              <ResponsiveContainer width="100%" height={80}>
                <LineChart data={chartData}><XAxis dataKey="date" hide /><YAxis hide domain={['auto', 'auto']} /><Tooltip />
                  <ReferenceLine y={indicator.threshold_warning} stroke="#F57F17" strokeDasharray="3 3" />
                  <ReferenceLine y={indicator.threshold_critical} stroke="#C62828" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="value" stroke="#0D47A1" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
        );
      })}
    </Grid>
  </CardContent></Card>
);

export default IndicatorCharts;
