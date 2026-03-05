/**
 * CompareYoY - Year-over-year alignment comparison chart.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';

const DEMO_DATA = [
  { year: '2022', turnover: 15.0, capex: 22.0, opex: 18.0 },
  { year: '2023', turnover: 25.0, capex: 35.0, opex: 28.0 },
  { year: '2024', turnover: 35.5, capex: 45.0, opex: 38.0 },
  { year: '2025', turnover: 42.0, capex: 56.0, opex: 46.7 },
];

const KPI_COLORS = { turnover: '#1B5E20', capex: '#0D47A1', opex: '#EF6C00' };

const CompareYoY: React.FC = () => {
  const latestYear = DEMO_DATA[DEMO_DATA.length - 1];
  const priorYear = DEMO_DATA[DEMO_DATA.length - 2];
  const turnoverDelta = +(latestYear.turnover - priorYear.turnover).toFixed(1);
  const capexDelta = +(latestYear.capex - priorYear.capex).toFixed(1);
  const opexDelta = +(latestYear.opex - priorYear.opex).toFixed(1);

  const renderDelta = (delta: number) => {
    if (delta > 0) return <Chip icon={<TrendingUp />} label={`+${delta}pp`} size="small" color="success" />;
    if (delta < 0) return <Chip icon={<TrendingDown />} label={`${delta}pp`} size="small" color="error" />;
    return <Chip icon={<TrendingFlat />} label="0pp" size="small" />;
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 1 }}>Year-over-Year Alignment</Typography>
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>Turnover:</Typography>
            {renderDelta(turnoverDelta)}
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>CapEx:</Typography>
            {renderDelta(capexDelta)}
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 500 }}>OpEx:</Typography>
            {renderDelta(opexDelta)}
          </Box>
        </Box>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={DEMO_DATA}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis tickFormatter={(v) => `${v}%`} domain={[0, 70]} />
            <Tooltip formatter={(v: number) => [`${v}%`, '']} />
            <Legend />
            <Bar dataKey="turnover" name="Turnover %" fill={KPI_COLORS.turnover} barSize={18} />
            <Bar dataKey="capex" name="CapEx %" fill={KPI_COLORS.capex} barSize={18} />
            <Bar dataKey="opex" name="OpEx %" fill={KPI_COLORS.opex} barSize={18} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default CompareYoY;
