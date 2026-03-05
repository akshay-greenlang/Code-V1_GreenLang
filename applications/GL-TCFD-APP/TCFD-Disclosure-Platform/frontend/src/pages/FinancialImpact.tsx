/**
 * FinancialImpact - Three-tab view (IS/BS/CF), MACC chart, carbon price sensitivity, NPV table.
 */

import React, { useState, useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Tabs, Tab, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Chip } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell, LineChart, Line, ReferenceLine, ScatterChart, Scatter, ZAxis } from 'recharts';
import { AttachMoney, TrendingDown, Assessment, ShowChart } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';

const IS_DATA = [
  { item: 'Revenue', baseline: 500, impact: -18, adjusted: 482, pct: -3.6, drivers: 'Demand shift, carbon costs' },
  { item: 'Cost of Goods Sold', baseline: -320, impact: -25, adjusted: -345, pct: 7.8, drivers: 'Energy costs, carbon pricing' },
  { item: 'Gross Profit', baseline: 180, impact: -43, adjusted: 137, pct: -23.9, drivers: '' },
  { item: 'Operating Expenses', baseline: -80, impact: -8, adjusted: -88, pct: 10.0, drivers: 'Compliance costs' },
  { item: 'Climate Opportunity Revenue', baseline: 0, impact: 32, adjusted: 32, pct: 0, drivers: 'Green products, services' },
  { item: 'EBITDA', baseline: 100, impact: -19, adjusted: 81, pct: -19.0, drivers: '' },
];

const BS_DATA = [
  { item: 'Property, Plant & Equipment', baseline: 450, impact: -35, adjusted: 415, pct: -7.8, drivers: 'Stranded assets, impairment' },
  { item: 'Intangible Assets', baseline: 120, impact: -8, adjusted: 112, pct: -6.7, drivers: 'IP devaluation' },
  { item: 'Right-of-Use Assets', baseline: 85, impact: -5, adjusted: 80, pct: -5.9, drivers: 'Lease adjustments' },
  { item: 'Environmental Provisions', baseline: -15, impact: -22, adjusted: -37, pct: 146.7, drivers: 'Carbon liabilities' },
  { item: 'Green Bond Proceeds', baseline: 0, impact: 50, adjusted: 50, pct: 0, drivers: 'Sustainability financing' },
];

const CF_DATA = [
  { item: 'Operating Cash Flow', baseline: 110, impact: -15, adjusted: 95, pct: -13.6, drivers: 'Lower margins' },
  { item: 'CapEx - Green Investments', baseline: -40, impact: -28, adjusted: -68, pct: 70.0, drivers: 'Transition investments' },
  { item: 'CapEx - Adaptation', baseline: -5, impact: -12, adjusted: -17, pct: 240.0, drivers: 'Physical risk adaptation' },
  { item: 'Carbon Cost Payments', baseline: 0, impact: -18, adjusted: -18, pct: 0, drivers: 'ETS, carbon tax' },
  { item: 'Green Revenue Inflows', baseline: 0, impact: 25, adjusted: 25, pct: 0, drivers: 'New product lines' },
  { item: 'Free Cash Flow', baseline: 65, impact: -48, adjusted: 17, pct: -73.8, drivers: '' },
];

const MACC_DATA = [
  { measure: 'LED Lighting', abatement: 2500, cost: -45, investment: 0.5, color: '#2E7D32' },
  { measure: 'HVAC Optimization', abatement: 4200, cost: -30, investment: 1.2, color: '#4CAF50' },
  { measure: 'Solar PV Installation', abatement: 8500, cost: -15, investment: 5.0, color: '#66BB6A' },
  { measure: 'Fleet Electrification', abatement: 6200, cost: 10, investment: 8.0, color: '#FFA726' },
  { measure: 'Process Heat Pump', abatement: 3800, cost: 35, investment: 4.5, color: '#EF5350' },
  { measure: 'Green Hydrogen', abatement: 5000, cost: 85, investment: 12.0, color: '#C62828' },
  { measure: 'Carbon Capture', abatement: 7500, cost: 120, investment: 20.0, color: '#B71C1C' },
];

const CARBON_SENSITIVITY = [
  { price: 0, nze: 0, aps: 0, steps: 0 },
  { price: 25, nze: -2, aps: -2, steps: -2 },
  { price: 50, nze: -5, aps: -5, steps: -5 },
  { price: 100, nze: -12, aps: -10, steps: -8 },
  { price: 150, nze: -20, aps: -16, steps: -12 },
  { price: 200, nze: -30, aps: -24, steps: -18 },
  { price: 250, nze: -42, aps: -32, steps: -24 },
];

const FinancialImpact: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  const activeData = activeTab === 0 ? IS_DATA : activeTab === 1 ? BS_DATA : CF_DATA;
  const tabLabels = ['Income Statement', 'Balance Sheet', 'Cash Flow'];

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Financial Impact Analysis</Typography>
        <Typography variant="body2" color="text.secondary">
          Climate-adjusted financial statements, MACC, and carbon price sensitivity
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="EBITDA Impact" value={-19} format="percent" icon={<TrendingDown />} color="error" subtitle="NZE 2050 scenario" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Stranded Asset Risk" value={35000000} format="currency" icon={<AttachMoney />} color="warning" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Abatement Potential" value={37700} icon={<Assessment />} subtitle="tCO2e from MACC" color="success" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Green Investment Req." value={51200000} format="currency" icon={<ShowChart />} color="info" />
        </Grid>
      </Grid>

      {/* Financial Statement Tabs */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ mb: 2 }}>
            {tabLabels.map((label) => <Tab key={label} label={label} />)}
          </Tabs>
          <Typography variant="h6" sx={{ mb: 2 }}>Climate-Adjusted {tabLabels[activeTab]} ($M) -- NZE 2050</Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Line Item</TableCell>
                  <TableCell align="right">Baseline</TableCell>
                  <TableCell align="right">Climate Impact</TableCell>
                  <TableCell align="right">Adjusted</TableCell>
                  <TableCell align="right">Change %</TableCell>
                  <TableCell>Key Drivers</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {activeData.map((row) => (
                  <TableRow key={row.item} hover sx={{ backgroundColor: row.drivers === '' ? '#FAFAFA' : undefined }}>
                    <TableCell sx={{ fontWeight: row.drivers === '' ? 700 : 400 }}>{row.item}</TableCell>
                    <TableCell align="right">${row.baseline}</TableCell>
                    <TableCell align="right" sx={{ color: row.impact >= 0 ? 'success.main' : 'error.main', fontWeight: 500 }}>
                      {row.impact >= 0 ? '+' : ''}{row.impact}
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>${row.adjusted}</TableCell>
                    <TableCell align="right">
                      {row.pct !== 0 && (
                        <Chip
                          label={`${row.pct >= 0 ? '+' : ''}${row.pct.toFixed(1)}%`}
                          size="small"
                          sx={{ backgroundColor: Math.abs(row.pct) > 20 ? '#FFCDD2' : Math.abs(row.pct) > 10 ? '#FFE0B2' : '#E8F5E9', fontWeight: 500 }}
                        />
                      )}
                    </TableCell>
                    <TableCell sx={{ fontSize: '0.8rem', color: 'text.secondary' }}>{row.drivers}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* MACC Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Marginal Abatement Cost Curve ($/tCO2e)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={MACC_DATA}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="measure" fontSize={10} angle={-20} textAnchor="end" height={60} />
                  <YAxis tickFormatter={(v) => `$${v}`} label={{ value: '$/tCO2e', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(v: number, name: string) => [name === 'cost' ? `$${v}/tCO2e` : `${v.toLocaleString()} tCO2e`, name === 'cost' ? 'Abatement Cost' : 'Potential']} />
                  <ReferenceLine y={0} stroke="#666" />
                  <Bar dataKey="cost" name="Cost">
                    {MACC_DATA.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Carbon Price Sensitivity */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Carbon Price Sensitivity ($M Impact)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={CARBON_SENSITIVITY}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="price" label={{ value: 'Carbon Price ($/tCO2)', position: 'bottom', offset: -5 }} />
                  <YAxis tickFormatter={(v) => `$${v}M`} />
                  <Tooltip formatter={(v: number) => [`$${v}M`, '']} />
                  <Legend />
                  <ReferenceLine y={0} stroke="#666" />
                  <Line type="monotone" dataKey="nze" stroke="#1B5E20" strokeWidth={2} name="NZE 2050" />
                  <Line type="monotone" dataKey="aps" stroke="#0D47A1" strokeWidth={2} name="APS" />
                  <Line type="monotone" dataKey="steps" stroke="#E65100" strokeWidth={2} name="STEPS" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FinancialImpact;
