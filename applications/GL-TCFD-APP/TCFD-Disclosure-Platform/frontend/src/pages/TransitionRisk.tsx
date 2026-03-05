/**
 * TransitionRisk - Policy timeline, technology disruption, market shifts, composite heat map.
 */

import React, { useMemo } from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, LinearProgress } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LineChart, Line, Cell } from 'recharts';
import { Policy, Memory, TrendingUp, Shield } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';
import RiskBadge from '../components/common/RiskBadge';

const POLICY_RISKS = [
  { id: '1', name: 'EU CBAM', jurisdiction: 'EU', effectiveDate: '2026-01-01', impact: 12500000, compliance: 'partial', riskLevel: 'high' },
  { id: '2', name: 'SEC Climate Disclosure', jurisdiction: 'USA', effectiveDate: '2026-06-01', impact: 3200000, compliance: 'compliant', riskLevel: 'medium' },
  { id: '3', name: 'UK ETS Enhancement', jurisdiction: 'UK', effectiveDate: '2025-04-01', impact: 4800000, compliance: 'partial', riskLevel: 'high' },
  { id: '4', name: 'CSRD Reporting', jurisdiction: 'EU', effectiveDate: '2025-01-01', impact: 2100000, compliance: 'compliant', riskLevel: 'low' },
  { id: '5', name: 'Japan GX Carbon Levy', jurisdiction: 'Japan', effectiveDate: '2028-01-01', impact: 5500000, compliance: 'non_compliant', riskLevel: 'medium' },
];

const TECH_RISKS = [
  { technology: 'Green Hydrogen', adoption2025: 5, adoption2035: 35, adoption2050: 65, revenueImpact: -8, costImpact: -15, readiness: 40 },
  { technology: 'Carbon Capture (CCUS)', adoption2025: 3, adoption2035: 20, adoption2050: 45, revenueImpact: -5, costImpact: -20, readiness: 30 },
  { technology: 'EV Fleet Electrification', adoption2025: 15, adoption2035: 60, adoption2050: 90, revenueImpact: 12, costImpact: -8, readiness: 65 },
  { technology: 'Battery Storage', adoption2025: 12, adoption2035: 45, adoption2050: 80, revenueImpact: 8, costImpact: -12, readiness: 55 },
  { technology: 'Circular Economy Tech', adoption2025: 8, adoption2035: 30, adoption2050: 60, revenueImpact: 15, costImpact: -10, readiness: 45 },
];

const MARKET_PROJECTIONS = [
  { year: 2025, green_premium: 5, carbon_adjusted_demand: -2, stranded_risk: 3 },
  { year: 2027, green_premium: 8, carbon_adjusted_demand: -5, stranded_risk: 7 },
  { year: 2030, green_premium: 15, carbon_adjusted_demand: -12, stranded_risk: 15 },
  { year: 2035, green_premium: 22, carbon_adjusted_demand: -20, stranded_risk: 28 },
  { year: 2040, green_premium: 28, carbon_adjusted_demand: -25, stranded_risk: 38 },
];

const HEAT_MAP_DATA = [
  [0, 0, 1, 0, 0],
  [0, 1, 2, 1, 0],
  [1, 2, 3, 2, 0],
  [0, 1, 2, 1, 0],
  [0, 0, 0, 1, 0],
];

const LIKELIHOOD_LABELS = ['Rare', 'Unlikely', 'Possible', 'Likely', 'Almost Certain'];
const IMPACT_LABELS = ['Insignificant', 'Minor', 'Moderate', 'Major', 'Catastrophic'];

const TransitionRisk: React.FC = () => {
  const totalExposure = POLICY_RISKS.reduce((sum, p) => sum + p.impact, 0);

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Transition Risk Assessment</Typography>
        <Typography variant="body2" color="text.secondary">
          Policy, technology, market, and reputation transition risk analysis
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Total Transition Exposure" value={totalExposure} format="currency" icon={<Policy />} color="error" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Policy Risks" value={POLICY_RISKS.length} icon={<Policy />} subtitle="Across 4 jurisdictions" color="secondary" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Technology Disruptions" value={TECH_RISKS.length} icon={<Memory />} subtitle="Monitored technologies" color="info" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Stranded Asset Risk" value={15} format="percent" icon={<TrendingUp />} trend={5} trendLabel="increasing" color="warning" />
        </Grid>
      </Grid>

      {/* Policy Timeline + Heat Map */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={7}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Policy Risk Timeline</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Regulation</TableCell>
                      <TableCell>Jurisdiction</TableCell>
                      <TableCell>Effective Date</TableCell>
                      <TableCell>Risk Level</TableCell>
                      <TableCell align="right">Impact</TableCell>
                      <TableCell>Compliance</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {POLICY_RISKS.sort((a, b) => new Date(a.effectiveDate).getTime() - new Date(b.effectiveDate).getTime()).map((p) => (
                      <TableRow key={p.id} hover>
                        <TableCell sx={{ fontWeight: 500 }}>{p.name}</TableCell>
                        <TableCell><Chip label={p.jurisdiction} size="small" variant="outlined" /></TableCell>
                        <TableCell>{new Date(p.effectiveDate).toLocaleDateString()}</TableCell>
                        <TableCell><RiskBadge level={p.riskLevel} size="small" /></TableCell>
                        <TableCell align="right">${(p.impact / 1e6).toFixed(1)}M</TableCell>
                        <TableCell>
                          <Chip label={p.compliance.replace(/_/g, ' ')} size="small" color={p.compliance === 'compliant' ? 'success' : p.compliance === 'partial' ? 'warning' : 'error'} sx={{ textTransform: 'capitalize' }} />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={5}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Composite Risk Heat Map</Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.5, mb: 0.5, pr: 1 }}>
                  {IMPACT_LABELS.map((label) => (
                    <Typography key={label} variant="caption" sx={{ width: 52, textAlign: 'center', fontSize: '0.6rem' }}>
                      {label}
                    </Typography>
                  ))}
                </Box>
                {HEAT_MAP_DATA.slice().reverse().map((row, rowIdx) => (
                  <Box key={rowIdx} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Typography variant="caption" sx={{ width: 80, textAlign: 'right', pr: 1, fontSize: '0.65rem' }}>
                      {LIKELIHOOD_LABELS[4 - rowIdx]}
                    </Typography>
                    {row.map((count, colIdx) => (
                      <Box
                        key={colIdx}
                        sx={{
                          width: 52, height: 40, borderRadius: 1, display: 'flex',
                          alignItems: 'center', justifyContent: 'center',
                          backgroundColor: count === 0 ? '#E8F5E9' : count === 1 ? '#FFF9C4' : count === 2 ? '#FFE0B2' : '#FFCDD2',
                          border: '1px solid #E0E0E0',
                        }}
                      >
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>{count || ''}</Typography>
                      </Box>
                    ))}
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Technology Disruption + Market Projections */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Technology Disruption Readiness</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Technology</TableCell>
                      <TableCell align="center">Adoption 2035</TableCell>
                      <TableCell align="center">Readiness</TableCell>
                      <TableCell align="center">Revenue Impact</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {TECH_RISKS.map((t) => (
                      <TableRow key={t.technology} hover>
                        <TableCell sx={{ fontWeight: 500 }}>{t.technology}</TableCell>
                        <TableCell align="center">{t.adoption2035}%</TableCell>
                        <TableCell align="center">
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <LinearProgress variant="determinate" value={t.readiness} sx={{ width: 60, height: 6, borderRadius: 3 }} />
                            <Typography variant="caption">{t.readiness}%</Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="center">
                          <Typography variant="body2" sx={{ color: t.revenueImpact >= 0 ? 'success.main' : 'error.main', fontWeight: 500 }}>
                            {t.revenueImpact >= 0 ? '+' : ''}{t.revenueImpact}%
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Market Shift Projections (% Impact)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={MARKET_PROJECTIONS}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                  <Legend />
                  <Line type="monotone" dataKey="green_premium" stroke="#2E7D32" strokeWidth={2} name="Green Premium" />
                  <Line type="monotone" dataKey="carbon_adjusted_demand" stroke="#C62828" strokeWidth={2} name="Carbon-Adjusted Demand" />
                  <Line type="monotone" dataKey="stranded_risk" stroke="#E65100" strokeWidth={2} strokeDasharray="5 5" name="Stranded Asset Risk" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TransitionRisk;
