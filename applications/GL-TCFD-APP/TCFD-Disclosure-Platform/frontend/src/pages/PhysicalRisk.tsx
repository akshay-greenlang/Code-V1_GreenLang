/**
 * PhysicalRisk - Asset map, risk cards, hazard distribution, and scenario projections.
 */

import React, { useMemo, useState } from 'react';
import { Grid, Card, CardContent, Typography, Box, Chip, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, LineChart, Line, Cell } from 'recharts';
import { LocationOn, Warning, WaterDrop, Thermostat } from '@mui/icons-material';
import StatCard from '../components/common/StatCard';
import RiskBadge from '../components/common/RiskBadge';

const ASSETS = [
  { id: '1', name: 'Hamburg Manufacturing Plant', type: 'Manufacturing', lat: 53.55, lng: 9.99, country: 'Germany', bookValue: 120000000, riskLevel: 'high' as const, hazards: ['flood', 'storm_surge'], riskScore: 78 },
  { id: '2', name: 'Singapore HQ', type: 'Real Estate', lat: 1.35, lng: 103.82, country: 'Singapore', bookValue: 85000000, riskLevel: 'medium' as const, hazards: ['extreme_heat', 'sea_level_rise'], riskScore: 55 },
  { id: '3', name: 'Texas Data Center', type: 'Data Center', lat: 30.27, lng: -97.74, country: 'USA', bookValue: 45000000, riskLevel: 'high' as const, hazards: ['extreme_heat', 'drought', 'wildfire'], riskScore: 72 },
  { id: '4', name: 'Mumbai Warehouse', type: 'Supply Chain', lat: 19.08, lng: 72.88, country: 'India', bookValue: 25000000, riskLevel: 'critical' as const, hazards: ['flood', 'tropical_cyclone', 'extreme_heat'], riskScore: 88 },
  { id: '5', name: 'Sydney Office', type: 'Real Estate', lat: -33.87, lng: 151.21, country: 'Australia', bookValue: 35000000, riskLevel: 'low' as const, hazards: ['wildfire'], riskScore: 32 },
  { id: '6', name: 'Rotterdam Port Facility', type: 'Infrastructure', lat: 51.92, lng: 4.48, country: 'Netherlands', bookValue: 95000000, riskLevel: 'high' as const, hazards: ['sea_level_rise', 'storm_surge', 'flood'], riskScore: 75 },
];

const HAZARD_DISTRIBUTION = [
  { hazard: 'Flood', count: 3, exposure: 240000000, color: '#0D47A1' },
  { hazard: 'Extreme Heat', count: 3, exposure: 155000000, color: '#C62828' },
  { hazard: 'Sea Level Rise', count: 2, exposure: 180000000, color: '#00838F' },
  { hazard: 'Storm Surge', count: 2, exposure: 215000000, color: '#4527A0' },
  { hazard: 'Wildfire', count: 2, exposure: 80000000, color: '#E65100' },
  { hazard: 'Drought', count: 1, exposure: 45000000, color: '#F57F17' },
  { hazard: 'Tropical Cyclone', count: 1, exposure: 25000000, color: '#1B5E20' },
];

const RCP_PROJECTIONS = [
  { year: 2025, rcp26: 2.1, rcp45: 2.3, rcp85: 2.5 },
  { year: 2030, rcp26: 2.5, rcp45: 3.2, rcp85: 4.1 },
  { year: 2040, rcp26: 3.0, rcp45: 5.1, rcp85: 8.5 },
  { year: 2050, rcp26: 3.2, rcp45: 7.8, rcp85: 15.2 },
  { year: 2060, rcp26: 3.4, rcp45: 10.5, rcp85: 24.0 },
  { year: 2070, rcp26: 3.5, rcp45: 13.2, rcp85: 35.0 },
  { year: 2080, rcp26: 3.6, rcp45: 15.8, rcp85: 48.0 },
];

const PhysicalRisk: React.FC = () => {
  const [scenarioFilter, setScenarioFilter] = useState('all');

  const totalExposure = ASSETS.reduce((sum, a) => sum + a.bookValue, 0);
  const assetsAtRisk = ASSETS.filter((a) => a.riskLevel === 'high' || a.riskLevel === 'critical').length;
  const avgRiskScore = Math.round(ASSETS.reduce((sum, a) => sum + a.riskScore, 0) / ASSETS.length);

  return (
    <Box>
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4">Physical Risk Assessment</Typography>
        <Typography variant="body2" color="text.secondary">
          Asset-level physical risk analysis across acute and chronic climate hazards
        </Typography>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Total Asset Exposure" value={totalExposure} format="currency" icon={<LocationOn />} color="primary" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Assets at High/Critical Risk" value={assetsAtRisk} icon={<Warning />} subtitle={`of ${ASSETS.length} total`} color="error" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Avg Risk Score" value={avgRiskScore} icon={<Thermostat />} subtitle="out of 100" color="warning" />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard title="Insurance Gap" value={18500000} format="currency" icon={<WaterDrop />} color="info" />
        </Grid>
      </Grid>

      {/* Map Placeholder + Asset Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={7}>
          <Card sx={{ height: 400 }}>
            <CardContent sx={{ height: '100%' }}>
              <Typography variant="h6" sx={{ mb: 1 }}>Asset Locations & Risk Exposure</Typography>
              <Box sx={{
                height: 'calc(100% - 40px)', backgroundColor: '#E8F5E9', borderRadius: 1,
                display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative',
              }}>
                <Typography variant="body2" color="text.secondary">
                  Interactive Leaflet map with {ASSETS.length} asset pins -- requires map tiles initialization
                </Typography>
                {/* Asset markers overlay */}
                {ASSETS.map((asset) => (
                  <Box key={asset.id} sx={{
                    position: 'absolute',
                    top: `${20 + Math.random() * 60}%`,
                    left: `${10 + Math.random() * 80}%`,
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                  }}>
                    <LocationOn sx={{
                      color: asset.riskLevel === 'critical' ? '#B71C1C' : asset.riskLevel === 'high' ? '#E65100' : asset.riskLevel === 'medium' ? '#F57F17' : '#2E7D32',
                      fontSize: 24,
                    }} />
                    <Typography variant="caption" sx={{ fontSize: '0.6rem', whiteSpace: 'nowrap' }}>
                      {asset.name.split(' ')[0]}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={5}>
          <Card sx={{ height: 400, overflow: 'auto' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Asset Risk Cards</Typography>
              {ASSETS.sort((a, b) => b.riskScore - a.riskScore).map((asset) => (
                <Box key={asset.id} sx={{ p: 1.5, mb: 1, border: '1px solid #E0E0E0', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{asset.name}</Typography>
                    <RiskBadge level={asset.riskLevel} size="small" />
                  </Box>
                  <Typography variant="caption" color="text.secondary">{asset.type} | {asset.country} | ${(asset.bookValue / 1e6).toFixed(0)}M</Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
                    {asset.hazards.map((h) => (
                      <Chip key={h} label={h.replace(/_/g, ' ')} size="small" variant="outlined" sx={{ fontSize: '0.65rem', height: 20, textTransform: 'capitalize' }} />
                    ))}
                  </Box>
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Hazard Distribution + RCP Projections */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Hazard Distribution (Asset Count & Exposure)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={HAZARD_DISTRIBUTION}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hazard" fontSize={11} />
                  <YAxis yAxisId="left" orientation="left" />
                  <YAxis yAxisId="right" orientation="right" tickFormatter={(v) => `$${(v / 1e6).toFixed(0)}M`} />
                  <Tooltip formatter={(v: number, name: string) => [name === 'count' ? v : `$${(v / 1e6).toFixed(0)}M`, name === 'count' ? 'Assets' : 'Exposure']} />
                  <Legend />
                  <Bar yAxisId="left" dataKey="count" name="Asset Count" fill="#0D47A1" />
                  <Bar yAxisId="right" dataKey="exposure" name="Exposure ($)">
                    {HAZARD_DISTRIBUTION.map((entry, idx) => (
                      <Cell key={idx} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Physical Risk Projections by RCP/SSP Scenario (%VAR)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={RCP_PROJECTIONS}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="year" />
                  <YAxis tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v: number) => [`${v.toFixed(1)}%`, '']} />
                  <Legend />
                  <Line type="monotone" dataKey="rcp26" stroke="#2E7D32" strokeWidth={2} name="RCP 2.6 / SSP1-2.6" />
                  <Line type="monotone" dataKey="rcp45" stroke="#EF6C00" strokeWidth={2} name="RCP 4.5 / SSP2-4.5" />
                  <Line type="monotone" dataKey="rcp85" stroke="#C62828" strokeWidth={2} name="RCP 8.5 / SSP5-8.5" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PhysicalRisk;
