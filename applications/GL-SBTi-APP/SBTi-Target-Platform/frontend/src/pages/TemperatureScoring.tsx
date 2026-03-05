/**
 * TemperatureScoring - Temperature rating by scope with peer ranking and trends.
 *
 * Displays temperature gauges, scope-level scores, peer comparison bar chart,
 * and historical temperature trend.
 */

import React, { useEffect, useMemo } from 'react';
import {
  Grid, Box, Typography, Card, CardContent, Chip, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
} from '@mui/material';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, Cell,
} from 'recharts';
import { Refresh, Thermostat } from '@mui/icons-material';
import TempGauge from '../components/temperature/TempGauge';
import ScopeTemperature from '../components/temperature/ScopeTemperature';
import PeerRanking from '../components/temperature/PeerRanking';
import PortfolioTemp from '../components/temperature/PortfolioTemp';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import {
  fetchTemperatureScore, fetchTemperatureTimeSeries, fetchPeerRanking, recalculateTemperature,
  selectTemperatureScore, selectTemperatureTimeSeries, selectPeerRanking, selectTemperatureLoading,
} from '../store/slices/temperatureSlice';
import { selectActiveOrgId } from '../store/slices/settingsSlice';

const DEMO_SCORE = {
  id: 'ts_1',
  organization_id: 'org_default',
  overall_temperature: 1.8,
  scope_1_temperature: 1.5,
  scope_2_temperature: 1.6,
  scope_3_temperature: 2.1,
  scope_1_2_temperature: 1.55,
  methodology: 'SBTi Temperature Rating' as const,
  confidence: 'high' as const,
  calculated_at: '2025-02-28',
};

const DEMO_TIME_SERIES = [
  { date: '2021-12', temperature: 2.5 },
  { date: '2022-06', temperature: 2.3 },
  { date: '2022-12', temperature: 2.1 },
  { date: '2023-06', temperature: 2.0 },
  { date: '2023-12', temperature: 1.9 },
  { date: '2024-06', temperature: 1.85 },
  { date: '2024-12', temperature: 1.8 },
];

const DEMO_PEERS = [
  { company_name: 'TechCorp Alpha', sector: 'Technology', temperature: 1.5, rank: 1 },
  { company_name: 'GreenLang Corp (You)', sector: 'Technology', temperature: 1.8, rank: 2 },
  { company_name: 'Digital Systems Inc', sector: 'Technology', temperature: 2.0, rank: 3 },
  { company_name: 'CloudNet Global', sector: 'Technology', temperature: 2.2, rank: 4 },
  { company_name: 'DataWave Solutions', sector: 'Technology', temperature: 2.4, rank: 5 },
  { company_name: 'InfoTech Holdings', sector: 'Technology', temperature: 2.7, rank: 6 },
  { company_name: 'MegaSoft Ltd', sector: 'Technology', temperature: 3.0, rank: 7 },
];

const TemperatureScoring: React.FC = () => {
  const dispatch = useAppDispatch();
  const orgId = useAppSelector(selectActiveOrgId);
  const score = useAppSelector(selectTemperatureScore);
  const timeSeries = useAppSelector(selectTemperatureTimeSeries);
  const peerRanking = useAppSelector(selectPeerRanking);
  const loading = useAppSelector(selectTemperatureLoading);

  useEffect(() => {
    dispatch(fetchTemperatureScore(orgId));
    dispatch(fetchTemperatureTimeSeries(orgId));
    dispatch(fetchPeerRanking(orgId));
  }, [dispatch, orgId]);

  const data = score || DEMO_SCORE;
  const trend = timeSeries.length > 0 ? timeSeries : DEMO_TIME_SERIES;
  const peers = peerRanking.length > 0 ? peerRanking : DEMO_PEERS;

  const getTempColor = (t: number) => t <= 1.5 ? '#1B5E20' : t <= 1.8 ? '#2E7D32' : t <= 2.0 ? '#EF6C00' : '#C62828';
  const getTempLabel = (t: number) => t <= 1.5 ? '1.5C Aligned' : t <= 1.8 ? 'Well Below 2C' : t <= 2.0 ? 'Below 2C' : 'Above 2C';

  const handleRecalculate = () => {
    dispatch(recalculateTemperature(orgId));
  };

  if (loading && !score) return <LoadingSpinner message="Loading temperature scores..." />;

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4">Temperature Scoring</Typography>
          <Typography variant="body2" color="text.secondary">
            SBTi temperature rating methodology applied to organizational targets
          </Typography>
        </Box>
        <Button variant="outlined" startIcon={<Refresh />} onClick={handleRecalculate}>
          Recalculate
        </Button>
      </Box>

      {/* Overall Temperature */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card sx={{ textAlign: 'center' }}>
            <CardContent>
              <Thermostat sx={{ fontSize: 40, color: getTempColor(data.overall_temperature), mb: 1 }} />
              <Typography variant="h2" sx={{ fontWeight: 700, color: getTempColor(data.overall_temperature) }}>
                {data.overall_temperature.toFixed(1)}&deg;C
              </Typography>
              <Chip
                label={getTempLabel(data.overall_temperature)}
                size="small"
                sx={{ mt: 1, backgroundColor: getTempColor(data.overall_temperature), color: 'white' }}
              />
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                Overall Temperature Score
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Scope Temperatures */}
        {[
          { label: 'Scope 1', temp: data.scope_1_temperature },
          { label: 'Scope 2', temp: data.scope_2_temperature },
          { label: 'Scope 3', temp: data.scope_3_temperature },
        ].map((scope) => (
          <Grid item xs={12} sm={4} md={3} key={scope.label}>
            <Card sx={{ textAlign: 'center' }}>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>{scope.label}</Typography>
                <Typography variant="h3" sx={{ fontWeight: 700, color: getTempColor(scope.temp) }}>
                  {scope.temp.toFixed(1)}&deg;C
                </Typography>
                <Chip
                  label={getTempLabel(scope.temp)}
                  size="small"
                  sx={{ mt: 1, fontSize: '0.65rem', backgroundColor: getTempColor(scope.temp) + '22', color: getTempColor(scope.temp) }}
                />
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Trend + Peer Ranking */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Temperature Score Trend</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trend}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={11} />
                  <YAxis domain={[1.0, 3.0]} tickFormatter={(v) => `${v.toFixed(1)}C`} />
                  <Tooltip formatter={(value: number) => [`${value.toFixed(2)}C`, 'Temperature']} />
                  {/* Reference lines for 1.5C and 2.0C thresholds */}
                  <Line type="monotone" dataKey="temperature" stroke="#1B5E20" strokeWidth={2.5} dot={{ r: 4 }} name="Temperature Score" />
                </LineChart>
              </ResponsiveContainer>
              <Box sx={{ display: 'flex', gap: 2, mt: 1, justifyContent: 'center' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 20, height: 2, backgroundColor: '#1B5E20' }} />
                  <Typography variant="caption">1.5C target</Typography>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 20, height: 2, backgroundColor: '#EF6C00' }} />
                  <Typography variant="caption">2.0C threshold</Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>Peer Comparison (Sector: Technology)</Typography>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={peers} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 3.5]} tickFormatter={(v) => `${v.toFixed(1)}C`} />
                  <YAxis type="category" dataKey="company_name" width={140} fontSize={11} />
                  <Tooltip formatter={(value: number) => [`${value.toFixed(1)}C`, 'Temperature']} />
                  <Bar dataKey="temperature" name="Temperature Score">
                    {peers.map((p: any, idx: number) => (
                      <Cell
                        key={idx}
                        fill={p.company_name.includes('You') ? '#1B5E20' : getTempColor(p.temperature)}
                        opacity={p.company_name.includes('You') ? 1 : 0.6}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Methodology Info */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 1 }}>Methodology</Typography>
          <Typography variant="body2" color="text.secondary">
            Temperature scores are calculated using the SBTi Temperature Rating methodology, which translates
            corporate emission reduction targets into an implied temperature rise. Scores are based on the
            ambition level and timeframe of validated targets. A score of 1.5C or below indicates alignment
            with the Paris Agreement goal. Confidence level: <strong>{data.confidence}</strong>. Last calculated: {data.calculated_at}.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default TemperatureScoring;
