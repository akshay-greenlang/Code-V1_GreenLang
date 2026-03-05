/**
 * PeerComparison - Benchmarking chart comparing organizational TCFD maturity against peers.
 *
 * Displays grouped bar chart comparing pillar scores against peer companies,
 * industry average, and best-in-class. Includes percentile ranking and insights.
 */

import React, { useMemo, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Grid,
  LinearProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Cell,
} from 'recharts';
import { CompareArrows, TrendingUp, Star, Groups } from '@mui/icons-material';

interface PeerScore {
  company: string;
  is_self: boolean;
  governance: number;
  strategy: number;
  risk_management: number;
  metrics_targets: number;
  overall: number;
  industry: string;
  sbti_committed: boolean;
  tcfd_supporter: boolean;
}

interface PeerComparisonProps {
  data: PeerScore[];
  selfCompanyName?: string;
}

const PILLAR_COLORS: Record<string, string> = {
  governance: '#1B5E20',
  strategy: '#0D47A1',
  risk_management: '#E65100',
  metrics_targets: '#6A1B9A',
};

const PILLAR_LABELS: Record<string, string> = {
  governance: 'Governance',
  strategy: 'Strategy',
  risk_management: 'Risk Mgmt',
  metrics_targets: 'Metrics & Targets',
};

const PeerComparison: React.FC<PeerComparisonProps> = ({ data, selfCompanyName }) => {
  const [viewMode, setViewMode] = useState<'bar' | 'radar'>('bar');

  const selfData = useMemo(
    () => data.find((d) => d.is_self) || data[0],
    [data]
  );

  const peers = useMemo(
    () => data.filter((d) => !d.is_self),
    [data]
  );

  const industryAvg = useMemo(() => {
    if (peers.length === 0) return { governance: 0, strategy: 0, risk_management: 0, metrics_targets: 0, overall: 0 };
    return {
      governance: Math.round(peers.reduce((s, p) => s + p.governance, 0) / peers.length),
      strategy: Math.round(peers.reduce((s, p) => s + p.strategy, 0) / peers.length),
      risk_management: Math.round(peers.reduce((s, p) => s + p.risk_management, 0) / peers.length),
      metrics_targets: Math.round(peers.reduce((s, p) => s + p.metrics_targets, 0) / peers.length),
      overall: Math.round(peers.reduce((s, p) => s + p.overall, 0) / peers.length),
    };
  }, [peers]);

  const bestInClass = useMemo(() => ({
    governance: Math.max(...data.map((d) => d.governance)),
    strategy: Math.max(...data.map((d) => d.strategy)),
    risk_management: Math.max(...data.map((d) => d.risk_management)),
    metrics_targets: Math.max(...data.map((d) => d.metrics_targets)),
    overall: Math.max(...data.map((d) => d.overall)),
  }), [data]);

  const percentileRank = useMemo(() => {
    if (!selfData) return 0;
    const sortedOverall = data.map((d) => d.overall).sort((a, b) => a - b);
    const rank = sortedOverall.indexOf(selfData.overall);
    return Math.round(((rank + 1) / sortedOverall.length) * 100);
  }, [data, selfData]);

  const barData = useMemo(() => {
    const pillars = ['governance', 'strategy', 'risk_management', 'metrics_targets'] as const;
    return pillars.map((pillar) => ({
      pillar: PILLAR_LABELS[pillar],
      'Our Company': selfData?.[pillar] || 0,
      'Industry Avg': industryAvg[pillar],
      'Best in Class': bestInClass[pillar],
    }));
  }, [selfData, industryAvg, bestInClass]);

  const radarData = useMemo(() => {
    const pillars = ['governance', 'strategy', 'risk_management', 'metrics_targets'] as const;
    return pillars.map((pillar) => ({
      dimension: PILLAR_LABELS[pillar],
      self: selfData?.[pillar] || 0,
      avg: industryAvg[pillar],
      best: bestInClass[pillar],
    }));
  }, [selfData, industryAvg, bestInClass]);

  const pillarGaps = useMemo(() => {
    const pillars = ['governance', 'strategy', 'risk_management', 'metrics_targets'] as const;
    return pillars.map((pillar) => ({
      pillar: PILLAR_LABELS[pillar],
      score: selfData?.[pillar] || 0,
      vsAvg: (selfData?.[pillar] || 0) - industryAvg[pillar],
      vsBest: (selfData?.[pillar] || 0) - bestInClass[pillar],
      color: PILLAR_COLORS[pillar],
    }));
  }, [selfData, industryAvg, bestInClass]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Peer Benchmarking
            </Typography>
            <Typography variant="body2" color="text.secondary">
              TCFD maturity comparison against {peers.length} peer companies
            </Typography>
          </Box>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>View</InputLabel>
            <Select
              value={viewMode}
              label="View"
              onChange={(e: SelectChangeEvent) => setViewMode(e.target.value as 'bar' | 'radar')}
            >
              <MenuItem value="bar">Bar Chart</MenuItem>
              <MenuItem value="radar">Radar Chart</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Summary KPIs */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1, backgroundColor: '#E8F5E9', borderRadius: 1 }}>
              <Star sx={{ color: '#1B5E20' }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{selfData?.overall || 0}%</Typography>
              <Typography variant="caption">Overall Score</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1, backgroundColor: '#E3F2FD', borderRadius: 1 }}>
              <CompareArrows sx={{ color: '#0D47A1' }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                {selfData ? selfData.overall - industryAvg.overall : 0}pts
              </Typography>
              <Typography variant="caption">vs Industry Avg</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1, backgroundColor: '#FFF3E0', borderRadius: 1 }}>
              <TrendingUp sx={{ color: '#E65100' }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>P{percentileRank}</Typography>
              <Typography variant="caption">Percentile Rank</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box sx={{ textAlign: 'center', p: 1, backgroundColor: '#F3E5F5', borderRadius: 1 }}>
              <Groups sx={{ color: '#6A1B9A' }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{peers.length}</Typography>
              <Typography variant="caption">Peer Companies</Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Charts */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={7}>
            {viewMode === 'bar' ? (
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="pillar" fontSize={11} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                  <Legend />
                  <Bar dataKey="Our Company" fill="#1B5E20" barSize={20} />
                  <Bar dataKey="Industry Avg" fill="#BDBDBD" barSize={20} />
                  <Bar dataKey="Best in Class" fill="#F57F17" barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height={350}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="dimension" fontSize={11} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar name="Our Company" dataKey="self" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.3} strokeWidth={2} />
                  <Radar name="Industry Avg" dataKey="avg" stroke="#9E9E9E" fill="none" strokeWidth={1.5} strokeDasharray="5 5" />
                  <Radar name="Best in Class" dataKey="best" stroke="#F57F17" fill="none" strokeWidth={1.5} strokeDasharray="3 3" />
                  <Legend />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </Grid>

          {/* Pillar Gap Table */}
          <Grid item xs={12} md={5}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Pillar-Level Analysis
            </Typography>
            {pillarGaps.map((pg) => (
              <Box key={pg.pillar} sx={{ mb: 1.5 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>{pg.pillar}</Typography>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <Chip
                      label={`${pg.vsAvg >= 0 ? '+' : ''}${pg.vsAvg}`}
                      size="small"
                      sx={{
                        backgroundColor: pg.vsAvg >= 0 ? '#E8F5E9' : '#FFEBEE',
                        color: pg.vsAvg >= 0 ? '#2E7D32' : '#C62828',
                        fontWeight: 600,
                        fontSize: '0.65rem',
                        height: 20,
                      }}
                    />
                    <Typography variant="caption" sx={{ fontWeight: 600 }}>
                      {pg.score}%
                    </Typography>
                  </Box>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={pg.score}
                  sx={{
                    height: 8,
                    borderRadius: 4,
                    backgroundColor: '#E0E0E0',
                    '& .MuiLinearProgress-bar': { backgroundColor: pg.color },
                  }}
                />
              </Box>
            ))}

            <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 3, mb: 1 }}>
              Peer Leaderboard
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>#</TableCell>
                    <TableCell>Company</TableCell>
                    <TableCell align="right">Score</TableCell>
                    <TableCell align="center">SBTi</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data
                    .sort((a, b) => b.overall - a.overall)
                    .map((peer, idx) => (
                      <TableRow
                        key={peer.company}
                        hover
                        sx={{
                          backgroundColor: peer.is_self ? '#E8F5E9' : 'transparent',
                        }}
                      >
                        <TableCell sx={{ fontWeight: 500 }}>{idx + 1}</TableCell>
                        <TableCell sx={{ fontWeight: peer.is_self ? 700 : 400, fontSize: '0.8rem' }}>
                          {peer.company}
                          {peer.tcfd_supporter && (
                            <Chip label="TCFD" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.55rem' }} />
                          )}
                        </TableCell>
                        <TableCell align="right" sx={{ fontWeight: 600 }}>
                          {peer.overall}%
                        </TableCell>
                        <TableCell align="center">
                          {peer.sbti_committed ? (
                            <Chip label="Y" size="small" color="success" sx={{ height: 18, fontSize: '0.6rem' }} />
                          ) : (
                            <Chip label="N" size="small" variant="outlined" sx={{ height: 18, fontSize: '0.6rem' }} />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default PeerComparison;
