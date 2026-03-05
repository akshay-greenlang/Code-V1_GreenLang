/**
 * DualScorecard - Side-by-side TCFD and ISSB/IFRS S2 compliance scorecard.
 *
 * Displays radar chart and bar chart comparing compliance scores across both
 * frameworks with dimension-level drill-down and gap highlighting.
 */

import React, { useMemo, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';

interface ScorecardDimension {
  dimension: string;
  pillar: string;
  tcfd_score: number;
  issb_score: number;
  gap: number;
  key_difference: string;
}

interface DualScorecardProps {
  dimensions: ScorecardDimension[];
  tcfdOverall?: number;
  issbOverall?: number;
}

const PILLAR_COLORS: Record<string, string> = {
  Governance: '#1B5E20',
  Strategy: '#0D47A1',
  'Risk Management': '#E65100',
  'Metrics & Targets': '#6A1B9A',
};

const DualScorecard: React.FC<DualScorecardProps> = ({
  dimensions,
  tcfdOverall: tcfdOverallProp,
  issbOverall: issbOverallProp,
}) => {
  const [chartType, setChartType] = useState<'radar' | 'bar'>('radar');

  const tcfdOverall = useMemo(
    () => tcfdOverallProp ?? Math.round(dimensions.reduce((s, d) => s + d.tcfd_score, 0) / dimensions.length),
    [dimensions, tcfdOverallProp]
  );
  const issbOverall = useMemo(
    () => issbOverallProp ?? Math.round(dimensions.reduce((s, d) => s + d.issb_score, 0) / dimensions.length),
    [dimensions, issbOverallProp]
  );
  const overallGap = tcfdOverall - issbOverall;

  const pillarSummary = useMemo(() => {
    const pillars = Array.from(new Set(dimensions.map((d) => d.pillar)));
    return pillars.map((pillar) => {
      const dims = dimensions.filter((d) => d.pillar === pillar);
      const tcfd = Math.round(dims.reduce((s, d) => s + d.tcfd_score, 0) / dims.length);
      const issb = Math.round(dims.reduce((s, d) => s + d.issb_score, 0) / dims.length);
      return { pillar, tcfd, issb, gap: tcfd - issb, color: PILLAR_COLORS[pillar] || '#9E9E9E' };
    });
  }, [dimensions]);

  const largestGaps = useMemo(
    () => [...dimensions].sort((a, b) => b.gap - a.gap).slice(0, 5),
    [dimensions]
  );

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              TCFD / ISSB Dual Compliance Scorecard
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Side-by-side framework compliance assessment
            </Typography>
          </Box>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={(_, val) => val && setChartType(val)}
            size="small"
          >
            <ToggleButton value="radar">Radar</ToggleButton>
            <ToggleButton value="bar">Bar</ToggleButton>
          </ToggleButtonGroup>
        </Box>

        {/* Overall Score Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={4}>
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: '2px solid #1B5E20',
                backgroundColor: '#E8F5E9',
                textAlign: 'center',
              }}
            >
              <Typography variant="caption" sx={{ fontWeight: 600, color: '#1B5E20', letterSpacing: 1 }}>
                TCFD COMPLIANCE
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#1B5E20' }}>
                {tcfdOverall}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={tcfdOverall}
                sx={{ height: 6, borderRadius: 3, mt: 1, '& .MuiLinearProgress-bar': { backgroundColor: '#1B5E20' } }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: '2px solid #0D47A1',
                backgroundColor: '#E3F2FD',
                textAlign: 'center',
              }}
            >
              <Typography variant="caption" sx={{ fontWeight: 600, color: '#0D47A1', letterSpacing: 1 }}>
                ISSB / IFRS S2 COMPLIANCE
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: '#0D47A1' }}>
                {issbOverall}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={issbOverall}
                sx={{ height: 6, borderRadius: 3, mt: 1, '& .MuiLinearProgress-bar': { backgroundColor: '#0D47A1' } }}
              />
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Box
              sx={{
                p: 2,
                borderRadius: 1,
                border: '1px solid #E0E0E0',
                textAlign: 'center',
              }}
            >
              <Typography variant="caption" sx={{ fontWeight: 600, letterSpacing: 1 }}>
                MIGRATION GAP
              </Typography>
              <Typography
                variant="h3"
                sx={{
                  fontWeight: 700,
                  color: overallGap > 20 ? '#C62828' : overallGap > 10 ? '#EF6C00' : '#2E7D32',
                }}
              >
                {overallGap}pts
              </Typography>
              <Typography variant="caption" color="text.secondary">
                TCFD ahead of ISSB readiness
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Charts */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={7}>
            {chartType === 'radar' ? (
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={dimensions}>
                  <PolarGrid gridType="polygon" />
                  <PolarAngleAxis dataKey="dimension" fontSize={9} tick={{ fill: '#616161' }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
                  <Radar
                    name="TCFD"
                    dataKey="tcfd_score"
                    stroke="#1B5E20"
                    fill="#1B5E20"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                  <Radar
                    name="ISSB"
                    dataKey="issb_score"
                    stroke="#0D47A1"
                    fill="#0D47A1"
                    fillOpacity={0.2}
                    strokeWidth={2}
                  />
                  <Legend />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={dimensions} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="dimension" width={120} fontSize={10} />
                  <Tooltip formatter={(v: number) => [`${v}%`, '']} />
                  <Legend />
                  <Bar dataKey="tcfd_score" name="TCFD" fill="#1B5E20" barSize={8} />
                  <Bar dataKey="issb_score" name="ISSB" fill="#0D47A1" barSize={8} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </Grid>

          {/* Gap Details */}
          <Grid item xs={12} md={5}>
            {/* Pillar Summary */}
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
              Pillar-Level Comparison
            </Typography>
            {pillarSummary.map((ps) => (
              <Box key={ps.pillar} sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                  <Typography variant="body2" sx={{ fontWeight: 500, color: ps.color }}>
                    {ps.pillar}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Chip label={`TCFD: ${ps.tcfd}%`} size="small" sx={{ backgroundColor: '#E8F5E9', fontSize: '0.65rem', height: 20 }} />
                    <Chip label={`ISSB: ${ps.issb}%`} size="small" sx={{ backgroundColor: '#E3F2FD', fontSize: '0.65rem', height: 20 }} />
                  </Box>
                </Box>
                <Box sx={{ display: 'flex', gap: 0.5, height: 6 }}>
                  <Box sx={{ flex: 1, position: 'relative', backgroundColor: '#E0E0E0', borderRadius: 3, overflow: 'hidden' }}>
                    <Box sx={{ position: 'absolute', height: '100%', width: `${ps.tcfd}%`, backgroundColor: '#1B5E20', borderRadius: 3 }} />
                  </Box>
                  <Box sx={{ flex: 1, position: 'relative', backgroundColor: '#E0E0E0', borderRadius: 3, overflow: 'hidden' }}>
                    <Box sx={{ position: 'absolute', height: '100%', width: `${ps.issb}%`, backgroundColor: '#0D47A1', borderRadius: 3 }} />
                  </Box>
                </Box>
              </Box>
            ))}

            {/* Largest Gaps */}
            <Typography variant="subtitle2" sx={{ fontWeight: 600, mt: 3, mb: 1 }}>
              Largest TCFD-to-ISSB Gaps
            </Typography>
            {largestGaps.map((g) => (
              <Box
                key={g.dimension}
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  py: 0.5,
                  borderBottom: '1px solid #F0F0F0',
                }}
              >
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" sx={{ fontSize: '0.8rem', fontWeight: 500 }}>
                    {g.dimension}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                    {g.key_difference}
                  </Typography>
                </Box>
                <Chip
                  label={`${g.gap}pts`}
                  size="small"
                  sx={{
                    backgroundColor: g.gap > 25 ? '#FFEBEE' : g.gap > 15 ? '#FFF3E0' : '#E8F5E9',
                    color: g.gap > 25 ? '#C62828' : g.gap > 15 ? '#E65100' : '#2E7D32',
                    fontWeight: 600,
                    fontSize: '0.7rem',
                    ml: 1,
                  }}
                />
              </Box>
            ))}
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default DualScorecard;
