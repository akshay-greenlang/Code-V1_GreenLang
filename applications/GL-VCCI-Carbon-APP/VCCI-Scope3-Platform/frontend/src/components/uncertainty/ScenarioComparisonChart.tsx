/**
 * ScenarioComparisonChart - Box-and-whisker comparison of emission scenarios
 *
 * Renders 2-5 scenarios side by side as custom box-and-whisker plots built
 * with Recharts. Each box shows P25-P75 (IQR), whiskers at P5 and P95,
 * and a median line at P50. Includes an optional target line and computes
 * basic statistical significance indicators between adjacent scenarios.
 */

import React, { useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
  ErrorBar,
  Scatter,
  ZAxis,
} from 'recharts';
import {
  Paper,
  Typography,
  Box,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Alert,
} from '@mui/material';
import { TrendingDown, TrendingUp, CompareArrows } from '@mui/icons-material';
import { formatNumber } from '../../utils/formatters';

// ==============================================================================
// Types
// ==============================================================================

interface ScenarioData {
  name: string;
  mean: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
  color?: string;
}

interface ScenarioComparisonChartProps {
  scenarios: ScenarioData[];
  target?: number;
  unit?: string;
  title?: string;
}

interface BoxPlotData {
  name: string;
  boxBottom: number;
  boxHeight: number;
  whiskerLow: number;
  whiskerHigh: number;
  median: number;
  mean: number;
  p5: number;
  p25: number;
  p75: number;
  p95: number;
  color: string;
  probabilityBelowTarget: number | null;
}

// ==============================================================================
// Constants
// ==============================================================================

const DEFAULT_COLORS = ['#1976d2', '#4caf50', '#ff9800', '#9c27b0', '#f44336'];

// ==============================================================================
// Helpers
// ==============================================================================

/**
 * Estimate the probability that a scenario falls below the target value,
 * assuming a roughly normal distribution parameterized by the given percentiles.
 * Uses a linear interpolation between percentile bands.
 */
const estimateProbabilityBelowTarget = (scenario: ScenarioData, target: number): number => {
  const percentiles = [
    { p: 0.05, value: scenario.p5 },
    { p: 0.25, value: scenario.p25 },
    { p: 0.50, value: scenario.p50 },
    { p: 0.75, value: scenario.p75 },
    { p: 0.95, value: scenario.p95 },
  ];

  if (target <= scenario.p5) return 0.05 * (target / scenario.p5);
  if (target >= scenario.p95) return 0.95 + 0.05 * ((target - scenario.p95) / (scenario.p95 * 0.1 || 1));

  for (let i = 0; i < percentiles.length - 1; i++) {
    const lower = percentiles[i];
    const upper = percentiles[i + 1];
    if (target >= lower.value && target <= upper.value) {
      const fraction = (target - lower.value) / (upper.value - lower.value || 1);
      return lower.p + fraction * (upper.p - lower.p);
    }
  }

  return 0.5;
};

/**
 * Compute a rough significance indicator between two scenarios.
 * Uses overlap of IQR ranges as a proxy.
 */
const computeOverlapSignificance = (
  a: ScenarioData,
  b: ScenarioData
): { significant: boolean; overlapPercent: number } => {
  const overlapStart = Math.max(a.p25, b.p25);
  const overlapEnd = Math.min(a.p75, b.p75);
  const overlap = Math.max(0, overlapEnd - overlapStart);

  const rangeA = a.p75 - a.p25;
  const rangeB = b.p75 - b.p25;
  const avgRange = (rangeA + rangeB) / 2 || 1;
  const overlapPercent = (overlap / avgRange) * 100;

  return {
    significant: overlapPercent < 25,
    overlapPercent,
  };
};

// ==============================================================================
// Component
// ==============================================================================

const ScenarioComparisonChart: React.FC<ScenarioComparisonChartProps> = ({
  scenarios,
  target,
  unit = 'tCO2e',
  title = 'Scenario Comparison',
}) => {
  // Prepare box plot data
  const boxPlotData = useMemo((): BoxPlotData[] => {
    return scenarios.map((scenario, idx) => ({
      name: scenario.name,
      boxBottom: scenario.p25,
      boxHeight: scenario.p75 - scenario.p25,
      whiskerLow: scenario.p5,
      whiskerHigh: scenario.p95,
      median: scenario.p50,
      mean: scenario.mean,
      p5: scenario.p5,
      p25: scenario.p25,
      p75: scenario.p75,
      p95: scenario.p95,
      color: scenario.color || DEFAULT_COLORS[idx % DEFAULT_COLORS.length],
      probabilityBelowTarget: target != null ? estimateProbabilityBelowTarget(scenario, target) : null,
    }));
  }, [scenarios, target]);

  // Pairwise significance comparisons
  const significanceResults = useMemo(() => {
    const results: Array<{ a: string; b: string; significant: boolean; overlapPercent: number }> = [];
    for (let i = 0; i < scenarios.length - 1; i++) {
      const { significant, overlapPercent } = computeOverlapSignificance(scenarios[i], scenarios[i + 1]);
      results.push({
        a: scenarios[i].name,
        b: scenarios[i + 1].name,
        significant,
        overlapPercent,
      });
    }
    return results;
  }, [scenarios]);

  // Y-axis domain
  const yDomain = useMemo(() => {
    const allValues = scenarios.flatMap((s) => [s.p5, s.p95]);
    if (target != null) allValues.push(target);
    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const padding = (max - min) * 0.1;
    return [Math.max(0, min - padding), max + padding];
  }, [scenarios, target]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    const data = payload[0]?.payload as BoxPlotData;
    if (!data) return null;

    return (
      <Paper sx={{ p: 1.5 }} elevation={3}>
        <Typography variant="body2" fontWeight="bold" gutterBottom>
          {data.name}
        </Typography>
        <Typography variant="body2">Mean: {formatNumber(data.mean, 2)} {unit}</Typography>
        <Typography variant="body2">Median (P50): {formatNumber(data.median, 2)} {unit}</Typography>
        <Typography variant="body2" color="textSecondary">
          P5: {formatNumber(data.p5, 2)} | P25: {formatNumber(data.p25, 2)}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          P75: {formatNumber(data.p75, 2)} | P95: {formatNumber(data.p95, 2)}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          IQR: {formatNumber(data.p75 - data.p25, 2)} {unit}
        </Typography>
        {data.probabilityBelowTarget != null && (
          <Typography variant="body2" sx={{ mt: 0.5 }} color="primary">
            P(below target): {(data.probabilityBelowTarget * 100).toFixed(1)}%
          </Typography>
        )}
      </Paper>
    );
  };

  // Custom box shape renderer
  const renderBoxPlot = (props: any) => {
    const { x, y, width, height, payload } = props;
    if (!payload) return null;

    const data = payload as BoxPlotData;
    const boxX = x;
    const boxWidth = width;
    const chartHeight = 400;

    // Calculate Y positions using the chart scale
    const yScale = (value: number) => {
      const [domainMin, domainMax] = yDomain;
      const range = domainMax - domainMin;
      return chartHeight - 30 - ((value - domainMin) / range) * (chartHeight - 60);
    };

    const whiskerLowY = yScale(data.whiskerLow);
    const whiskerHighY = yScale(data.whiskerHigh);
    const medianY = yScale(data.median);
    const meanY = yScale(data.mean);

    return (
      <g>
        {/* Whisker line (vertical) */}
        <line
          x1={boxX + boxWidth / 2}
          y1={whiskerHighY}
          x2={boxX + boxWidth / 2}
          y2={y}
          stroke={data.color}
          strokeWidth={1.5}
          strokeDasharray="4 2"
        />
        <line
          x1={boxX + boxWidth / 2}
          y1={y + height}
          x2={boxX + boxWidth / 2}
          y2={whiskerLowY}
          stroke={data.color}
          strokeWidth={1.5}
          strokeDasharray="4 2"
        />

        {/* Whisker caps */}
        <line x1={boxX + boxWidth * 0.25} y1={whiskerHighY} x2={boxX + boxWidth * 0.75} y2={whiskerHighY} stroke={data.color} strokeWidth={2} />
        <line x1={boxX + boxWidth * 0.25} y1={whiskerLowY} x2={boxX + boxWidth * 0.75} y2={whiskerLowY} stroke={data.color} strokeWidth={2} />

        {/* Median line */}
        <line x1={boxX} y1={medianY} x2={boxX + boxWidth} y2={medianY} stroke={data.color} strokeWidth={3} />

        {/* Mean diamond */}
        <polygon
          points={`${boxX + boxWidth / 2},${meanY - 5} ${boxX + boxWidth / 2 + 5},${meanY} ${boxX + boxWidth / 2},${meanY + 5} ${boxX + boxWidth / 2 - 5},${meanY}`}
          fill={data.color}
          stroke="#fff"
          strokeWidth={1}
        />
      </g>
    );
  };

  if (scenarios.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="textSecondary">
          No scenario data available for comparison.
        </Typography>
      </Paper>
    );
  }

  if (scenarios.length < 2) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Alert severity="info">
          At least 2 scenarios are needed for comparison. Currently only 1 scenario is available.
        </Alert>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
        <Box>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
            {scenarios.map((s, idx) => (
              <Chip
                key={s.name}
                label={`${s.name}: ${formatNumber(s.mean, 2)} ${unit}`}
                size="small"
                sx={{
                  backgroundColor: `${s.color || DEFAULT_COLORS[idx % DEFAULT_COLORS.length]}20`,
                  borderColor: s.color || DEFAULT_COLORS[idx % DEFAULT_COLORS.length],
                }}
                variant="outlined"
              />
            ))}
            {target != null && (
              <Chip
                label={`Target: ${formatNumber(target, 2)} ${unit}`}
                size="small"
                color="error"
                variant="outlined"
              />
            )}
          </Stack>
        </Box>
      </Box>

      {/* Box Plot Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart
          data={boxPlotData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 12 }}
          />
          <YAxis
            domain={yDomain}
            label={{ value: unit, angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* IQR Box (P25 to P75) */}
          <Bar
            dataKey="boxHeight"
            stackId="box"
            shape={renderBoxPlot}
          >
            {boxPlotData.map((entry, index) => (
              <Cell
                key={`box-${index}`}
                fill={entry.color}
                fillOpacity={0.3}
                stroke={entry.color}
                strokeWidth={2}
              />
            ))}
          </Bar>

          {/* Invisible base to position the box correctly */}
          <Bar
            dataKey="boxBottom"
            stackId="box"
            fill="transparent"
            stroke="none"
          />

          {/* Target reference line */}
          {target != null && (
            <ReferenceLine
              y={target}
              stroke="#f44336"
              strokeDasharray="8 4"
              strokeWidth={2}
              label={{
                value: `Target: ${formatNumber(target, 2)}`,
                position: 'right',
                fill: '#f44336',
                fontSize: 12,
                fontWeight: 'bold',
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Significance indicators */}
      {significanceResults.length > 0 && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Statistical Significance (IQR Overlap)
          </Typography>
          <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
            {significanceResults.map((result, idx) => (
              <Chip
                key={idx}
                icon={<CompareArrows />}
                label={`${result.a} vs ${result.b}: ${result.significant ? 'Significant' : 'Not significant'} (${result.overlapPercent.toFixed(0)}% overlap)`}
                size="small"
                color={result.significant ? 'success' : 'warning'}
                variant="outlined"
              />
            ))}
          </Stack>
        </Box>
      )}

      {/* Probability of meeting target */}
      {target != null && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Probability of Meeting Target ({formatNumber(target, 2)} {unit})
          </Typography>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Scenario</TableCell>
                  <TableCell align="right">Mean ({unit})</TableCell>
                  <TableCell align="right">Gap to Target</TableCell>
                  <TableCell align="right">P(Below Target)</TableCell>
                  <TableCell align="center">Assessment</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {boxPlotData.map((scenario) => {
                  const gap = scenario.mean - target;
                  const prob = scenario.probabilityBelowTarget || 0;

                  return (
                    <TableRow key={scenario.name}>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              backgroundColor: scenario.color,
                            }}
                          />
                          {scenario.name}
                        </Box>
                      </TableCell>
                      <TableCell align="right">{formatNumber(scenario.mean, 2)}</TableCell>
                      <TableCell align="right">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                          {gap > 0 ? (
                            <TrendingUp fontSize="small" color="error" />
                          ) : (
                            <TrendingDown fontSize="small" color="success" />
                          )}
                          {formatNumber(Math.abs(gap), 2)} {gap > 0 ? 'above' : 'below'}
                        </Box>
                      </TableCell>
                      <TableCell align="right">{(prob * 100).toFixed(1)}%</TableCell>
                      <TableCell align="center">
                        <Chip
                          label={prob >= 0.8 ? 'Likely' : prob >= 0.5 ? 'Possible' : 'Unlikely'}
                          size="small"
                          color={prob >= 0.8 ? 'success' : prob >= 0.5 ? 'warning' : 'error'}
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Paper>
  );
};

export default ScenarioComparisonChart;
