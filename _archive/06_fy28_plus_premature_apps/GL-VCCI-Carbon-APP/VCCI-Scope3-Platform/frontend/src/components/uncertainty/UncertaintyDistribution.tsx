/**
 * UncertaintyDistribution - Monte Carlo distribution histogram with KDE overlay
 *
 * Renders a combined histogram (50 bins) and KDE smoothed density curve
 * from Monte Carlo simulation samples. Displays percentile reference lines
 * (p5, p25, p50, p75, p95) and mean/median indicators. Supports configurable
 * bin counts and CSV export of the raw distribution data.
 */

import React, { useState, useMemo, useCallback } from 'react';
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
  Legend,
} from 'recharts';
import {
  Paper,
  Typography,
  Box,
  Button,
  Slider,
  Chip,
  Stack,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import { Download, Settings as SettingsIcon } from '@mui/icons-material';

// ==============================================================================
// Types
// ==============================================================================

interface UncertaintyDistributionProps {
  data: number[];
  mean: number;
  median: number;
  p5: number;
  p50: number;
  p95: number;
  stdDev: number;
  title?: string;
  unit?: string;
}

interface BinData {
  binStart: number;
  binEnd: number;
  binLabel: string;
  count: number;
  density: number;
  kdeDensity: number;
}

// ==============================================================================
// Helpers
// ==============================================================================

/**
 * Compute a Gaussian KDE estimate at a given point x.
 * Uses Silverman's rule of thumb for bandwidth selection.
 */
const gaussianKDE = (data: number[], x: number, bandwidth: number): number => {
  const n = data.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const u = (x - data[i]) / bandwidth;
    sum += Math.exp(-0.5 * u * u);
  }
  return sum / (n * bandwidth * Math.sqrt(2 * Math.PI));
};

const computeBandwidth = (data: number[], stdDev: number): number => {
  const n = data.length;
  // Silverman's rule of thumb
  return 1.06 * stdDev * Math.pow(n, -0.2);
};

const buildHistogram = (
  data: number[],
  binCount: number,
  bandwidth: number
): BinData[] => {
  if (data.length === 0) return [];

  const sorted = [...data].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];
  const range = max - min || 1;
  const binWidth = range / binCount;

  const bins: BinData[] = [];

  for (let i = 0; i < binCount; i++) {
    const binStart = min + i * binWidth;
    const binEnd = min + (i + 1) * binWidth;
    const binMid = (binStart + binEnd) / 2;

    let count = 0;
    for (const val of data) {
      if (i === binCount - 1) {
        if (val >= binStart && val <= binEnd) count++;
      } else {
        if (val >= binStart && val < binEnd) count++;
      }
    }

    const density = count / (data.length * binWidth);
    const kdeDensity = gaussianKDE(data, binMid, bandwidth);

    bins.push({
      binStart,
      binEnd,
      binLabel: binMid.toFixed(1),
      count,
      density,
      kdeDensity,
    });
  }

  return bins;
};

const formatValue = (value: number, unit?: string): string => {
  const formatted = value >= 1000
    ? `${(value / 1000).toFixed(2)}k`
    : value.toFixed(2);
  return unit ? `${formatted} ${unit}` : formatted;
};

// ==============================================================================
// Component
// ==============================================================================

const UncertaintyDistribution: React.FC<UncertaintyDistributionProps> = ({
  data,
  mean,
  median,
  p5,
  p50,
  p95,
  stdDev,
  title = 'Monte Carlo Distribution',
  unit = 'tCO2e',
}) => {
  const [binCount, setBinCount] = useState<number>(50);
  const [showPercentiles, setShowPercentiles] = useState(true);
  const [settingsAnchor, setSettingsAnchor] = useState<null | HTMLElement>(null);

  // Compute histogram bins and KDE curve
  const histogramData = useMemo(() => {
    if (data.length === 0) return [];
    const bw = computeBandwidth(data, stdDev);
    return buildHistogram(data, binCount, bw);
  }, [data, binCount, stdDev]);

  // CSV export handler
  const handleExportCSV = useCallback(() => {
    if (data.length === 0) return;

    const csvHeader = 'sample_index,value\n';
    const csvRows = data.map((val, idx) => `${idx + 1},${val}`).join('\n');
    const csvContent = csvHeader + csvRows;

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'monte_carlo_distribution.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [data]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    return (
      <Paper sx={{ p: 1.5 }} elevation={3}>
        <Typography variant="body2" fontWeight="bold">
          Bin: {label} {unit}
        </Typography>
        {payload.map((entry: any, index: number) => (
          <Typography key={index} variant="body2" sx={{ color: entry.color }}>
            {entry.name}: {entry.value.toFixed(4)}
          </Typography>
        ))}
      </Paper>
    );
  };

  if (data.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="textSecondary">
          No distribution data available. Run a Monte Carlo simulation to generate results.
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper sx={{ p: 2 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h6" gutterBottom>
            {title}
          </Typography>
          <Stack direction="row" spacing={1}>
            <Chip
              label={`Mean: ${formatValue(mean, unit)}`}
              size="small"
              color="primary"
              variant="outlined"
            />
            <Chip
              label={`Median: ${formatValue(median, unit)}`}
              size="small"
              color="secondary"
              variant="outlined"
            />
            <Chip
              label={`Std Dev: ${formatValue(stdDev, unit)}`}
              size="small"
              variant="outlined"
            />
            <Chip
              label={`n = ${data.length.toLocaleString()}`}
              size="small"
              variant="outlined"
            />
          </Stack>
        </Box>
        <Box>
          <IconButton
            size="small"
            onClick={(e) => setSettingsAnchor(e.currentTarget)}
            aria-label="Chart settings"
          >
            <SettingsIcon />
          </IconButton>
          <Button
            variant="outlined"
            size="small"
            startIcon={<Download />}
            onClick={handleExportCSV}
            sx={{ ml: 1 }}
          >
            Export CSV
          </Button>
        </Box>
      </Box>

      {/* Settings Menu */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
      >
        <MenuItem disableRipple sx={{ flexDirection: 'column', alignItems: 'flex-start', width: 250 }}>
          <Typography variant="body2" gutterBottom>
            Bin Count: {binCount}
          </Typography>
          <Slider
            value={binCount}
            onChange={(_, val) => setBinCount(val as number)}
            min={10}
            max={100}
            step={5}
            valueLabelDisplay="auto"
            sx={{ width: '100%' }}
          />
        </MenuItem>
        <MenuItem
          onClick={() => {
            setShowPercentiles(!showPercentiles);
            setSettingsAnchor(null);
          }}
        >
          {showPercentiles ? 'Hide' : 'Show'} Percentile Lines
        </MenuItem>
      </Menu>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart
          data={histogramData}
          margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="binLabel"
            label={{ value: unit, position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 11 }}
            interval={Math.floor(binCount / 10)}
          />
          <YAxis
            yAxisId="density"
            label={{ value: 'Density', angle: -90, position: 'insideLeft' }}
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Histogram bars */}
          <Bar
            yAxisId="density"
            dataKey="density"
            fill="#1976d2"
            fillOpacity={0.6}
            name="Histogram Density"
          />

          {/* KDE curve overlay */}
          <Line
            yAxisId="density"
            type="monotone"
            dataKey="kdeDensity"
            stroke="#f44336"
            strokeWidth={2}
            dot={false}
            name="KDE Curve"
          />

          {/* Percentile reference lines */}
          {showPercentiles && (
            <>
              <ReferenceLine
                yAxisId="density"
                x={histogramData.reduce((closest, bin) =>
                  Math.abs(parseFloat(bin.binLabel) - p5) < Math.abs(parseFloat(closest.binLabel) - p5) ? bin : closest
                , histogramData[0])?.binLabel}
                stroke="#ff9800"
                strokeDasharray="5 5"
                label={{ value: 'P5', position: 'top', fill: '#ff9800', fontSize: 11 }}
              />
              <ReferenceLine
                yAxisId="density"
                x={histogramData.reduce((closest, bin) =>
                  Math.abs(parseFloat(bin.binLabel) - p50) < Math.abs(parseFloat(closest.binLabel) - p50) ? bin : closest
                , histogramData[0])?.binLabel}
                stroke="#4caf50"
                strokeDasharray="5 5"
                label={{ value: 'P50', position: 'top', fill: '#4caf50', fontSize: 11 }}
              />
              <ReferenceLine
                yAxisId="density"
                x={histogramData.reduce((closest, bin) =>
                  Math.abs(parseFloat(bin.binLabel) - p95) < Math.abs(parseFloat(closest.binLabel) - p95) ? bin : closest
                , histogramData[0])?.binLabel}
                stroke="#f44336"
                strokeDasharray="5 5"
                label={{ value: 'P95', position: 'top', fill: '#f44336', fontSize: 11 }}
              />
            </>
          )}

          {/* Mean indicator */}
          <ReferenceLine
            yAxisId="density"
            x={histogramData.reduce((closest, bin) =>
              Math.abs(parseFloat(bin.binLabel) - mean) < Math.abs(parseFloat(closest.binLabel) - mean) ? bin : closest
            , histogramData[0])?.binLabel}
            stroke="#1976d2"
            strokeWidth={2}
            label={{ value: 'Mean', position: 'top', fill: '#1976d2', fontSize: 11, fontWeight: 'bold' }}
          />

          {/* Median indicator */}
          <ReferenceLine
            yAxisId="density"
            x={histogramData.reduce((closest, bin) =>
              Math.abs(parseFloat(bin.binLabel) - median) < Math.abs(parseFloat(closest.binLabel) - median) ? bin : closest
            , histogramData[0])?.binLabel}
            stroke="#9c27b0"
            strokeWidth={2}
            strokeDasharray="3 3"
            label={{ value: 'Median', position: 'top', fill: '#9c27b0', fontSize: 11, fontWeight: 'bold' }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Percentile summary bar */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1, flexWrap: 'wrap' }}>
        <Chip label={`P5: ${formatValue(p5, unit)}`} size="small" sx={{ backgroundColor: '#ff980030' }} />
        <Chip label={`P25: ${formatValue((p5 + p50) / 2, unit)}`} size="small" variant="outlined" />
        <Chip label={`P50: ${formatValue(p50, unit)}`} size="small" sx={{ backgroundColor: '#4caf5030' }} />
        <Chip label={`P75: ${formatValue((p50 + p95) / 2, unit)}`} size="small" variant="outlined" />
        <Chip label={`P95: ${formatValue(p95, unit)}`} size="small" sx={{ backgroundColor: '#f4433630' }} />
      </Box>
    </Paper>
  );
};

export default UncertaintyDistribution;
