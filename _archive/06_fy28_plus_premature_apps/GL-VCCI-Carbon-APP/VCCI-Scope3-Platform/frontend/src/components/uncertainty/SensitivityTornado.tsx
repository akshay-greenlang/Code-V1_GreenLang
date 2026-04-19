/**
 * SensitivityTornado - Tornado diagram for parameter sensitivity analysis
 *
 * Renders a horizontal bar chart showing the impact of each parameter on the
 * output variable. Bars extend left (negative impact, red) and right (positive
 * impact, green) from the baseline. Parameters are sorted by absolute impact
 * with highest at top. Displays Sobol first-order sensitivity indices as
 * labels and supports interactive filtering by top-N parameters.
 */

import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import {
  Paper,
  Typography,
  Box,
  Slider,
  Chip,
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';

// ==============================================================================
// Types
// ==============================================================================

interface SensitivityParameter {
  name: string;
  sobolIndex: number;
  lowValue: number;
  highValue: number;
  baseValue: number;
  category?: string;
}

interface SensitivityTornadoProps {
  parameters: SensitivityParameter[];
  baseline: number;
  title?: string;
}

interface TornadoBarData {
  name: string;
  negativeImpact: number;
  positiveImpact: number;
  sobolIndex: number;
  lowValue: number;
  highValue: number;
  baseValue: number;
  absoluteImpact: number;
  category: string;
}

// ==============================================================================
// Constants
// ==============================================================================

const CATEGORY_COLORS: Record<string, { positive: string; negative: string }> = {
  'Emission Factors': { positive: '#4caf50', negative: '#f44336' },
  'Activity Data': { positive: '#2196f3', negative: '#ff5722' },
  'Methodology': { positive: '#009688', negative: '#e91e63' },
  'Allocation': { positive: '#8bc34a', negative: '#ff9800' },
  'default': { positive: '#4caf50', negative: '#f44336' },
};

// ==============================================================================
// Component
// ==============================================================================

const SensitivityTornado: React.FC<SensitivityTornadoProps> = ({
  parameters,
  baseline,
  title = 'Sensitivity Analysis - Tornado Diagram',
}) => {
  const [topN, setTopN] = useState<number>(10);
  const [categoryFilter, setCategoryFilter] = useState<string>('all');

  // Extract unique categories
  const categories = useMemo(() => {
    const cats = new Set(parameters.map((p) => p.category || 'Uncategorized'));
    return ['all', ...Array.from(cats)];
  }, [parameters]);

  // Transform and sort data
  const tornadoData = useMemo((): TornadoBarData[] => {
    let filtered = parameters;

    if (categoryFilter !== 'all') {
      filtered = filtered.filter(
        (p) => (p.category || 'Uncategorized') === categoryFilter
      );
    }

    const transformed = filtered.map((param): TornadoBarData => {
      const lowImpact = param.lowValue - baseline;
      const highImpact = param.highValue - baseline;

      return {
        name: param.name,
        negativeImpact: Math.min(lowImpact, highImpact),
        positiveImpact: Math.max(lowImpact, highImpact),
        sobolIndex: param.sobolIndex,
        lowValue: param.lowValue,
        highValue: param.highValue,
        baseValue: param.baseValue,
        absoluteImpact: Math.abs(highImpact) + Math.abs(lowImpact),
        category: param.category || 'Uncategorized',
      };
    });

    // Sort by absolute impact (descending) and take top N
    return transformed
      .sort((a, b) => b.absoluteImpact - a.absoluteImpact)
      .slice(0, topN);
  }, [parameters, baseline, topN, categoryFilter]);

  // Compute axis domain
  const axisDomain = useMemo(() => {
    if (tornadoData.length === 0) return [-100, 100];
    const maxAbs = Math.max(
      ...tornadoData.map((d) => Math.max(Math.abs(d.negativeImpact), Math.abs(d.positiveImpact)))
    );
    const padding = maxAbs * 0.15;
    return [-(maxAbs + padding), maxAbs + padding];
  }, [tornadoData]);

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || payload.length === 0) return null;
    const data = payload[0]?.payload as TornadoBarData;
    if (!data) return null;

    const colors = CATEGORY_COLORS[data.category] || CATEGORY_COLORS['default'];

    return (
      <Paper sx={{ p: 1.5 }} elevation={3}>
        <Typography variant="body2" fontWeight="bold" gutterBottom>
          {data.name}
        </Typography>
        <Typography variant="body2" color="textSecondary">
          Category: {data.category}
        </Typography>
        <Typography variant="body2" sx={{ color: colors.negative }}>
          Low scenario: {data.lowValue.toFixed(2)} tCO2e
        </Typography>
        <Typography variant="body2" sx={{ color: colors.positive }}>
          High scenario: {data.highValue.toFixed(2)} tCO2e
        </Typography>
        <Typography variant="body2">
          Base value: {data.baseValue.toFixed(2)}
        </Typography>
        <Typography variant="body2" fontWeight="bold" sx={{ mt: 0.5 }}>
          Sobol Si: {data.sobolIndex.toFixed(4)}
        </Typography>
      </Paper>
    );
  };

  // Custom bar label showing Sobol index
  const renderSobolLabel = (props: any) => {
    const { x, y, width, height, value } = props;
    if (!value || Math.abs(value) < 0.001) return null;

    const data = tornadoData.find((d) => d.positiveImpact === value || d.negativeImpact === value);
    if (!data) return null;

    const labelX = value > 0 ? x + width + 5 : x - 5;
    const textAnchor = value > 0 ? 'start' : 'end';

    return (
      <text
        x={labelX}
        y={y + height / 2}
        fill="#666"
        textAnchor={textAnchor}
        dominantBaseline="middle"
        fontSize={11}
      >
        Si={data.sobolIndex.toFixed(3)}
      </text>
    );
  };

  if (parameters.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="textSecondary">
          No sensitivity data available. Run a sensitivity analysis to see parameter impacts.
        </Typography>
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
          <Stack direction="row" spacing={1}>
            <Chip
              label={`Baseline: ${baseline.toFixed(2)} tCO2e`}
              size="small"
              color="primary"
              variant="outlined"
            />
            <Chip
              label={`Showing top ${Math.min(topN, tornadoData.length)} of ${parameters.length} parameters`}
              size="small"
              variant="outlined"
            />
          </Stack>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={categoryFilter}
              label="Category"
              onChange={(e) => setCategoryFilter(e.target.value)}
            >
              {categories.map((cat) => (
                <MenuItem key={cat} value={cat}>
                  {cat === 'all' ? 'All Categories' : cat}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>
      </Box>

      {/* Top-N slider */}
      <Box sx={{ px: 2, mb: 2 }}>
        <Typography variant="body2" color="textSecondary" gutterBottom>
          Number of parameters: {topN}
        </Typography>
        <Slider
          value={topN}
          onChange={(_, val) => setTopN(val as number)}
          min={3}
          max={Math.min(25, parameters.length)}
          step={1}
          valueLabelDisplay="auto"
          sx={{ maxWidth: 300 }}
        />
      </Box>

      {/* Tornado Chart */}
      <ResponsiveContainer width="100%" height={Math.max(300, tornadoData.length * 40 + 80)}>
        <BarChart
          data={tornadoData}
          layout="vertical"
          margin={{ top: 10, right: 80, left: 160, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={false} />
          <XAxis
            type="number"
            domain={axisDomain}
            label={{ value: 'Impact on emissions (tCO2e)', position: 'insideBottom', offset: -10 }}
            tick={{ fontSize: 11 }}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={150}
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* Baseline reference line at zero */}
          <ReferenceLine x={0} stroke="#666" strokeWidth={2} />

          {/* Negative impact bars (left of baseline) */}
          <Bar dataKey="negativeImpact" name="Decrease" label={renderSobolLabel}>
            {tornadoData.map((entry, index) => {
              const colors = CATEGORY_COLORS[entry.category] || CATEGORY_COLORS['default'];
              return <Cell key={`neg-${index}`} fill={colors.negative} fillOpacity={0.8} />;
            })}
          </Bar>

          {/* Positive impact bars (right of baseline) */}
          <Bar dataKey="positiveImpact" name="Increase" label={renderSobolLabel}>
            {tornadoData.map((entry, index) => {
              const colors = CATEGORY_COLORS[entry.category] || CATEGORY_COLORS['default'];
              return <Cell key={`pos-${index}`} fill={colors.positive} fillOpacity={0.8} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend for categories */}
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 1, flexWrap: 'wrap' }}>
        {Object.entries(CATEGORY_COLORS)
          .filter(([key]) => key !== 'default')
          .map(([category, colors]) => (
            <Stack key={category} direction="row" spacing={0.5} alignItems="center">
              <Box sx={{ width: 12, height: 12, backgroundColor: colors.negative, borderRadius: 0.5 }} />
              <Box sx={{ width: 12, height: 12, backgroundColor: colors.positive, borderRadius: 0.5 }} />
              <Typography variant="caption">{category}</Typography>
            </Stack>
          ))}
      </Box>
    </Paper>
  );
};

export default SensitivityTornado;
