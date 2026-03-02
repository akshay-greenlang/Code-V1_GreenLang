/**
 * MaterialityMatrix - Scatter plot of Scope 3 categories
 *
 * Plots categories on X-axis (emissions magnitude) vs. Y-axis
 * (data quality score) with bubble size proportional to % of total.
 * Four quadrants: Focus, Monitor, De-prioritize, Investigate.
 */

import React, { useMemo } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
  ReferenceLine,
  ReferenceArea,
  Label,
} from 'recharts';
import { Box, Typography, Card, CardContent } from '@mui/material';
import type { MaterialityResult } from '../../types';
import { SCOPE3_CATEGORY_NAMES } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface MaterialityMatrixProps {
  materiality: MaterialityResult[];
}

interface MatrixPoint {
  x: number;
  y: number;
  z: number;
  name: string;
  category: string;
  isMaterial: boolean;
}

const QUADRANT_COLORS = {
  focus: 'rgba(229, 57, 53, 0.06)',
  monitor: 'rgba(46, 125, 50, 0.06)',
  deprioritize: 'rgba(158, 158, 158, 0.06)',
  investigate: 'rgba(239, 108, 0, 0.06)',
};

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: MatrixPoint }>;
}) => {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <Box
      sx={{
        bgcolor: 'background.paper',
        p: 1.5,
        borderRadius: 1,
        boxShadow: 2,
        border: '1px solid',
        borderColor: 'divider',
        maxWidth: 220,
      }}
    >
      <Typography variant="subtitle2">{point.name}</Typography>
      <Typography variant="body2">
        Emissions: {formatNumber(point.x)} tCO2e
      </Typography>
      <Typography variant="body2">
        Quality: {point.y.toFixed(0)}%
      </Typography>
      <Typography variant="body2">
        Share: {point.z.toFixed(1)}%
      </Typography>
    </Box>
  );
};

const MaterialityMatrix: React.FC<MaterialityMatrixProps> = ({ materiality }) => {
  const { points, maxX, midX, midY } = useMemo(() => {
    const pts: MatrixPoint[] = materiality.map((m) => ({
      x: m.materiality_score * 100,
      y: m.data_availability === 'high' ? 85 : m.data_availability === 'medium' ? 55 : 25,
      z: m.materiality_score * 20,
      name: SCOPE3_CATEGORY_NAMES[m.category] || m.category,
      category: m.category,
      isMaterial: m.is_material,
    }));
    const mx = Math.max(...pts.map((p) => p.x), 100);
    return { points: pts, maxX: mx * 1.1, midX: mx * 0.5, midY: 50 };
  }, [materiality]);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Materiality Matrix
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Categories plotted by emissions magnitude (X) vs. data quality (Y).
          Bubble size indicates share of total Scope 3.
        </Typography>

        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, left: 10, bottom: 20 }}>
            {/* Quadrant backgrounds */}
            <ReferenceArea x1={midX} x2={maxX} y1={0} y2={midY} fill={QUADRANT_COLORS.focus} />
            <ReferenceArea x1={midX} x2={maxX} y1={midY} y2={100} fill={QUADRANT_COLORS.monitor} />
            <ReferenceArea x1={0} x2={midX} y1={midY} y2={100} fill={QUADRANT_COLORS.deprioritize} />
            <ReferenceArea x1={0} x2={midX} y1={0} y2={midY} fill={QUADRANT_COLORS.investigate} />

            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis
              type="number"
              dataKey="x"
              name="Emissions"
              tick={{ fontSize: 12 }}
              label={{ value: 'Emissions Magnitude', position: 'bottom', style: { fontSize: 12 } }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Quality"
              domain={[0, 100]}
              tick={{ fontSize: 12 }}
              label={{ value: 'Data Quality (%)', angle: -90, position: 'insideLeft', style: { fontSize: 12 } }}
            />
            <ZAxis type="number" dataKey="z" range={[60, 400]} />

            {/* Quadrant dividers */}
            <ReferenceLine x={midX} stroke="#757575" strokeDasharray="4 4">
              <Label value="" position="top" />
            </ReferenceLine>
            <ReferenceLine y={midY} stroke="#757575" strokeDasharray="4 4" />

            <Tooltip content={<CustomTooltip />} />

            <Scatter
              data={points.filter((p) => p.isMaterial)}
              fill="#2e7d32"
              name="Material"
              opacity={0.8}
            />
            <Scatter
              data={points.filter((p) => !p.isMaterial)}
              fill="#9e9e9e"
              name="Immaterial"
              opacity={0.6}
            />
          </ScatterChart>
        </ResponsiveContainer>

        {/* Legend for quadrants */}
        <Box sx={{ display: 'flex', gap: 3, mt: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: '#c62828' }}>
            Bottom-Right: FOCUS (high emissions, low quality)
          </Typography>
          <Typography variant="caption" sx={{ color: '#2e7d32' }}>
            Top-Right: MONITOR (high emissions, high quality)
          </Typography>
          <Typography variant="caption" sx={{ color: '#757575' }}>
            Top-Left: DE-PRIORITIZE (low emissions, high quality)
          </Typography>
          <Typography variant="caption" sx={{ color: '#ef6c00' }}>
            Bottom-Left: INVESTIGATE (low emissions, low quality)
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MaterialityMatrix;
