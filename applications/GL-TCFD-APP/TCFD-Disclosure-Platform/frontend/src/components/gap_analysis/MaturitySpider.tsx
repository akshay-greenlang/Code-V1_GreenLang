/**
 * MaturitySpider - Radar/spider chart showing TCFD maturity across all four pillars.
 *
 * Displays current vs. target maturity scores across 10+ assessment dimensions,
 * grouped by TCFD pillar. Supports drill-down via dimension click.
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, Typography, Box, Chip, ToggleButton, ToggleButtonGroup } from '@mui/material';
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';

interface MaturityDimension {
  dimension: string;
  pillar: string;
  current: number;
  target: number;
  benchmark?: number;
}

interface MaturitySpiderProps {
  data: MaturityDimension[];
  pillarFilter?: string;
  showBenchmark?: boolean;
  onDimensionClick?: (dimension: string) => void;
}

const PILLAR_COLORS: Record<string, string> = {
  Governance: '#1B5E20',
  Strategy: '#0D47A1',
  'Risk Management': '#E65100',
  'Metrics & Targets': '#6A1B9A',
};

const MaturitySpider: React.FC<MaturitySpiderProps> = ({
  data,
  pillarFilter: initialPillarFilter,
  showBenchmark = false,
  onDimensionClick,
}) => {
  const [activePillar, setActivePillar] = useState<string>(initialPillarFilter || 'all');

  const filteredData = useMemo(() => {
    if (activePillar === 'all') return data;
    return data.filter((d) => d.pillar === activePillar);
  }, [data, activePillar]);

  const overallCurrent = useMemo(
    () => Math.round(data.reduce((sum, d) => sum + d.current, 0) / data.length),
    [data]
  );
  const overallTarget = useMemo(
    () => Math.round(data.reduce((sum, d) => sum + d.target, 0) / data.length),
    [data]
  );
  const overallGap = overallTarget - overallCurrent;

  const pillars = useMemo(() => {
    const unique = new Set(data.map((d) => d.pillar));
    return Array.from(unique);
  }, [data]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              TCFD Maturity Spider Chart
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
              <Chip
                label={`Current: ${overallCurrent}%`}
                size="small"
                sx={{ backgroundColor: '#E8F5E9', color: '#1B5E20', fontWeight: 600 }}
              />
              <Chip
                label={`Target: ${overallTarget}%`}
                size="small"
                sx={{ backgroundColor: '#FFEBEE', color: '#C62828', fontWeight: 600 }}
              />
              <Chip
                label={`Gap: ${overallGap}pts`}
                size="small"
                variant="outlined"
                color={overallGap > 20 ? 'error' : overallGap > 10 ? 'warning' : 'success'}
              />
            </Box>
          </Box>

          <ToggleButtonGroup
            value={activePillar}
            exclusive
            onChange={(_, val) => val && setActivePillar(val)}
            size="small"
          >
            <ToggleButton value="all">All</ToggleButton>
            {pillars.map((pillar) => (
              <ToggleButton key={pillar} value={pillar} sx={{ textTransform: 'none', fontSize: '0.75rem' }}>
                {pillar.split(' ')[0]}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        <ResponsiveContainer width="100%" height={400}>
          <RadarChart data={filteredData} cx="50%" cy="50%" outerRadius="70%">
            <PolarGrid gridType="polygon" />
            <PolarAngleAxis
              dataKey="dimension"
              fontSize={10}
              tick={{ fill: '#616161' }}
              onClick={(e) => onDimensionClick?.(e?.value as string)}
              style={{ cursor: onDimensionClick ? 'pointer' : 'default' }}
            />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9 }} />
            <Radar
              name="Current"
              dataKey="current"
              stroke="#1B5E20"
              fill="#1B5E20"
              fillOpacity={0.3}
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Radar
              name="Target"
              dataKey="target"
              stroke="#C62828"
              fill="#C62828"
              fillOpacity={0.1}
              strokeWidth={2}
              strokeDasharray="5 5"
            />
            {showBenchmark && (
              <Radar
                name="Industry Benchmark"
                dataKey="benchmark"
                stroke="#F57F17"
                fill="none"
                strokeWidth={1.5}
                strokeDasharray="3 3"
              />
            )}
            <Legend />
            <Tooltip
              formatter={(value: number, name: string) => [`${value}%`, name]}
              contentStyle={{ fontSize: '0.85rem' }}
            />
          </RadarChart>
        </ResponsiveContainer>

        {/* Pillar Summary Row */}
        <Box sx={{ display: 'flex', gap: 2, mt: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          {pillars.map((pillar) => {
            const pillarDims = data.filter((d) => d.pillar === pillar);
            const avg = Math.round(pillarDims.reduce((s, d) => s + d.current, 0) / pillarDims.length);
            const tgt = Math.round(pillarDims.reduce((s, d) => s + d.target, 0) / pillarDims.length);
            return (
              <Box
                key={pillar}
                sx={{
                  textAlign: 'center',
                  px: 2,
                  py: 1,
                  borderRadius: 1,
                  backgroundColor: activePillar === pillar ? `${PILLAR_COLORS[pillar]}10` : 'transparent',
                  border: '1px solid',
                  borderColor: activePillar === pillar ? PILLAR_COLORS[pillar] : '#E0E0E0',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                }}
                onClick={() => setActivePillar(activePillar === pillar ? 'all' : pillar)}
              >
                <Typography variant="caption" sx={{ fontWeight: 600, color: PILLAR_COLORS[pillar] }}>
                  {pillar}
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                  {avg}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Target: {tgt}%
                </Typography>
              </Box>
            );
          })}
        </Box>
      </CardContent>
    </Card>
  );
};

export default MaturitySpider;
