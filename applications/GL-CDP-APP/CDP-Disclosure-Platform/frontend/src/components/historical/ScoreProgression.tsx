/**
 * ScoreProgression - Multi-year score trend chart
 *
 * Plots historical CDP scores across multiple years with
 * scoring band reference areas, level labels, and submission
 * status indicators.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
} from 'recharts';
import { Timeline } from '@mui/icons-material';
import type { HistoricalScore } from '../../types';
import { ScoringLevel, SCORING_LEVEL_COLORS } from '../../types';

interface ScoreProgressionProps {
  scores: HistoricalScore[];
}

const BAND_AREAS = [
  { y1: 0, y2: 30, fill: '#ffcdd2', label: 'Disclosure' },
  { y1: 30, y2: 50, fill: '#ffe0b2', label: 'Awareness' },
  { y1: 50, y2: 70, fill: '#bbdefb', label: 'Management' },
  { y1: 70, y2: 100, fill: '#c8e6c9', label: 'Leadership' },
];

const ScoreProgression: React.FC<ScoreProgressionProps> = ({ scores }) => {
  const sorted = [...scores].sort((a, b) => a.year - b.year);
  const latestScore = sorted.length > 0 ? sorted[sorted.length - 1] : null;
  const firstScore = sorted.length > 0 ? sorted[0] : null;
  const totalChange = latestScore && firstScore
    ? latestScore.score - firstScore.score
    : 0;

  const chartData = sorted.map((s) => ({
    year: s.year,
    score: s.score,
    level: s.level,
    submitted: s.submitted,
  }));

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline sx={{ color: '#1565c0' }} />
            <Typography variant="h6">Score Progression</Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {sorted.length > 0 && (
              <Chip
                label={`${sorted.length} years`}
                size="small"
                variant="outlined"
              />
            )}
            {totalChange !== 0 && (
              <Chip
                label={`${totalChange > 0 ? '+' : ''}${totalChange.toFixed(1)} pts total`}
                size="small"
                color={totalChange > 0 ? 'success' : 'error'}
              />
            )}
          </Box>
        </Box>

        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={chartData}>
            {/* Band reference areas */}
            {BAND_AREAS.map((band) => (
              <ReferenceArea
                key={band.label}
                y1={band.y1}
                y2={band.y2}
                fill={band.fill}
                fillOpacity={0.3}
              />
            ))}

            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e050" />
            <XAxis
              dataKey="year"
              tick={{ fontSize: 12 }}
              domain={['dataMin', 'dataMax']}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 11 }}
              ticks={[0, 20, 30, 40, 50, 60, 70, 80, 100]}
              label={{
                value: 'Score (%)',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 11 },
              }}
            />

            {/* Band boundary reference lines */}
            <ReferenceLine y={30} stroke="#ef6c00" strokeDasharray="3 3" strokeOpacity={0.5} />
            <ReferenceLine y={50} stroke="#1565c0" strokeDasharray="3 3" strokeOpacity={0.5} />
            <ReferenceLine y={70} stroke="#2e7d32" strokeDasharray="3 3" strokeOpacity={0.5} />

            <Tooltip
              formatter={(value: number, _name: string, entry: { payload: { level: ScoringLevel; submitted: boolean } }) => {
                const { level, submitted } = entry.payload;
                return [
                  `${value.toFixed(1)}% (${level}) ${submitted ? '[Submitted]' : '[Not Submitted]'}`,
                  'Score',
                ];
              }}
              labelFormatter={(label) => `Year: ${label}`}
            />

            <Line
              type="monotone"
              dataKey="score"
              stroke="#1b5e20"
              strokeWidth={3}
              dot={(props: { cx: number; cy: number; payload: { level: ScoringLevel } }) => {
                const { cx, cy, payload } = props;
                const color = SCORING_LEVEL_COLORS[payload.level] || '#1b5e20';
                return (
                  <circle
                    key={`dot-${cx}-${cy}`}
                    cx={cx}
                    cy={cy}
                    r={6}
                    fill={color}
                    stroke="#fff"
                    strokeWidth={2}
                  />
                );
              }}
              activeDot={{ r: 8 }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Year score labels */}
        <Box sx={{ display: 'flex', gap: 1, mt: 1, flexWrap: 'wrap', justifyContent: 'center' }}>
          {sorted.map((s) => (
            <Chip
              key={s.year}
              label={`${s.year}: ${s.level}`}
              size="small"
              sx={{
                bgcolor: SCORING_LEVEL_COLORS[s.level] + '20',
                color: SCORING_LEVEL_COLORS[s.level],
                border: `1px solid ${SCORING_LEVEL_COLORS[s.level]}40`,
                fontWeight: 600,
                fontSize: 11,
              }}
            />
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ScoreProgression;
