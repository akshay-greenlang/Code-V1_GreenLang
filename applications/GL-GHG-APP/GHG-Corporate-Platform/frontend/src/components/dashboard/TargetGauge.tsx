/**
 * TargetGauge - Semi-circle gauge showing target progress
 *
 * Renders a semi-circular arc from 0% to 100% with the filled
 * portion representing current progress toward an emissions
 * reduction target. Includes milestone markers at 25%, 50%, 75%,
 * target year and reduction labels, and an on-track indicator.
 */

import React from 'react';
import { Box, Typography, Card, CardContent, Chip } from '@mui/material';
import { CheckCircle, Warning } from '@mui/icons-material';
import type { Target } from '../../types';

interface TargetGaugeProps {
  target: Target;
  currentProgress: number;
}

const GAUGE_SIZE = 200;
const STROKE_WIDTH = 16;
const RADIUS = (GAUGE_SIZE - STROKE_WIDTH) / 2;
const CENTER = GAUGE_SIZE / 2;
const CIRCUMFERENCE = Math.PI * RADIUS;

const getPointOnArc = (percent: number): { x: number; y: number } => {
  const angle = Math.PI * (1 - percent);
  return {
    x: CENTER + RADIUS * Math.cos(angle),
    y: CENTER - RADIUS * Math.sin(angle),
  };
};

const TargetGauge: React.FC<TargetGaugeProps> = ({ target, currentProgress }) => {
  const clampedProgress = Math.min(Math.max(currentProgress, 0), 100);
  const fillLength = (clampedProgress / 100) * CIRCUMFERENCE;
  const onTrack = target.status === 'on_track' || target.status === 'achieved';

  const progressColor = onTrack ? '#2e7d32' : '#c62828';
  const milestones = [25, 50, 75];

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
          <Typography variant="h6">{target.name}</Typography>
          <Chip
            icon={onTrack ? <CheckCircle fontSize="small" /> : <Warning fontSize="small" />}
            label={onTrack ? 'On Track' : 'Behind'}
            size="small"
            color={onTrack ? 'success' : 'error'}
            variant="outlined"
          />
        </Box>

        {/* SVG Gauge */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
          <svg
            width={GAUGE_SIZE}
            height={GAUGE_SIZE / 2 + 20}
            viewBox={`0 ${CENTER - RADIUS - STROKE_WIDTH} ${GAUGE_SIZE} ${RADIUS + STROKE_WIDTH + 20}`}
          >
            {/* Background arc */}
            <path
              d={`M ${STROKE_WIDTH / 2} ${CENTER} A ${RADIUS} ${RADIUS} 0 0 1 ${GAUGE_SIZE - STROKE_WIDTH / 2} ${CENTER}`}
              fill="none"
              stroke="#e0e0e0"
              strokeWidth={STROKE_WIDTH}
              strokeLinecap="round"
            />

            {/* Progress arc */}
            <path
              d={`M ${STROKE_WIDTH / 2} ${CENTER} A ${RADIUS} ${RADIUS} 0 0 1 ${GAUGE_SIZE - STROKE_WIDTH / 2} ${CENTER}`}
              fill="none"
              stroke={progressColor}
              strokeWidth={STROKE_WIDTH}
              strokeLinecap="round"
              strokeDasharray={`${fillLength} ${CIRCUMFERENCE}`}
            />

            {/* Milestone markers */}
            {milestones.map((m) => {
              const point = getPointOnArc(m / 100);
              return (
                <g key={m}>
                  <circle cx={point.x} cy={point.y} r={3} fill="#757575" />
                  <text
                    x={point.x}
                    y={point.y - 12}
                    textAnchor="middle"
                    fontSize={10}
                    fill="#757575"
                  >
                    {m}%
                  </text>
                </g>
              );
            })}

            {/* Center label */}
            <text
              x={CENTER}
              y={CENTER - 12}
              textAnchor="middle"
              fontSize={28}
              fontWeight={700}
              fill={progressColor}
            >
              {clampedProgress.toFixed(0)}%
            </text>
            <text
              x={CENTER}
              y={CENTER + 8}
              textAnchor="middle"
              fontSize={11}
              fill="#4a4a68"
            >
              progress
            </text>
          </svg>
        </Box>

        {/* Target details */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            mt: 1,
            pt: 1.5,
            borderTop: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Box>
            <Typography variant="caption" color="text.secondary">
              Target Year
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {target.target_year}
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="caption" color="text.secondary">
              Reduction
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {target.target_reduction_percent.toFixed(1)}%
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="caption" color="text.secondary">
              Annual Rate
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {target.annual_reduction_rate.toFixed(1)}%/yr
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default TargetGauge;
