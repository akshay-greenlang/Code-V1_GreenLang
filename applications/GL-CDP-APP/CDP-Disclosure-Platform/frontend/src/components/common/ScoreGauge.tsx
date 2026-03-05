/**
 * ScoreGauge - CDP score gauge visualization (D- to A)
 *
 * Renders a semicircular gauge with color-coded bands representing
 * the 8 CDP scoring levels. A needle indicates the current score.
 */

import React from 'react';
import { Box, Typography } from '@mui/material';
import { GAUGE_SEGMENTS } from '../../utils/scoringHelpers';
import type { ScoringLevel } from '../../types';

interface ScoreGaugeProps {
  score: number;
  level: ScoringLevel;
  size?: number;
  showLabel?: boolean;
}

const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score,
  level,
  size = 200,
  showLabel = true,
}) => {
  const currentSegment = GAUGE_SEGMENTS.find(
    (s) => score >= s.min && score < s.max,
  ) || GAUGE_SEGMENTS[GAUGE_SEGMENTS.length - 1];

  const normalizedScore = Math.min(100, Math.max(0, score));
  const angle = (normalizedScore / 100) * 180;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ position: 'relative', width: size, height: size / 2 + 20 }}>
        {/* Gauge background segments */}
        <svg
          width={size}
          height={size / 2 + 10}
          viewBox={`0 0 ${size} ${size / 2 + 10}`}
        >
          {GAUGE_SEGMENTS.map((segment, idx) => {
            const startAngle = (segment.min / 100) * 180;
            const endAngle = (segment.max / 100) * 180;
            const cx = size / 2;
            const cy = size / 2;
            const r = size / 2 - 10;
            const startRad = ((180 - startAngle) * Math.PI) / 180;
            const endRad = ((180 - endAngle) * Math.PI) / 180;
            const x1 = cx + r * Math.cos(startRad);
            const y1 = cy - r * Math.sin(startRad);
            const x2 = cx + r * Math.cos(endRad);
            const y2 = cy - r * Math.sin(endRad);
            const largeArc = endAngle - startAngle > 90 ? 1 : 0;

            return (
              <path
                key={idx}
                d={`M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 0 ${x2} ${y2}`}
                fill="none"
                stroke={segment.color}
                strokeWidth={16}
                strokeLinecap="butt"
                opacity={segment.level === level ? 1 : 0.3}
              />
            );
          })}

          {/* Needle */}
          {(() => {
            const cx = size / 2;
            const cy = size / 2;
            const needleLen = size / 2 - 30;
            const rad = ((180 - angle) * Math.PI) / 180;
            const nx = cx + needleLen * Math.cos(rad);
            const ny = cy - needleLen * Math.sin(rad);
            return (
              <>
                <line
                  x1={cx}
                  y1={cy}
                  x2={nx}
                  y2={ny}
                  stroke="#1a1a2e"
                  strokeWidth={2.5}
                  strokeLinecap="round"
                />
                <circle cx={cx} cy={cy} r={5} fill="#1a1a2e" />
              </>
            );
          })()}
        </svg>

        {/* Score text overlay */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            textAlign: 'center',
          }}
        >
          <Typography
            variant="h4"
            fontWeight={800}
            sx={{ color: currentSegment.color }}
          >
            {level}
          </Typography>
        </Box>
      </Box>

      {showLabel && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          {normalizedScore.toFixed(0)}% -- {currentSegment.label} Band
        </Typography>
      )}
    </Box>
  );
};

export default ScoreGauge;
