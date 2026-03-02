/**
 * UncertaintyMetricsCard - Enhanced stat card with uncertainty visualization
 *
 * Displays a primary metric value with its uncertainty range (value +/- range),
 * an inline sparkline showing the distribution shape, a data quality tier badge,
 * and a CV warning when variability exceeds 50%. Follows the existing StatCard
 * pattern but extends it with Monte Carlo uncertainty context.
 */

import React, { useMemo } from 'react';
import { Card, CardContent, Typography, Box, Chip, Stack } from '@mui/material';
import { SvgIconComponent } from '@mui/icons-material';
import { Warning } from '@mui/icons-material';
import { BarChart, Bar, ResponsiveContainer } from 'recharts';
import { formatNumber } from '../../utils/formatters';

// ==============================================================================
// Types
// ==============================================================================

interface UncertaintyMetricsCardProps {
  title: string;
  value: number;
  uncertainty: number;
  unit: string;
  tier?: 1 | 2 | 3;
  icon: SvgIconComponent;
  color?: 'primary' | 'secondary' | 'success' | 'error' | 'warning' | 'info';
  distributionSamples?: number[];
}

// ==============================================================================
// Constants
// ==============================================================================

const TIER_CONFIG: Record<number, { label: string; color: 'success' | 'warning' | 'error' }> = {
  1: { label: 'Tier 1', color: 'success' },
  2: { label: 'Tier 2', color: 'warning' },
  3: { label: 'Tier 3', color: 'error' },
};

// ==============================================================================
// Helpers
// ==============================================================================

/**
 * Build a mini histogram from sample data for the sparkline display.
 * Returns an array of { count } objects suitable for a tiny bar chart.
 */
const buildSparklineData = (samples: number[], binCount: number = 20): Array<{ count: number }> => {
  if (samples.length === 0) return [];

  const min = Math.min(...samples);
  const max = Math.max(...samples);
  const range = max - min || 1;
  const binWidth = range / binCount;

  const bins: number[] = new Array(binCount).fill(0);

  for (const val of samples) {
    let binIdx = Math.floor((val - min) / binWidth);
    if (binIdx >= binCount) binIdx = binCount - 1;
    bins[binIdx]++;
  }

  return bins.map((count) => ({ count }));
};

// ==============================================================================
// Component
// ==============================================================================

const UncertaintyMetricsCard: React.FC<UncertaintyMetricsCardProps> = ({
  title,
  value,
  uncertainty,
  unit,
  tier,
  icon: Icon,
  color = 'primary',
  distributionSamples,
}) => {
  const cv = useMemo(() => (value !== 0 ? (uncertainty / value) * 100 : 0), [value, uncertainty]);
  const highCV = cv > 50;
  const tierConfig = tier ? TIER_CONFIG[tier] : null;

  const sparklineData = useMemo(() => {
    if (!distributionSamples || distributionSamples.length === 0) return [];
    // Use a subset for performance
    const subset = distributionSamples.length > 500
      ? distributionSamples.filter((_, i) => i % Math.ceil(distributionSamples.length / 500) === 0)
      : distributionSamples;
    return buildSparklineData(subset);
  }, [distributionSamples]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography color="textSecondary" gutterBottom variant="body2" noWrap>
              {title}
            </Typography>
            <Typography variant="h5" component="div" noWrap>
              {formatNumber(value, 2)}
              <Typography
                component="span"
                variant="body1"
                color="textSecondary"
                sx={{ ml: 0.5 }}
              >
                +/- {formatNumber(uncertainty, 2)} {unit}
              </Typography>
            </Typography>

            {/* Badges row */}
            <Stack direction="row" spacing={0.5} sx={{ mt: 1 }} flexWrap="wrap" useFlexGap>
              {tierConfig && (
                <Chip
                  label={tierConfig.label}
                  size="small"
                  color={tierConfig.color}
                  variant="outlined"
                  sx={{ height: 20, fontSize: '0.7rem' }}
                />
              )}
              {highCV && (
                <Chip
                  icon={<Warning sx={{ fontSize: 14 }} />}
                  label={`CV: ${cv.toFixed(0)}%`}
                  size="small"
                  color="error"
                  variant="outlined"
                  sx={{ height: 20, fontSize: '0.7rem' }}
                />
              )}
              {!highCV && (
                <Chip
                  label={`CV: ${cv.toFixed(0)}%`}
                  size="small"
                  variant="outlined"
                  sx={{ height: 20, fontSize: '0.7rem' }}
                />
              )}
            </Stack>

            {/* Mini sparkline */}
            {sparklineData.length > 0 && (
              <Box sx={{ mt: 1, height: 30, width: '100%' }}>
                <ResponsiveContainer width="100%" height={30}>
                  <BarChart data={sparklineData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                    <Bar
                      dataKey="count"
                      fill={highCV ? '#f44336' : '#1976d2'}
                      fillOpacity={0.5}
                      radius={[1, 1, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            )}
          </Box>

          <Box
            sx={{
              backgroundColor: (theme) => `${theme.palette[color].main}20`,
              borderRadius: 2,
              p: 1.5,
              ml: 1,
              flexShrink: 0,
            }}
          >
            <Icon sx={{ color: `${color}.main`, fontSize: 32 }} />
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default UncertaintyMetricsCard;
