/**
 * CategoryOverview - 6 ISO categories as cards
 *
 * Renders a grid of cards for each ISO 14064-1 category showing
 * emissions, removals, net tCO2e, significance badge, and
 * data quality tier indicator.
 */

import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import type { CategoryResult } from '../../types';
import {
  ISOCategory,
  ISO_CATEGORY_SHORT_NAMES,
  CATEGORY_COLORS,
} from '../../types';
import { formatTCO2e, getStatusColor, getDataQualityColor, getDataQualityLabel } from '../../utils/formatters';

interface CategoryOverviewProps {
  categories: CategoryResult[];
  onCategoryClick?: (category: ISOCategory) => void;
}

const CategoryOverview: React.FC<CategoryOverviewProps> = ({
  categories,
  onCategoryClick,
}) => {
  const allCategories = Object.values(ISOCategory);
  const totalEmissions = categories.reduce((sum, c) => sum + c.total_tco2e, 0);

  return (
    <Grid container spacing={2}>
      {allCategories.map((cat) => {
        const result = categories.find((c) => c.category === cat);
        const emissions = result?.total_tco2e ?? 0;
        const removals = result?.removals_tco2e ?? 0;
        const net = result?.net_tco2e ?? 0;
        const pct = totalEmissions > 0 ? (emissions / totalEmissions) * 100 : 0;
        const color = CATEGORY_COLORS[cat];

        return (
          <Grid item xs={12} sm={6} md={4} key={cat}>
            <Card
              sx={{
                cursor: onCategoryClick ? 'pointer' : 'default',
                borderTop: `3px solid ${color}`,
                '&:hover': onCategoryClick
                  ? { boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }
                  : {},
                transition: 'box-shadow 0.2s ease',
              }}
              onClick={() => onCategoryClick?.(cat)}
            >
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                  <Typography variant="subtitle2" color="text.secondary" noWrap sx={{ maxWidth: '65%' }}>
                    {ISO_CATEGORY_SHORT_NAMES[cat]}
                  </Typography>
                  {result && (
                    <Chip
                      label={result.significance.replace(/_/g, ' ')}
                      color={getStatusColor(result.significance)}
                      size="small"
                      sx={{ fontSize: '0.65rem' }}
                    />
                  )}
                </Box>

                <Typography variant="h5" fontWeight={700} sx={{ color }}>
                  {formatTCO2e(emissions)}
                </Typography>

                {removals > 0 && (
                  <Typography variant="body2" color="success.main">
                    Removals: -{formatTCO2e(removals)}
                  </Typography>
                )}

                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  Net: {formatTCO2e(net)}
                </Typography>

                <Box sx={{ mt: 1.5, mb: 0.5 }}>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(pct, 100)}
                    sx={{
                      height: 6,
                      borderRadius: 3,
                      backgroundColor: '#e0e0e0',
                      '& .MuiLinearProgress-bar': {
                        backgroundColor: color,
                        borderRadius: 3,
                      },
                    }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {pct.toFixed(1)}% of total
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    {result?.source_count ?? 0} sources
                  </Typography>
                  {result && (
                    <Chip
                      label={getDataQualityLabel(result.data_quality_tier)}
                      color={getDataQualityColor(result.data_quality_tier)}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: '0.6rem' }}
                    />
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
};

export default CategoryOverview;
