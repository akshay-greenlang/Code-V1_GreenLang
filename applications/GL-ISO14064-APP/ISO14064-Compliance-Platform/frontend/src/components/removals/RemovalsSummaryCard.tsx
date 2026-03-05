/**
 * RemovalsSummaryCard - Summary card for removal totals
 *
 * Displays total gross removals, permanence discount, and
 * total credited removals in a compact card format.
 */

import React, { useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  Divider,
} from '@mui/material';
import { Park, TrendingDown, CheckCircle } from '@mui/icons-material';
import type { RemovalSource } from '../../types';
import { formatNumber } from '../../utils/formatters';

interface RemovalsSummaryCardProps {
  removals: RemovalSource[];
}

const RemovalsSummaryCard: React.FC<RemovalsSummaryCardProps> = ({ removals }) => {
  const summary = useMemo(() => {
    const totalGross = removals.reduce((s, r) => s + r.gross_removals_tco2e, 0);
    const totalCredited = removals.reduce((s, r) => s + r.credited_removals_tco2e, 0);
    const discount = totalGross > 0 ? totalGross - totalCredited : 0;
    const discountPct = totalGross > 0 ? (discount / totalGross) * 100 : 0;
    return { totalGross, totalCredited, discount, discountPct };
  }, [removals]);

  return (
    <Card>
      <CardHeader
        title="Removals Summary"
        subheader={`${removals.length} removal sources`}
      />
      <CardContent>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={4}>
            <Box sx={{ textAlign: 'center' }}>
              <Park sx={{ color: '#2e7d32', fontSize: 32, mb: 0.5 }} />
              <Typography variant="caption" color="text.secondary" display="block">
                Total Gross Removals
              </Typography>
              <Typography variant="h5" fontWeight={700} color="success.main">
                {formatNumber(summary.totalGross, 2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                tCO2e
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Box sx={{ textAlign: 'center' }}>
              <TrendingDown sx={{ color: '#ef6c00', fontSize: 32, mb: 0.5 }} />
              <Typography variant="caption" color="text.secondary" display="block">
                Permanence Discount
              </Typography>
              <Typography variant="h5" fontWeight={700} color="warning.main">
                -{formatNumber(summary.discount, 2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                tCO2e ({summary.discountPct.toFixed(1)}%)
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Box sx={{ textAlign: 'center' }}>
              <CheckCircle sx={{ color: '#1b5e20', fontSize: 32, mb: 0.5 }} />
              <Typography variant="caption" color="text.secondary" display="block">
                Total Credited Removals
              </Typography>
              <Typography variant="h5" fontWeight={700} sx={{ color: '#1b5e20' }}>
                {formatNumber(summary.totalCredited, 2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                tCO2e
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default RemovalsSummaryCard;
