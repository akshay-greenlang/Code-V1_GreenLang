/**
 * TurnoverKPICard - Detailed turnover KPI with gauge and breakdown.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Divider, Chip } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import { currencyFormat } from '../../utils/formatters';

interface TurnoverKPICardProps {
  total?: number;
  eligible?: number;
  aligned?: number;
  enabling?: number;
  transitional?: number;
}

const TurnoverKPICard: React.FC<TurnoverKPICardProps> = ({
  total = 682000000,
  eligible = 465240000,
  aligned = 289850000,
  enabling = 45000000,
  transitional = 28000000,
}) => {
  const eligiblePct = (eligible / total) * 100;
  const alignedPct = (aligned / total) * 100;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Turnover KPI</Typography>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', justifyContent: 'space-around' }}>
              <ScoreGauge value={eligiblePct} label="Eligible" color="#0277BD" />
              <ScoreGauge value={alignedPct} label="Aligned" color="#1B5E20" />
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">Total</Typography>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>{currencyFormat(total)}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">Eligible</Typography>
                <Typography variant="body2" sx={{ fontWeight: 600, color: '#0277BD' }}>{currencyFormat(eligible)}</Typography>
              </Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="body2" color="text.secondary">Aligned</Typography>
                <Typography variant="body2" sx={{ fontWeight: 600, color: '#1B5E20' }}>{currencyFormat(aligned)}</Typography>
              </Box>
              <Divider />
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip label={`Enabling: ${currencyFormat(enabling)}`} size="small" variant="outlined" />
                <Chip label={`Transitional: ${currencyFormat(transitional)}`} size="small" variant="outlined" />
              </Box>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default TurnoverKPICard;
