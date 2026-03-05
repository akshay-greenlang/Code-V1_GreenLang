/**
 * CapExKPICard - CapEx KPI with eligible/aligned gauges.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Divider } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import { currencyFormat } from '../../utils/formatters';

interface CapExKPICardProps {
  total?: number; eligible?: number; aligned?: number;
}

const CapExKPICard: React.FC<CapExKPICardProps> = ({
  total = 195000000, eligible = 140595000, aligned = 100035000,
}) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>CapEx KPI</Typography>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={6}>
          <Box sx={{ display: 'flex', justifyContent: 'space-around' }}>
            <ScoreGauge value={(eligible / total) * 100} label="Eligible" color="#0277BD" />
            <ScoreGauge value={(aligned / total) * 100} label="Aligned" color="#1B5E20" />
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Total CapEx</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>{currencyFormat(total)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Eligible CapEx</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#0277BD' }}>{currencyFormat(eligible)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Aligned CapEx</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#1B5E20' }}>{currencyFormat(aligned)}</Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default CapExKPICard;
