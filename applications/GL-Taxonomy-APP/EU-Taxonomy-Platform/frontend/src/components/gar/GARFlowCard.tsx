/**
 * GARFlowCard - Green Asset Ratio (flow/new originations).
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Grid } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import { currencyFormat } from '../../utils/formatters';

const GARFlowCard: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>GAR (Flow)</Typography>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={4}>
          <ScoreGauge value={35.2} label="GAR Flow" size={100} color="#0D47A1" />
        </Grid>
        <Grid item xs={8}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">New originations</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>{currencyFormat(850000000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="primary.main">Taxonomy-aligned</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#0D47A1' }}>{currencyFormat(299200000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">Eligible (not aligned)</Typography>
              <Typography variant="body2">{currencyFormat(170000000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Non-eligible</Typography>
              <Typography variant="body2">{currencyFormat(380800000)}</Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default GARFlowCard;
