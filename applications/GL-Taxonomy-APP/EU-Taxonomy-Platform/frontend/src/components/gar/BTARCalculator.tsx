/**
 * BTARCalculator - Banking Book Taxonomy Alignment Ratio calculator.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Divider, Chip } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import { currencyFormat } from '../../utils/formatters';

const BTARCalculator: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>BTAR (Trading Book)</Typography>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={4}>
          <ScoreGauge value={18.7} label="BTAR" size={100} color="#4A148C" />
        </Grid>
        <Grid item xs={8}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">Total trading book</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>{currencyFormat(1200000000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">Taxonomy-aligned</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#4A148C' }}>{currencyFormat(224400000)}</Typography>
            </Box>
            <Divider />
            <Typography variant="caption" color="text.secondary">By Instrument Type</Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip label="Equity: 22.1%" size="small" variant="outlined" />
              <Chip label="Bonds: 15.3%" size="small" variant="outlined" />
              <Chip label="Derivatives: 8.2%" size="small" variant="outlined" />
            </Box>
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default BTARCalculator;
