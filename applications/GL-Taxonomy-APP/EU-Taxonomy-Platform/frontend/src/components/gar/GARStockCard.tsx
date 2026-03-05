/**
 * GARStockCard - Green Asset Ratio (stock) with gauge and breakdown.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Grid, Divider } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import { currencyFormat } from '../../utils/formatters';

const GARStockCard: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>GAR (Stock)</Typography>
      <Grid container spacing={3} alignItems="center">
        <Grid item xs={4}>
          <ScoreGauge value={28.5} label="GAR Stock" size={100} color="#1B5E20" />
        </Grid>
        <Grid item xs={8}>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">Total covered assets</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>{currencyFormat(4200000000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="primary.main">Taxonomy-aligned</Typography>
              <Typography variant="body2" sx={{ fontWeight: 600, color: '#1B5E20' }}>{currencyFormat(1197000000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2">Eligible (not aligned)</Typography>
              <Typography variant="body2">{currencyFormat(840000000)}</Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="body2" color="text.secondary">Non-eligible</Typography>
              <Typography variant="body2">{currencyFormat(2163000000)}</Typography>
            </Box>
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default GARStockCard;
