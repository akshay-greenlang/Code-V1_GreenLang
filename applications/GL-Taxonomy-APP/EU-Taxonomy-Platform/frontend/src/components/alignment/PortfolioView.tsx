/**
 * PortfolioView - Portfolio-level alignment summary.
 */

import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip, Divider } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';

const PortfolioView: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Portfolio Alignment</Typography>
      <Grid container spacing={3}>
        <Grid item xs={3}>
          <ScoreGauge value={45.3} label="Alignment Rate" size={100} color="#1B5E20" />
        </Grid>
        <Grid item xs={3}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" sx={{ fontWeight: 700, color: '#1B5E20' }}>24</Typography>
            <Typography variant="body2" color="text.secondary">Aligned</Typography>
          </Box>
        </Grid>
        <Grid item xs={3}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" sx={{ fontWeight: 700, color: '#0277BD' }}>38</Typography>
            <Typography variant="body2" color="text.secondary">Eligible</Typography>
          </Box>
        </Grid>
        <Grid item xs={3}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h3" sx={{ fontWeight: 700, color: '#757575' }}>53</Typography>
            <Typography variant="body2" color="text.secondary">Total</Typography>
          </Box>
        </Grid>
      </Grid>
      <Divider sx={{ my: 2 }} />
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip label="CCM: 18 aligned" size="small" sx={{ backgroundColor: '#C8E6C9' }} />
        <Chip label="CCA: 4 aligned" size="small" sx={{ backgroundColor: '#BBDEFB' }} />
        <Chip label="WTR: 1 aligned" size="small" sx={{ backgroundColor: '#B3E5FC' }} />
        <Chip label="CE: 1 aligned" size="small" sx={{ backgroundColor: '#FFE0B2' }} />
      </Box>
    </CardContent>
  </Card>
);

export default PortfolioView;
