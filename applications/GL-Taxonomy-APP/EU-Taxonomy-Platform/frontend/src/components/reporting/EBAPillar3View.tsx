/**
 * EBAPillar3View - EBA Pillar 3 disclosure view.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Grid } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';

const EBAPillar3View: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>EBA Pillar 3 Summary</Typography>
        <Chip label="CRR Article 449a" size="small" variant="outlined" />
      </Box>
      <Grid container spacing={3}>
        <Grid item xs={3}>
          <ScoreGauge value={28.5} label="GAR Stock" size={90} color="#1B5E20" />
        </Grid>
        <Grid item xs={3}>
          <ScoreGauge value={35.2} label="GAR Flow" size={90} color="#0D47A1" />
        </Grid>
        <Grid item xs={3}>
          <ScoreGauge value={18.7} label="BTAR" size={90} color="#4A148C" />
        </Grid>
        <Grid item xs={3}>
          <Box sx={{ textAlign: 'center', pt: 1 }}>
            <Typography variant="h4" sx={{ fontWeight: 700 }}>8</Typography>
            <Typography variant="body2" color="text.secondary">Templates Generated</Typography>
            <Chip label="Ready for Submission" size="small" color="success" sx={{ mt: 1 }} />
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default EBAPillar3View;
