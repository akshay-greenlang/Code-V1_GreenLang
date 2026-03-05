/**
 * FLAGTrigger - 20% trigger assessment display.
 */
import React from 'react';
import { Card, CardContent, Typography, Alert, Box, Chip } from '@mui/material';
import type { FLAGTriggerResult } from '../../types';

interface FLAGTriggerProps { trigger: FLAGTriggerResult; }

const FLAGTrigger: React.FC<FLAGTriggerProps> = ({ trigger }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>FLAG Trigger Assessment</Typography>
      <Alert severity={trigger.flag_exceeds_20_pct ? 'warning' : 'info'} sx={{ mb: 2 }}>
        FLAG emissions are <strong>{trigger.flag_pct_of_total.toFixed(1)}%</strong> of total.
        {trigger.separate_flag_target_required ? ' A separate FLAG target IS required (exceeds 20%).' : ' No separate FLAG target required.'}
      </Alert>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
        <Chip label={`FLAG: ${trigger.flag_pct_of_total.toFixed(1)}%`} color={trigger.flag_exceeds_20_pct ? 'warning' : 'success'} />
        <Chip label={`Threshold: 20%`} variant="outlined" />
        {trigger.separate_flag_target_required && <Chip label={`Target Year: ${trigger.recommended_flag_target_year}`} color="primary" />}
        {trigger.deforestation_free_required && <Chip label="Zero deforestation required" color="error" />}
      </Box>
    </CardContent>
  </Card>
);

export default FLAGTrigger;
