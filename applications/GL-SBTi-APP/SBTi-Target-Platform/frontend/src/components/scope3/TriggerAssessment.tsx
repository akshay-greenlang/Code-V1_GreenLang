/**
 * TriggerAssessment - 40% trigger result display.
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Alert, Chip } from '@mui/material';
import type { TriggerAssessment as TriggerAssessmentType } from '../../types';

interface TriggerAssessmentProps { trigger: TriggerAssessmentType; }

const TriggerAssessment: React.FC<TriggerAssessmentProps> = ({ trigger }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Scope 3 Trigger Assessment</Typography>
      <Alert severity={trigger.scope_3_exceeds_40_pct ? 'warning' : 'info'} sx={{ mb: 2 }}>
        Scope 3 is <strong>{trigger.scope_3_pct_of_total.toFixed(1)}%</strong> of total emissions.
        {trigger.scope_3_exceeds_40_pct ? ' Scope 3 target IS required (exceeds 40% threshold).' : ' Scope 3 target is NOT required (below 40% threshold).'}
      </Alert>
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
        <Chip label={`Threshold: 40%`} variant="outlined" />
        <Chip label={`Actual: ${trigger.scope_3_pct_of_total.toFixed(1)}%`} color={trigger.scope_3_exceeds_40_pct ? 'warning' : 'success'} />
        <Chip label={`Coverage: ${trigger.current_coverage_pct.toFixed(0)}%`} color={trigger.coverage_sufficient ? 'success' : 'error'} />
        <Chip label={trigger.two_thirds_coverage_met ? '2/3 coverage met' : '2/3 coverage NOT met'} color={trigger.two_thirds_coverage_met ? 'success' : 'error'} />
      </Box>
      {trigger.categories_over_threshold.length > 0 && (
        <Typography variant="body2" color="text.secondary">
          Significant categories: {trigger.categories_over_threshold.join(', ')}
        </Typography>
      )}
    </CardContent>
  </Card>
);

export default TriggerAssessment;
