/**
 * SCResultCard - Summary card showing SC assessment result.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Divider, Grid } from '@mui/material';
import { CheckCircle, Cancel } from '@mui/icons-material';
import ScoreGauge from '../common/ScoreGauge';

interface SCResultCardProps {
  activityName?: string;
  objective?: string;
  passes?: boolean;
  score?: number;
  activityType?: string;
  criteriaCount?: number;
  passedCount?: number;
}

const SCResultCard: React.FC<SCResultCardProps> = ({
  activityName = 'Solar PV Electricity Generation',
  objective = 'Climate Change Mitigation',
  passes = true,
  score = 87.5,
  activityType = 'Own Performance',
  criteriaCount = 5,
  passedCount = 4,
}) => (
  <Card sx={{ border: `2px solid ${passes ? '#2E7D32' : '#C62828'}` }}>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>{activityName}</Typography>
          <Typography variant="body2" color="text.secondary">{objective}</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {passes ? (
            <Chip icon={<CheckCircle />} label="SC PASS" color="success" />
          ) : (
            <Chip icon={<Cancel />} label="SC FAIL" color="error" />
          )}
        </Box>
      </Box>

      <Divider sx={{ my: 2 }} />

      <Grid container spacing={3} alignItems="center">
        <Grid item xs={4}>
          <ScoreGauge value={score} label="Overall Score" size={90} color={passes ? '#2E7D32' : '#C62828'} />
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" sx={{ fontWeight: 700, color: passes ? '#2E7D32' : '#C62828' }}>
              {passedCount}/{criteriaCount}
            </Typography>
            <Typography variant="body2" color="text.secondary">Criteria Passed</Typography>
          </Box>
        </Grid>
        <Grid item xs={4}>
          <Box sx={{ textAlign: 'center' }}>
            <Chip label={activityType} color="primary" variant="outlined" />
            <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 1 }}>
              Activity Classification
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </CardContent>
  </Card>
);

export default SCResultCard;
