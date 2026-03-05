/**
 * ScoreGaugeDetail - Detailed score with breakdown
 */
import React from 'react';
import { Box, Typography, Card, CardContent, Chip } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';
import type { ScoringResult } from '../../types';
import { getBandLabel } from '../../utils/scoringHelpers';

interface ScoreGaugeDetailProps { result: ScoringResult; }

const ScoreGaugeDetail: React.FC<ScoreGaugeDetailProps> = ({ result }) => (
  <Card>
    <CardContent sx={{ textAlign: 'center' }}>
      <ScoreGauge score={result.overall_score} level={result.scoring_level} size={240} />
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 2 }}>
        <Chip label={`Band: ${getBandLabel(result.scoring_level)}`} color="primary" variant="outlined" />
        <Chip label={`Confidence: ${result.confidence.toFixed(0)}%`} variant="outlined" />
        {result.a_level_eligible && <Chip label="A-List Eligible" color="success" />}
      </Box>
      {result.previous_score != null && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Previous: {result.previous_level} ({result.previous_score.toFixed(0)}%)
        </Typography>
      )}
    </CardContent>
  </Card>
);

export default ScoreGaugeDetail;
