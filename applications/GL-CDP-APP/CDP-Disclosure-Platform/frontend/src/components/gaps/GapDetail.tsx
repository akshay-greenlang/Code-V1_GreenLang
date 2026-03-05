/**
 * GapDetail - Gap detail with recommendations
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Divider, Alert } from '@mui/material';
import { Lightbulb, TrendingUp } from '@mui/icons-material';
import type { GapItem } from '../../types';
import { getSeverityColor } from '../../utils/formatters';
import { SCORING_CATEGORY_NAMES, ScoringCategory } from '../../types';

interface GapDetailProps { gap: GapItem; }

const GapDetail: React.FC<GapDetailProps> = ({ gap }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Chip label={gap.severity} color={getSeverityColor(gap.severity)} size="small" />
        <Chip label={`Effort: ${gap.effort}`} variant="outlined" size="small" />
        <Typography variant="h6">{gap.question_number}</Typography>
      </Box>
      <Typography variant="body2" sx={{ mb: 1 }}>{gap.gap_description}</Typography>
      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <Box>
          <Typography variant="caption" color="text.secondary">Current Level</Typography>
          <Typography variant="body2" fontWeight={600} color="error.main">{gap.current_level}</Typography>
        </Box>
        <Box>
          <Typography variant="caption" color="text.secondary">Target Level</Typography>
          <Typography variant="body2" fontWeight={600} color="success.main">{gap.target_level}</Typography>
        </Box>
        <Box>
          <Typography variant="caption" color="text.secondary">Category</Typography>
          <Typography variant="body2">{SCORING_CATEGORY_NAMES[gap.category as ScoringCategory] || gap.category}</Typography>
        </Box>
      </Box>
      <Divider sx={{ my: 1.5 }} />
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Lightbulb sx={{ color: '#ef6c00' }} />
        <Typography variant="subtitle2" fontWeight={600}>Recommendation</Typography>
      </Box>
      <Typography variant="body2" sx={{ mb: 1.5 }}>{gap.recommendation}</Typography>
      {gap.example_response && (
        <Alert severity="info" sx={{ mb: 1.5 }}>
          <Typography variant="caption"><strong>Example:</strong> {gap.example_response}</Typography>
        </Alert>
      )}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <TrendingUp fontSize="small" sx={{ color: '#2e7d32' }} />
        <Typography variant="body2" color="success.main" fontWeight={600}>
          Potential uplift: +{gap.uplift_points.toFixed(1)} points
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

export default GapDetail;
