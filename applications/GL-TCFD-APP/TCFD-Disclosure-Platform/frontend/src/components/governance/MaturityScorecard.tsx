import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from 'recharts';
import type { GovernanceAssessment } from '../../types';

interface MaturityScorecardProps {
  assessment: GovernanceAssessment | null;
}

const MaturityScorecard: React.FC<MaturityScorecardProps> = ({ assessment }) => {
  if (!assessment) {
    return (
      <Card><CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Governance Maturity</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          No assessment data available. Run a governance assessment to see results.
        </Typography>
      </CardContent></Card>
    );
  }

  const dimensions = [
    { dimension: 'Board Oversight', value: assessment.board_oversight_score },
    { dimension: 'Management Role', value: assessment.management_role_score },
    { dimension: 'Climate Competency', value: assessment.climate_competency_score },
    { dimension: 'Reporting Frequency', value: assessment.reporting_frequency_score },
    { dimension: 'Integration', value: assessment.integration_score },
    { dimension: 'Incentive Alignment', value: assessment.incentive_alignment_score },
    { dimension: 'Training', value: assessment.training_score },
    { dimension: 'Stakeholder Engagement', value: assessment.stakeholder_engagement_score },
  ];

  const maturityColor: Record<string, string> = {
    initial: '#B71C1C', developing: '#E65100', defined: '#F57F17',
    managed: '#2E7D32', optimizing: '#1B5E20',
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>Governance Maturity Scorecard</Typography>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="h4" sx={{ fontWeight: 700, color: 'primary.main' }}>{assessment.overall_score.toFixed(0)}%</Typography>
            <Chip label={assessment.overall_maturity.replace(/_/g, ' ')} sx={{ bgcolor: maturityColor[assessment.overall_maturity], color: 'white', textTransform: 'capitalize' }} size="small" />
          </Box>
        </Box>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={dimensions}>
            <PolarGrid />
            <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 11 }} />
            <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Radar name="Score" dataKey="value" stroke="#1B5E20" fill="#1B5E20" fillOpacity={0.25} />
            <Tooltip formatter={(v: number) => [`${v}%`, 'Score']} />
          </RadarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

export default MaturityScorecard;
