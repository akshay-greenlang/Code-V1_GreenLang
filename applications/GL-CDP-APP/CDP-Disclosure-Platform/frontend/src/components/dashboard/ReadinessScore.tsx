/**
 * ReadinessScore - Overall readiness percentage
 *
 * Shows the overall submission readiness with answered, reviewed,
 * and approved question counts.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, CircularProgress } from '@mui/material';
import { CheckCircle } from '@mui/icons-material';

interface ReadinessScoreProps {
  readinessPct: number;
  answeredQuestions: number;
  totalQuestions: number;
  reviewedQuestions: number;
  approvedQuestions: number;
}

const ReadinessScore: React.FC<ReadinessScoreProps> = ({
  readinessPct,
  answeredQuestions,
  totalQuestions,
  reviewedQuestions,
  approvedQuestions,
}) => {
  const color = readinessPct >= 80 ? '#2e7d32' : readinessPct >= 50 ? '#ef6c00' : '#c62828';

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Submission Readiness
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, mb: 2 }}>
          <Box sx={{ position: 'relative', display: 'inline-flex' }}>
            <CircularProgress
              variant="determinate"
              value={readinessPct}
              size={80}
              thickness={4}
              sx={{ color }}
            />
            <Box
              sx={{
                position: 'absolute',
                top: 0, left: 0, bottom: 0, right: 0,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
            >
              <Typography variant="h6" fontWeight={700} sx={{ color }}>
                {readinessPct.toFixed(0)}%
              </Typography>
            </Box>
          </Box>
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
              <CheckCircle fontSize="small" sx={{ color: '#2e7d32' }} />
              <Typography variant="body2">
                {answeredQuestions}/{totalQuestions} answered
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
              <CheckCircle fontSize="small" sx={{ color: '#1565c0' }} />
              <Typography variant="body2">
                {reviewedQuestions} reviewed
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <CheckCircle fontSize="small" sx={{ color: '#1b5e20' }} />
              <Typography variant="body2">
                {approvedQuestions} approved
              </Typography>
            </Box>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ReadinessScore;
