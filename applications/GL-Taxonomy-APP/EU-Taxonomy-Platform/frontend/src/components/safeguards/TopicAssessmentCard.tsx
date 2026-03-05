/**
 * TopicAssessmentCard - Card for each safeguard topic (HR, Anti-Corruption, Tax, Competition).
 */

import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress, Chip, Divider } from '@mui/material';
import { CheckCircle, Cancel } from '@mui/icons-material';

interface TopicAssessmentCardProps {
  topic?: string;
  proceduralScore?: number;
  outcomeScore?: number;
  pass?: boolean;
  frameworks?: string[];
  findings?: string[];
}

const TopicAssessmentCard: React.FC<TopicAssessmentCardProps> = ({
  topic = 'Human Rights',
  proceduralScore = 85,
  outcomeScore = 90,
  pass = true,
  frameworks = ['UN Guiding Principles', 'OECD Guidelines', 'ILO Core Conventions'],
  findings = [],
}) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>{topic}</Typography>
        <Chip
          icon={pass ? <CheckCircle /> : <Cancel />}
          label={pass ? 'PASS' : 'FAIL'}
          color={pass ? 'success' : 'error'}
          size="small"
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="body2">Procedural</Typography>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>{proceduralScore}%</Typography>
        </Box>
        <LinearProgress variant="determinate" value={proceduralScore} color="primary" sx={{ height: 6, borderRadius: 3 }} />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="body2">Outcome</Typography>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>{outcomeScore}%</Typography>
        </Box>
        <LinearProgress variant="determinate" value={outcomeScore} color="success" sx={{ height: 6, borderRadius: 3 }} />
      </Box>

      <Divider sx={{ my: 1.5 }} />

      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
        Referenced Frameworks
      </Typography>
      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
        {frameworks.map(f => <Chip key={f} label={f} size="small" variant="outlined" />)}
      </Box>

      {findings.length > 0 && (
        <Box sx={{ mt: 1.5 }}>
          <Typography variant="caption" color="error">
            {findings.length} adverse finding(s) recorded
          </Typography>
        </Box>
      )}
    </CardContent>
  </Card>
);

export default TopicAssessmentCard;
