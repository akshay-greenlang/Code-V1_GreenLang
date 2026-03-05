/**
 * ReadinessScore - Circular gauge showing SBTi readiness 0-100%.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemText, Chip } from '@mui/material';
import ScoreGauge from '../common/ScoreGauge';

interface ReadinessScoreProps {
  score: number;
  categoryScores?: { category: string; score: number }[];
}

const ReadinessScore: React.FC<ReadinessScoreProps> = ({ score, categoryScores = [] }) => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>SBTi Readiness</Typography>
        <ScoreGauge value={score} label="Overall Readiness" subtitle="Target validation score" />
        {categoryScores.length > 0 && (
          <List dense sx={{ width: '100%', mt: 2 }}>
            {categoryScores.map((cat) => (
              <ListItem key={cat.category} disableGutters sx={{ py: 0.5 }}>
                <ListItemText
                  primary={cat.category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                  primaryTypographyProps={{ fontSize: '0.8rem' }}
                />
                <Chip
                  label={`${cat.score}%`}
                  size="small"
                  sx={{
                    backgroundColor: cat.score >= 80 ? '#E8F5E9' : cat.score >= 50 ? '#FFF3E0' : '#FFEBEE',
                    color: cat.score >= 80 ? '#1B5E20' : cat.score >= 50 ? '#E65100' : '#B71C1C',
                    fontWeight: 600,
                    fontSize: '0.75rem',
                  }}
                />
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default ReadinessScore;
