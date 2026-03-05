/**
 * TargetStatusCards - Summary cards for each target type with status.
 */

import React from 'react';
import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import { TrackChanges, TrendingDown, AccountTree } from '@mui/icons-material';
import StatusBadge from '../common/StatusBadge';

interface TargetSummary {
  total_targets: number;
  validated: number;
  submitted: number;
  draft: number;
  near_term_count: number;
  long_term_count: number;
}

interface TargetStatusCardsProps {
  summary: TargetSummary;
}

const TargetStatusCards: React.FC<TargetStatusCardsProps> = ({ summary }) => {
  const cards = [
    { label: 'Total Targets', value: summary.total_targets, icon: <TrackChanges />, color: '#0D47A1' },
    { label: 'Near-Term', value: summary.near_term_count, icon: <TrendingDown />, color: '#1B5E20' },
    { label: 'Long-Term', value: summary.long_term_count, icon: <AccountTree />, color: '#4A148C' },
  ];

  return (
    <Grid container spacing={2}>
      {cards.map((card) => (
        <Grid item xs={12} sm={4} key={card.label}>
          <Card>
            <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box sx={{ p: 1, borderRadius: 1, backgroundColor: `${card.color}15` }}>
                {React.cloneElement(card.icon as React.ReactElement, { sx: { color: card.color, fontSize: 28 } })}
              </Box>
              <Box>
                <Typography variant="h4" sx={{ fontWeight: 700 }}>{card.value}</Typography>
                <Typography variant="body2" color="text.secondary">{card.label}</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>Validation Status</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <StatusBadge status="validated" variant="target" />
                <Typography variant="body2">{summary.validated}</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <StatusBadge status="submitted" variant="target" />
                <Typography variant="body2">{summary.submitted}</Typography>
              </Box>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <StatusBadge status="draft" variant="target" />
                <Typography variant="body2">{summary.draft}</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default TargetStatusCards;
