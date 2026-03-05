/**
 * AlignmentFunnel - Visual funnel from total activities to aligned.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

const DEMO_STAGES = [
  { label: 'Total Activities', count: 53, width: '100%', color: '#E0E0E0' },
  { label: 'Eligible', count: 38, width: '72%', color: '#BBDEFB' },
  { label: 'SC Passed', count: 32, width: '60%', color: '#81C784' },
  { label: 'DNSH Passed', count: 28, width: '53%', color: '#4CAF50' },
  { label: 'MS Passed', count: 26, width: '49%', color: '#388E3C' },
  { label: 'Aligned', count: 24, width: '45%', color: '#1B5E20' },
];

const AlignmentFunnel: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 3 }}>Alignment Funnel</Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
        {DEMO_STAGES.map((stage, idx) => (
          <Box
            key={idx}
            sx={{
              width: stage.width,
              backgroundColor: stage.color,
              borderRadius: 1,
              p: 1.5,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              transition: 'all 0.3s',
            }}
          >
            <Typography variant="body2" sx={{ fontWeight: 600, color: idx >= 3 ? '#FFF' : '#212121' }}>
              {stage.label}
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 700, color: idx >= 3 ? '#FFF' : '#212121' }}>
              {stage.count}
            </Typography>
          </Box>
        ))}
      </Box>
    </CardContent>
  </Card>
);

export default AlignmentFunnel;
