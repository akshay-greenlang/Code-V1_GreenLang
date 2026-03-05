/**
 * DAVersionTracker - Timeline of Delegated Act versions.
 */

import React from 'react';
import { Card, CardContent, Typography, Chip, Box, Timeline, TimelineItem, TimelineSeparator, TimelineConnector, TimelineContent, TimelineDot } from '@mui/material';

const DEMO_VERSIONS = [
  { version: 'Climate DA (June 2021)', date: '2021-06-04', status: 'in_force', objectives: 'CCM, CCA', key: 'Initial climate DA' },
  { version: 'Environmental DA (June 2023)', date: '2023-06-27', status: 'in_force', objectives: 'WTR, CE, PPC, BIO', key: 'Added 4 environmental objectives' },
  { version: 'Climate DA Amendment (2024)', date: '2024-02-15', status: 'in_force', objectives: 'CCM, CCA', key: 'Updated thresholds and added activities' },
  { version: 'Omnibus Simplification (2025)', date: '2025-10-01', status: 'draft', objectives: 'All 6', key: 'Simplified reporting for SMEs' },
];

const DAVersionTracker: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Delegated Act Timeline</Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {DEMO_VERSIONS.map((v, idx) => (
          <Box key={idx} sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', pl: 2, borderLeft: `3px solid ${v.status === 'in_force' ? '#1B5E20' : '#EF6C00'}` }}>
            <Box sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{v.version}</Typography>
                <Chip label={v.status === 'in_force' ? 'In Force' : 'Draft'} size="small" color={v.status === 'in_force' ? 'success' : 'warning'} />
              </Box>
              <Typography variant="caption" color="text.secondary">{v.date} | Objectives: {v.objectives}</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>{v.key}</Typography>
            </Box>
          </Box>
        ))}
      </Box>
    </CardContent>
  </Card>
);

export default DAVersionTracker;
