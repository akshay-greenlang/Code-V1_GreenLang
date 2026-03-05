/**
 * WaterAssessment - DNSH assessment for Water & Marine Resources objective.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Chip, Box } from '@mui/material';
import { CheckCircle, Cancel, Water } from '@mui/icons-material';

const DEMO_CRITERIA = [
  { criterion: 'Environmental degradation risks (WFD status)', met: true, detail: 'Good ecological status maintained' },
  { criterion: 'Water use efficiency measures in place', met: true, detail: 'ISO 14046 water footprint assessment completed' },
  { criterion: 'Water quality discharge standards compliance', met: true, detail: 'All discharge limits met per permit' },
  { criterion: 'No adverse impact on marine environment', met: true, detail: 'N/A - inland facility' },
  { criterion: 'RBMP consistency check', met: false, detail: 'River Basin Management Plan review pending' },
];

const WaterAssessment: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Water color="primary" />
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Water & Marine Resources (DNSH)
        </Typography>
      </Box>
      <List dense>
        {DEMO_CRITERIA.map((c, idx) => (
          <ListItem key={idx} sx={{ border: '1px solid #E0E0E0', borderRadius: 1, mb: 1 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              {c.met ? <CheckCircle sx={{ color: '#2E7D32' }} /> : <Cancel sx={{ color: '#C62828' }} />}
            </ListItemIcon>
            <ListItemText primary={c.criterion} secondary={c.detail} primaryTypographyProps={{ fontSize: '0.875rem', fontWeight: 500 }} />
          </ListItem>
        ))}
      </List>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
        <Chip label={`${DEMO_CRITERIA.filter(c => c.met).length}/${DEMO_CRITERIA.length} criteria met`} color={DEMO_CRITERIA.every(c => c.met) ? 'success' : 'warning'} />
      </Box>
    </CardContent>
  </Card>
);

export default WaterAssessment;
