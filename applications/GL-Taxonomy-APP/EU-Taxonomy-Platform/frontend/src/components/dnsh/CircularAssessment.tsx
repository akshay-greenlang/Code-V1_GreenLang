/**
 * CircularAssessment - DNSH assessment for Circular Economy objective.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box } from '@mui/material';
import { CheckCircle, Cancel, Recycling } from '@mui/icons-material';

const DEMO_CRITERIA = [
  { criterion: 'Waste management hierarchy compliance', met: true, detail: 'Prevention > Reuse > Recycling hierarchy followed' },
  { criterion: 'Durability and recyclability design', met: true, detail: 'Products designed for disassembly and material recovery' },
  { criterion: 'Waste generation minimization', met: true, detail: '12% reduction in waste generation vs baseline' },
  { criterion: 'No planned obsolescence', met: true, detail: 'Extended product lifetime program in place' },
];

const CircularAssessment: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Recycling color="primary" />
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Circular Economy (DNSH)</Typography>
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
    </CardContent>
  </Card>
);

export default CircularAssessment;
