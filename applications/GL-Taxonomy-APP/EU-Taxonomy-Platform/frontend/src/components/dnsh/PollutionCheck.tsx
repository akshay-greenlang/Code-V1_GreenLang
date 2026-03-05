/**
 * PollutionCheck - DNSH assessment for Pollution Prevention & Control.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box } from '@mui/material';
import { CheckCircle, Cancel, AirOutlined } from '@mui/icons-material';

const DEMO_CRITERIA = [
  { criterion: 'IED/BREFs compliance (where applicable)', met: true, detail: 'Compliant with relevant Best Available Techniques Reference Documents' },
  { criterion: 'REACH regulation compliance', met: true, detail: 'No substances of very high concern (SVHC) used' },
  { criterion: 'Air quality standards compliance', met: true, detail: 'Emissions within national Air Quality Directive limits' },
  { criterion: 'RoHS compliance (where applicable)', met: true, detail: 'All electronic components RoHS compliant' },
];

const PollutionCheck: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <AirOutlined color="primary" />
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Pollution Prevention & Control (DNSH)</Typography>
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

export default PollutionCheck;
