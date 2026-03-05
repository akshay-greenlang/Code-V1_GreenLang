/**
 * BiodiversityCheck - DNSH assessment for Biodiversity & Ecosystems.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box } from '@mui/material';
import { CheckCircle, Cancel, Park } from '@mui/icons-material';

const DEMO_CRITERIA = [
  { criterion: 'Environmental Impact Assessment (EIA) completed', met: true, detail: 'EIA conducted per EU EIA Directive 2011/92/EU' },
  { criterion: 'No operations in Natura 2000 sites without mitigation', met: true, detail: 'No Natura 2000 overlap identified' },
  { criterion: 'Habitats Directive compliance', met: true, detail: 'Article 6(3) appropriate assessment completed' },
  { criterion: 'No significant harm to biodiversity-rich areas', met: true, detail: 'Biodiversity Management Plan implemented' },
  { criterion: 'Birds Directive compliance', met: true, detail: 'No protected species nesting sites affected' },
];

const BiodiversityCheck: React.FC = () => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <Park color="primary" />
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Biodiversity & Ecosystems (DNSH)</Typography>
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

export default BiodiversityCheck;
