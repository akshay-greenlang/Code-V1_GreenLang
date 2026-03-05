/**
 * OmnibusTimeline - Omnibus simplification impact and timeline.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Alert, List, ListItem, ListItemText, ListItemIcon } from '@mui/material';
import { TrendingUp, TrendingDown, InfoOutlined } from '@mui/icons-material';

const DEMO_IMPACTS = [
  { area: 'SME Reporting', change: 'Simplified taxonomy disclosure for SMEs under CSRD', direction: 'positive', date: '2026-01-01' },
  { area: 'De Minimis Threshold', change: 'Increased de minimis threshold from 0% to 10% for non-financial companies', direction: 'positive', date: '2025-10-01' },
  { area: 'Voluntary Adoption', change: 'New voluntary simplified format for companies below EUR 450M turnover', direction: 'positive', date: '2026-01-01' },
  { area: 'Financial Institutions', change: 'Extended phase-in for GAR reporting of SME exposures', direction: 'positive', date: '2025-10-01' },
  { area: 'DNSH Requirements', change: 'Reduced DNSH evidence requirements for low-risk activities', direction: 'positive', date: '2026-01-01' },
];

const OmnibusTimeline: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Omnibus Simplification</Typography>
      <Alert severity="info" sx={{ mb: 2 }}>
        The EU Omnibus package simplifies sustainability reporting obligations. Key changes expected to take effect from October 2025 onwards.
      </Alert>
      <List dense>
        {DEMO_IMPACTS.map((impact, idx) => (
          <ListItem key={idx} sx={{ border: '1px solid #E0E0E0', borderRadius: 1, mb: 1 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              {impact.direction === 'positive' ? <TrendingUp color="success" /> : <TrendingDown color="error" />}
            </ListItemIcon>
            <ListItemText
              primary={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>{impact.area}</Typography>
                  <Chip label={impact.date} size="small" variant="outlined" />
                </Box>
              }
              secondary={impact.change}
            />
          </ListItem>
        ))}
      </List>
    </CardContent>
  </Card>
);

export default OmnibusTimeline;
