/**
 * OutcomeMonitor - Monitor and track safeguard outcomes.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, List, ListItem, ListItemText, Divider } from '@mui/material';

const DEMO_OUTCOMES = [
  { indicator: 'Workplace incidents', value: '0', target: '0', status: 'on_track' },
  { indicator: 'Discrimination complaints', value: '0', target: '0', status: 'on_track' },
  { indicator: 'Living wage compliance', value: '100%', target: '100%', status: 'on_track' },
  { indicator: 'Freedom of association coverage', value: '95%', target: '100%', status: 'at_risk' },
  { indicator: 'Child labor risk assessments', value: '87%', target: '100%', status: 'at_risk' },
];

const statusColor = (s: string) => s === 'on_track' ? 'success' : s === 'at_risk' ? 'warning' : 'error';

const OutcomeMonitor: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Outcome Monitoring
      </Typography>
      <List dense>
        {DEMO_OUTCOMES.map((o, idx) => (
          <React.Fragment key={idx}>
            <ListItem>
              <ListItemText
                primary={o.indicator}
                secondary={`Actual: ${o.value} | Target: ${o.target}`}
                primaryTypographyProps={{ fontWeight: 500, fontSize: '0.875rem' }}
              />
              <Chip
                label={o.status === 'on_track' ? 'On Track' : 'At Risk'}
                size="small"
                color={statusColor(o.status) as 'success' | 'warning' | 'error'}
              />
            </ListItem>
            {idx < DEMO_OUTCOMES.length - 1 && <Divider />}
          </React.Fragment>
        ))}
      </List>
    </CardContent>
  </Card>
);

export default OutcomeMonitor;
