/**
 * TriggerAlerts - 5% threshold alerts.
 */
import React from 'react';
import { Card, CardContent, Typography, Alert, Box, Chip, List, ListItem, ListItemText } from '@mui/material';
import type { ThresholdCheck } from '../../types';

interface TriggerAlertsProps { checks: ThresholdCheck[]; }

const TriggerAlerts: React.FC<TriggerAlertsProps> = ({ checks }) => {
  const exceeding = checks.filter((c) => c.exceeds_threshold);
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Threshold Alerts</Typography>
        {exceeding.length > 0 ? (
          <Alert severity="warning" sx={{ mb: 2 }}>
            {exceeding.length} change(s) exceed the significance threshold and may require base year recalculation.
          </Alert>
        ) : (
          <Alert severity="success" sx={{ mb: 2 }}>All changes are within the significance threshold.</Alert>
        )}
        <List dense>
          {checks.map((c, i) => (
            <ListItem key={i} disableGutters>
              <ListItemText
                primary={c.trigger.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())}
                secondary={c.recommendation}
                primaryTypographyProps={{ fontWeight: 500, fontSize: '0.875rem' }}
              />
              <Chip label={`${c.change_pct.toFixed(1)}% / ${c.threshold_pct}%`} size="small"
                color={c.exceeds_threshold ? 'error' : 'success'} sx={{ fontWeight: 600 }} />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default TriggerAlerts;
