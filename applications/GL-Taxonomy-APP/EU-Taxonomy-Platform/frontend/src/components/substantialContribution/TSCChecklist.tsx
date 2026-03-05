/**
 * TSCChecklist - Technical Screening Criteria checklist with pass/fail.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Chip, Box } from '@mui/material';
import { CheckCircle, Cancel, HelpOutline } from '@mui/icons-material';

const DEMO_CRITERIA = [
  { id: 'TSC-CCM-1', name: 'Life-cycle GHG emissions < 100 gCO2e/kWh', met: true, actual: '15 gCO2e/kWh', required: '< 100 gCO2e/kWh' },
  { id: 'TSC-CCM-2', name: 'Direct emissions comply with Best Available Techniques', met: true, actual: 'Compliant', required: 'BAT compliance' },
  { id: 'TSC-CCM-3', name: 'No use of fossil fuel feedstock', met: true, actual: 'Solar PV', required: 'Non-fossil' },
  { id: 'TSC-CCM-4', name: 'Decommissioning plan in place', met: false, actual: 'In progress', required: 'Complete plan' },
  { id: 'TSC-CCM-5', name: 'Environmental impact assessment completed', met: true, actual: 'Completed 2024', required: 'Required' },
];

interface TSCChecklistProps {
  criteria?: typeof DEMO_CRITERIA;
}

const TSCChecklist: React.FC<TSCChecklistProps> = ({ criteria = DEMO_CRITERIA }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>
          Technical Screening Criteria
        </Typography>
        <Chip
          label={`${criteria.filter(c => c.met).length}/${criteria.length} passed`}
          color={criteria.every(c => c.met) ? 'success' : 'warning'}
          size="small"
        />
      </Box>
      <List dense>
        {criteria.map(criterion => (
          <ListItem key={criterion.id} sx={{ border: '1px solid #E0E0E0', borderRadius: 1, mb: 1 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              {criterion.met ? (
                <CheckCircle sx={{ color: '#2E7D32' }} />
              ) : (
                <Cancel sx={{ color: '#C62828' }} />
              )}
            </ListItemIcon>
            <ListItemText
              primary={criterion.name}
              secondary={
                <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                  <Typography variant="caption">Actual: {criterion.actual}</Typography>
                  <Typography variant="caption" color="text.secondary">Required: {criterion.required}</Typography>
                </Box>
              }
            />
            <Chip label={criterion.id} size="small" variant="outlined" />
          </ListItem>
        ))}
      </List>
    </CardContent>
  </Card>
);

export default TSCChecklist;
