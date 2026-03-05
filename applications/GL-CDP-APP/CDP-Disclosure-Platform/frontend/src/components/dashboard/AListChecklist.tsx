/**
 * AListChecklist - A-level requirements checklist
 *
 * Displays the 5 mandatory A-level requirements with
 * pass/fail status indicators.
 */

import React from 'react';
import { Card, CardContent, Typography, Box, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { CheckCircle, Cancel, EmojiEvents } from '@mui/icons-material';
import type { ARequirement } from '../../types';

interface AListChecklistProps {
  requirements: ARequirement[];
  eligible: boolean;
}

const AListChecklist: React.FC<AListChecklistProps> = ({ requirements, eligible }) => {
  const metCount = requirements.filter((r) => r.met).length;

  return (
    <Card sx={{ borderLeft: eligible ? '4px solid #1b5e20' : '4px solid #c62828' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          <EmojiEvents sx={{ color: eligible ? '#1b5e20' : '#9e9e9e' }} />
          <Typography variant="h6">
            A-List Eligibility
          </Typography>
        </Box>
        <Typography
          variant="body2"
          fontWeight={600}
          sx={{ mb: 1.5, color: eligible ? '#1b5e20' : '#c62828' }}
        >
          {eligible ? 'All requirements met' : `${metCount}/${requirements.length} requirements met`}
        </Typography>
        <List dense disablePadding>
          {requirements.map((req) => (
            <ListItem key={req.id} disablePadding sx={{ mb: 0.5 }}>
              <ListItemIcon sx={{ minWidth: 32 }}>
                {req.met ? (
                  <CheckCircle fontSize="small" sx={{ color: '#2e7d32' }} />
                ) : (
                  <Cancel fontSize="small" sx={{ color: '#c62828' }} />
                )}
              </ListItemIcon>
              <ListItemText
                primary={req.description}
                secondary={req.details}
                primaryTypographyProps={{
                  variant: 'body2',
                  fontWeight: 500,
                  color: req.met ? 'text.primary' : 'error.main',
                }}
                secondaryTypographyProps={{ variant: 'caption' }}
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default AListChecklist;
