/**
 * ARequirementsCheck - A-level requirements detail
 */
import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box, Chip } from '@mui/material';
import { CheckCircle, Cancel, EmojiEvents } from '@mui/icons-material';
import type { ARequirement } from '../../types';

interface ARequirementsCheckProps { requirements: ARequirement[]; }

const ARequirementsCheck: React.FC<ARequirementsCheckProps> = ({ requirements }) => {
  const allMet = requirements.every((r) => r.met);
  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
          <EmojiEvents sx={{ color: allMet ? '#1b5e20' : '#9e9e9e' }} />
          <Typography variant="h6">A-Level Requirements</Typography>
          <Chip label={allMet ? 'ELIGIBLE' : 'NOT ELIGIBLE'} color={allMet ? 'success' : 'error'} size="small" />
        </Box>
        <List dense>
          {requirements.map((req) => (
            <ListItem key={req.id}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                {req.met ? <CheckCircle sx={{ color: '#2e7d32' }} /> : <Cancel sx={{ color: '#c62828' }} />}
              </ListItemIcon>
              <ListItemText
                primary={req.description}
                secondary={req.details}
                primaryTypographyProps={{ variant: 'body2', fontWeight: req.met ? 400 : 600, color: req.met ? 'text.primary' : 'error.main' }}
                secondaryTypographyProps={{ variant: 'caption' }}
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default ARequirementsCheck;
