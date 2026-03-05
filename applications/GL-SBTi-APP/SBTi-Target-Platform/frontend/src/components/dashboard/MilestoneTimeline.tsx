/**
 * MilestoneTimeline - Key upcoming milestones display.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box } from '@mui/material';
import { FiberManualRecord } from '@mui/icons-material';
import { formatDate } from '../../utils/formatters';

interface Milestone {
  date: string;
  description: string;
  target_name: string;
}

interface MilestoneTimelineProps {
  milestones: Milestone[];
}

const MilestoneTimeline: React.FC<MilestoneTimelineProps> = ({ milestones }) => {
  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Upcoming Milestones</Typography>
        {milestones.length === 0 ? (
          <Typography variant="body2" color="text.secondary">No upcoming milestones</Typography>
        ) : (
          <List dense>
            {milestones.slice(0, 6).map((ms, idx) => (
              <ListItem key={idx} disableGutters sx={{ alignItems: 'flex-start' }}>
                <ListItemIcon sx={{ minWidth: 24, mt: 0.5 }}>
                  <FiberManualRecord sx={{ fontSize: 10, color: 'primary.main' }} />
                </ListItemIcon>
                <ListItemText
                  primary={ms.description}
                  secondary={
                    <Box component="span">
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(ms.date)} -- {ms.target_name}
                      </Typography>
                    </Box>
                  }
                  primaryTypographyProps={{ fontSize: '0.85rem', fontWeight: 500 }}
                />
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default MilestoneTimeline;
