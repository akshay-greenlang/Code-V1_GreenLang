/**
 * ReviewChecklist - Review readiness checklist.
 */
import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Checkbox, Box, LinearProgress } from '@mui/material';
import type { ReviewChecklistItem } from '../../types';

interface ReviewChecklistProps { items: ReviewChecklistItem[]; onToggle?: (id: string, completed: boolean) => void; }

const ReviewChecklist: React.FC<ReviewChecklistProps> = ({ items, onToggle }) => {
  const completed = items.filter((i) => i.status === 'completed').length;
  const pct = items.length > 0 ? (completed / items.length) * 100 : 0;
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>Review Checklist</Typography>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="body2" color="text.secondary">{completed} of {items.length} completed</Typography>
          <Typography variant="body2" fontWeight={600}>{pct.toFixed(0)}%</Typography>
        </Box>
        <LinearProgress variant="determinate" value={pct} color={pct === 100 ? 'success' : 'primary'} sx={{ height: 6, borderRadius: 3, mb: 2 }} />
        <List dense>
          {items.map((item) => (
            <ListItem key={item.id} disableGutters>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Checkbox edge="start" size="small" checked={item.status === 'completed'}
                  onChange={(e) => onToggle?.(item.id, e.target.checked)} />
              </ListItemIcon>
              <ListItemText primary={item.requirement} secondary={`${item.category} | Assigned: ${item.assigned_to}`}
                primaryTypographyProps={{ fontSize: '0.85rem', textDecoration: item.status === 'completed' ? 'line-through' : 'none' }}
                secondaryTypographyProps={{ fontSize: '0.75rem' }} />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default ReviewChecklist;
