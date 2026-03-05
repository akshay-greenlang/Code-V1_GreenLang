/**
 * GapList - Gap items list with severity color coding
 */
import React from 'react';
import { Box, Typography, List, ListItem, ListItemText, Chip, IconButton } from '@mui/material';
import { ChevronRight, CheckCircle } from '@mui/icons-material';
import type { GapItem } from '../../types';
import { getSeverityColor } from '../../utils/formatters';

interface GapListProps { gaps: GapItem[]; onSelect: (gap: GapItem) => void; onResolve: (gapId: string) => void; }

const GapList: React.FC<GapListProps> = ({ gaps, onSelect, onResolve }) => (
  <List disablePadding>
    {gaps.map((gap) => (
      <ListItem key={gap.id} divider sx={{ cursor: 'pointer', opacity: gap.is_resolved ? 0.5 : 1, '&:hover': { backgroundColor: '#f5f7f5' } }} onClick={() => onSelect(gap)}
        secondaryAction={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip label={`+${gap.uplift_points.toFixed(1)} pts`} size="small" variant="outlined" sx={{ height: 22, fontSize: '0.65rem' }} />
            {!gap.is_resolved && (
              <IconButton size="small" onClick={(e) => { e.stopPropagation(); onResolve(gap.id); }}>
                <CheckCircle fontSize="small" sx={{ color: '#2e7d32' }} />
              </IconButton>
            )}
            <ChevronRight fontSize="small" sx={{ color: 'text.secondary' }} />
          </Box>
        }
      >
        <ListItemText
          primary={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip label={gap.severity} size="small" color={getSeverityColor(gap.severity)} sx={{ height: 20, fontSize: '0.6rem' }} />
              <Chip label={gap.effort} size="small" variant="outlined" sx={{ height: 20, fontSize: '0.6rem' }} />
              <Typography variant="body2" fontWeight={500}>{gap.question_number}</Typography>
            </Box>
          }
          secondary={gap.gap_description}
          secondaryTypographyProps={{ variant: 'caption', noWrap: true }}
        />
      </ListItem>
    ))}
    {gaps.length === 0 && (
      <ListItem><ListItemText primary="No gaps identified" primaryTypographyProps={{ color: 'text.secondary', textAlign: 'center' }} /></ListItem>
    )}
  </List>
);

export default GapList;
