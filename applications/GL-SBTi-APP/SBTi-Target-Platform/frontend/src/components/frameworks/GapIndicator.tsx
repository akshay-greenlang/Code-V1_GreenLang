/**
 * GapIndicator - Framework gap indicators.
 */
import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemText, Chip, Box } from '@mui/material';
import type { FrameworkMapping } from '../../types';

interface GapIndicatorProps { gaps: FrameworkMapping[]; }

const GapIndicator: React.FC<GapIndicatorProps> = ({ gaps }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Framework Gaps ({gaps.length})</Typography>
      {gaps.length === 0 ? (
        <Typography variant="body2" color="text.secondary">No gaps identified.</Typography>
      ) : (
        <List dense>
          {gaps.map((g) => (
            <ListItem key={g.id} disableGutters sx={{ borderBottom: '1px solid #F0F0F0', py: 1 }}>
              <ListItemText
                primary={<Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                  <Chip label={g.framework.toUpperCase()} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                  <Typography variant="body2" fontWeight={500}>{g.sbti_requirement}</Typography>
                </Box>}
                secondary={g.gap_description || 'No additional details'}
              />
              <Chip label={g.organization_status.replace(/_/g, ' ')} size="small"
                color={g.organization_status === 'met' ? 'success' : g.organization_status === 'partial' ? 'warning' : 'error'} />
            </ListItem>
          ))}
        </List>
      )}
    </CardContent>
  </Card>
);

export default GapIndicator;
