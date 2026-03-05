/**
 * IssueList - Failed criteria with resolution guidance.
 */
import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemText, Chip, Box } from '@mui/material';
import type { ValidationIssue } from '../../types';

interface IssueListProps { issues: ValidationIssue[]; }

const SEVERITY_COLORS: Record<string, string> = { critical: '#B71C1C', major: '#E65100', minor: '#F57F17' };

const IssueList: React.FC<IssueListProps> = ({ issues }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>Issues to Resolve ({issues.length})</Typography>
      <List dense>
        {issues.map((issue) => (
          <ListItem key={issue.id} disableGutters sx={{ borderBottom: '1px solid #F0F0F0', py: 1 }}>
            <ListItemText
              primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Chip label={issue.severity} size="small" sx={{ backgroundColor: SEVERITY_COLORS[issue.severity] + '20', color: SEVERITY_COLORS[issue.severity], fontWeight: 600, fontSize: '0.7rem' }} />
                <Chip label={issue.criterion_code} size="small" variant="outlined" sx={{ fontSize: '0.7rem' }} />
                <Typography variant="body2" fontWeight={500}>{issue.title}</Typography>
              </Box>}
              secondary={<Box sx={{ mt: 0.5 }}>
                <Typography variant="caption" color="text.secondary">{issue.description}</Typography>
                {issue.resolution_guidance && (
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'primary.main', fontWeight: 500 }}>
                    Resolution: {issue.resolution_guidance}
                  </Typography>
                )}
              </Box>}
            />
          </ListItem>
        ))}
      </List>
    </CardContent>
  </Card>
);

export default IssueList;
