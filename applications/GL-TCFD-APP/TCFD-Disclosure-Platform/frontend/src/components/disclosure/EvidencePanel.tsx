import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Chip, Box, IconButton } from '@mui/material';
import { Description, Link, Delete, Add } from '@mui/icons-material';
import type { Evidence } from '../../types';
import { formatDate } from '../../utils/formatters';

interface EvidencePanelProps { evidence: Evidence[]; linkedIds: string[]; onLink: (evidenceId: string) => void; onUnlink: (evidenceId: string) => void; }

const EvidencePanel: React.FC<EvidencePanelProps> = ({ evidence, linkedIds, onLink, onUnlink }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Evidence & Supporting Documents</Typography>
    <List dense>
      {evidence.map((ev) => {
        const isLinked = linkedIds.includes(ev.id);
        return (
          <ListItem key={ev.id} secondaryAction={
            isLinked ? <IconButton edge="end" onClick={() => onUnlink(ev.id)} size="small"><Delete fontSize="small" /></IconButton>
            : <IconButton edge="end" onClick={() => onLink(ev.id)} size="small"><Add fontSize="small" /></IconButton>
          } sx={{ bgcolor: isLinked ? 'rgba(27, 94, 32, 0.04)' : 'transparent', borderRadius: 1, mb: 0.5 }}>
            <ListItemIcon sx={{ minWidth: 36 }}><Description color={isLinked ? 'primary' : 'action'} fontSize="small" /></ListItemIcon>
            <ListItemText
              primary={<Typography variant="body2" sx={{ fontWeight: isLinked ? 600 : 400 }}>{ev.title}</Typography>}
              secondary={<Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                <Typography variant="caption">{ev.type.replace(/_/g, ' ')} | {formatDate(ev.date)}</Typography>
                {ev.tags.slice(0, 3).map((t) => <Chip key={t} label={t} size="small" variant="outlined" sx={{ fontSize: 9, height: 18 }} />)}
              </Box>}
            />
          </ListItem>
        );
      })}
    </List>
    {evidence.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No evidence documents available</Typography>}
  </CardContent></Card>
);

export default EvidencePanel;
