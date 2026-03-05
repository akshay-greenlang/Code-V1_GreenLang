import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Chip, Box, LinearProgress } from '@mui/material';
import { CheckCircle, RadioButtonUnchecked, Edit, RateReview } from '@mui/icons-material';

interface DisclosureChecklistProps { checklist: { code: string; title: string; pillar: string; status: string; completeness: number }[]; }

const STATUS_ICONS: Record<string, React.ReactNode> = { published: <CheckCircle color="success" />, final: <CheckCircle color="success" />, review: <RateReview color="primary" />, draft: <Edit color="warning" />, in_progress: <Edit color="action" />, not_started: <RadioButtonUnchecked color="disabled" /> };
const PILLAR_COLORS: Record<string, string> = { governance: '#1B5E20', strategy: '#0D47A1', risk_management: '#E65100', metrics_targets: '#6A1B9A' };

const DisclosureChecklist: React.FC<DisclosureChecklistProps> = ({ checklist }) => (
  <Card><CardContent>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>TCFD Disclosure Checklist (11 Recommendations)</Typography>
    <List dense>
      {checklist.map((item) => (
        <ListItem key={item.code} sx={{ borderBottom: '1px solid #F0F0F0' }}>
          <ListItemIcon sx={{ minWidth: 36 }}>{STATUS_ICONS[item.status] || <RadioButtonUnchecked />}</ListItemIcon>
          <ListItemText
            primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip label={item.code} size="small" sx={{ bgcolor: PILLAR_COLORS[item.pillar] || '#9E9E9E', color: 'white', fontSize: 10, fontWeight: 700 }} />
              <Typography variant="body2" sx={{ fontWeight: 500 }}>{item.title}</Typography>
            </Box>}
            secondary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
              <LinearProgress variant="determinate" value={item.completeness} sx={{ flexGrow: 1, height: 4, borderRadius: 2 }} />
              <Typography variant="caption" sx={{ minWidth: 30 }}>{item.completeness}%</Typography>
            </Box>}
          />
          <Chip label={item.status.replace(/_/g, ' ')} size="small" variant="outlined" />
        </ListItem>
      ))}
    </List>
    {checklist.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No checklist data</Typography>}
  </CardContent></Card>
);

export default DisclosureChecklist;
