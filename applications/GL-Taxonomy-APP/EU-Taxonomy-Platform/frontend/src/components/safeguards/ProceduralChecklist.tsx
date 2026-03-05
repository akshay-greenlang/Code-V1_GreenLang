/**
 * ProceduralChecklist - Checklist of procedural requirements for safeguards.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Checkbox } from '@mui/material';

const DEMO_ITEMS = [
  { id: 1, text: 'Due diligence process established per UNGP/OECD Guidelines', checked: true },
  { id: 2, text: 'Human rights policy publicly available', checked: true },
  { id: 3, text: 'Grievance mechanism established', checked: true },
  { id: 4, text: 'Regular impact assessments conducted', checked: false },
  { id: 5, text: 'Stakeholder engagement process documented', checked: true },
  { id: 6, text: 'Remediation process in place', checked: true },
  { id: 7, text: 'Supply chain due diligence implemented', checked: false },
  { id: 8, text: 'Board-level oversight of human rights', checked: true },
];

interface ProceduralChecklistProps {
  items?: typeof DEMO_ITEMS;
  onChange?: (id: number, checked: boolean) => void;
}

const ProceduralChecklist: React.FC<ProceduralChecklistProps> = ({ items = DEMO_ITEMS, onChange }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Procedural Requirements
      </Typography>
      <List dense>
        {items.map(item => (
          <ListItem key={item.id} sx={{ py: 0 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              <Checkbox
                checked={item.checked}
                onChange={(e) => onChange?.(item.id, e.target.checked)}
                size="small"
                color="primary"
              />
            </ListItemIcon>
            <ListItemText
              primary={item.text}
              primaryTypographyProps={{ fontSize: '0.85rem', textDecoration: item.checked ? 'none' : 'none' }}
            />
          </ListItem>
        ))}
      </List>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        {items.filter(i => i.checked).length}/{items.length} requirements met
      </Typography>
    </CardContent>
  </Card>
);

export default ProceduralChecklist;
