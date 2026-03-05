/**
 * CriteriaChecklist - Full C1-C28+ checklist with pass/fail indicators.
 */
import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Box, Chip } from '@mui/material';
import { CheckCircle, Cancel, Warning, HorizontalRule } from '@mui/icons-material';
import type { CriterionCheck } from '../../types';

interface CriteriaChecklistProps { criteria: CriterionCheck[]; }

const STATUS_ICONS: Record<string, React.ReactNode> = {
  pass: <CheckCircle sx={{ color: '#2E7D32', fontSize: 20 }} />,
  fail: <Cancel sx={{ color: '#C62828', fontSize: 20 }} />,
  warning: <Warning sx={{ color: '#EF6C00', fontSize: 20 }} />,
  not_applicable: <HorizontalRule sx={{ color: '#9E9E9E', fontSize: 20 }} />,
};

const CriteriaChecklist: React.FC<CriteriaChecklistProps> = ({ criteria }) => {
  const grouped = criteria.reduce<Record<string, CriterionCheck[]>>((acc, c) => {
    (acc[c.category] = acc[c.category] || []).push(c); return acc;
  }, {});

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>SBTi Criteria Checklist</Typography>
        {Object.entries(grouped).map(([category, items]) => (
          <Box key={category} sx={{ mb: 2 }}>
            <Typography variant="subtitle2" sx={{ mb: 1, textTransform: 'capitalize', fontWeight: 600, color: 'primary.main' }}>
              {category.replace(/_/g, ' ')}
            </Typography>
            <List dense>
              {items.map((item) => (
                <ListItem key={item.criterion_id} disableGutters sx={{ py: 0.25 }}>
                  <ListItemIcon sx={{ minWidth: 32 }}>{STATUS_ICONS[item.status]}</ListItemIcon>
                  <ListItemText
                    primary={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Chip label={item.criterion_code} size="small" sx={{ fontSize: '0.7rem', height: 20 }} />
                      <Typography variant="body2">{item.criterion_name}</Typography>
                    </Box>}
                    secondary={item.details}
                    secondaryTypographyProps={{ fontSize: '0.75rem' }}
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        ))}
      </CardContent>
    </Card>
  );
};

export default CriteriaChecklist;
