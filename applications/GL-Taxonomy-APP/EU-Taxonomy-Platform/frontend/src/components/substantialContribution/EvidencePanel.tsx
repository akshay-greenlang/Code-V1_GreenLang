/**
 * EvidencePanel - Panel for uploading and viewing evidence documents.
 */

import React from 'react';
import { Card, CardContent, Typography, List, ListItem, ListItemIcon, ListItemText, Chip, Divider, Box } from '@mui/material';
import { InsertDriveFile, CheckCircle, Schedule } from '@mui/icons-material';
import EvidenceUploader from '../common/EvidenceUploader';

const DEMO_EVIDENCE = [
  { name: 'LCA_Report_2025.pdf', type: 'audit_report', verified: true, date: '2025-01-15' },
  { name: 'GHG_Emissions_Certificate.pdf', type: 'certificate', verified: true, date: '2025-02-10' },
  { name: 'BAT_Compliance_Assessment.docx', type: 'document', verified: false, date: '2025-03-01' },
  { name: 'Environmental_Monitoring_Data.xlsx', type: 'measurement', verified: false, date: '2025-03-05' },
];

const EvidencePanel: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        Evidence Documents
      </Typography>

      <List dense>
        {DEMO_EVIDENCE.map((doc, idx) => (
          <ListItem key={idx} sx={{ border: '1px solid #E0E0E0', borderRadius: 1, mb: 1 }}>
            <ListItemIcon sx={{ minWidth: 36 }}>
              <InsertDriveFile fontSize="small" color="action" />
            </ListItemIcon>
            <ListItemText
              primary={doc.name}
              secondary={`${doc.type} - ${doc.date}`}
              primaryTypographyProps={{ fontSize: '0.85rem' }}
            />
            <Box sx={{ display: 'flex', gap: 0.5 }}>
              {doc.verified ? (
                <Chip icon={<CheckCircle />} label="Verified" size="small" color="success" variant="outlined" />
              ) : (
                <Chip icon={<Schedule />} label="Pending" size="small" variant="outlined" />
              )}
            </Box>
          </ListItem>
        ))}
      </List>

      <Divider sx={{ my: 2 }} />

      <EvidenceUploader label="Upload Additional Evidence" />
    </CardContent>
  </Card>
);

export default EvidencePanel;
