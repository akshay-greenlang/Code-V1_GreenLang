/**
 * ExportDialog - Dialog for exporting reports in various formats.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Box, Button, FormControl, InputLabel, Select, MenuItem, Checkbox, FormControlLabel, Grid, Alert } from '@mui/material';
import { Download, PictureAsPdf, TableChart, Code, DataObject } from '@mui/icons-material';

const FORMATS = [
  { id: 'pdf', label: 'PDF Report', icon: <PictureAsPdf />, desc: 'Full disclosure report with charts' },
  { id: 'excel', label: 'Excel Workbook', icon: <TableChart />, desc: 'Editable data tables' },
  { id: 'csv', label: 'CSV Data', icon: <DataObject />, desc: 'Raw data for analysis' },
  { id: 'xbrl', label: 'XBRL/iXBRL', icon: <Code />, desc: 'Machine-readable regulatory format' },
];

interface ExportDialogProps {
  onExport?: (format: string, options: Record<string, boolean>) => void;
}

const ExportDialog: React.FC<ExportDialogProps> = ({ onExport }) => {
  const [format, setFormat] = useState('pdf');
  const [includeAnnexes, setIncludeAnnexes] = useState(true);
  const [includeMethodology, setIncludeMethodology] = useState(true);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Export Report</Typography>
        <Grid container spacing={1} sx={{ mb: 2 }}>
          {FORMATS.map(f => (
            <Grid item xs={6} sm={3} key={f.id}>
              <Box
                onClick={() => setFormat(f.id)}
                sx={{
                  p: 2, border: '1px solid', borderColor: format === f.id ? 'primary.main' : '#E0E0E0',
                  borderRadius: 1, textAlign: 'center', cursor: 'pointer',
                  backgroundColor: format === f.id ? '#E8F5E9' : 'transparent',
                }}
              >
                <Box sx={{ color: format === f.id ? 'primary.main' : 'text.secondary', mb: 1 }}>{f.icon}</Box>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>{f.label}</Typography>
                <Typography variant="caption" color="text.secondary">{f.desc}</Typography>
              </Box>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <FormControlLabel control={<Checkbox checked={includeAnnexes} onChange={(e) => setIncludeAnnexes(e.target.checked)} size="small" />} label="Include annexes" />
          <FormControlLabel control={<Checkbox checked={includeMethodology} onChange={(e) => setIncludeMethodology(e.target.checked)} size="small" />} label="Include methodology" />
        </Box>

        <Button
          variant="contained"
          startIcon={<Download />}
          onClick={() => onExport?.(format, { includeAnnexes, includeMethodology })}
          fullWidth
        >
          Export as {FORMATS.find(f => f.id === format)?.label}
        </Button>
      </CardContent>
    </Card>
  );
};

export default ExportDialog;
