/**
 * ExposureUpload - Upload CSV/Excel with exposure data.
 */

import React from 'react';
import { Card, CardContent, Typography, Alert, Box, Button } from '@mui/material';
import { Upload } from '@mui/icons-material';
import EvidenceUploader from '../common/EvidenceUploader';

const ExposureUpload: React.FC = () => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Upload Exposure Data</Typography>
      <Alert severity="info" sx={{ mb: 2 }}>
        Upload a CSV or Excel file with counterparty exposures. Required columns: counterparty_name, lei, instrument_type, nominal_value, nace_code.
      </Alert>
      <EvidenceUploader label="Upload Exposure File" accept=".csv,.xlsx,.xls" maxFiles={1} />
      <Box sx={{ mt: 2 }}>
        <Button variant="outlined" size="small">Download Template</Button>
      </Box>
    </CardContent>
  </Card>
);

export default ExposureUpload;
