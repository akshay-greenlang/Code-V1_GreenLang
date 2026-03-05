/**
 * BatchScreener - Upload and screen multiple NACE codes at once.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, TextField, Button, Box, Chip, Alert } from '@mui/material';
import { PlayArrow, Upload } from '@mui/icons-material';

interface BatchScreenerProps {
  onScreen?: (codes: string[]) => void;
}

const BatchScreener: React.FC<BatchScreenerProps> = ({ onScreen }) => {
  const [input, setInput] = useState('D35.11, C23.1, F41.2, H49.1, J62.0');
  const [codes, setCodes] = useState<string[]>([]);

  const parseCodes = () => {
    const parsed = input.split(/[,;\n]+/).map(c => c.trim()).filter(c => c.length > 0);
    setCodes(parsed);
    return parsed;
  };

  const handleScreen = () => {
    const parsed = parseCodes();
    onScreen?.(parsed);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Batch Eligibility Screening
        </Typography>

        <TextField
          fullWidth
          multiline
          rows={4}
          label="NACE Codes (comma or newline separated)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="D35.11, C23.1, F41.2..."
          sx={{ mb: 2 }}
        />

        {codes.length > 0 && (
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mb: 2 }}>
            {codes.map(code => (
              <Chip key={code} label={code} size="small" color="primary" variant="outlined" />
            ))}
          </Box>
        )}

        <Alert severity="info" sx={{ mb: 2 }}>
          Enter NACE codes to screen against the EU Taxonomy Delegated Acts. The system will identify which activities are eligible and for which environmental objectives.
        </Alert>

        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="contained" startIcon={<PlayArrow />} onClick={handleScreen}>
            Screen Activities
          </Button>
          <Button variant="outlined" startIcon={<Upload />}>
            Upload CSV
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default BatchScreener;
