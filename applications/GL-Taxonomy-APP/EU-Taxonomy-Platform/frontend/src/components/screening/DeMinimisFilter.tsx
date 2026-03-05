/**
 * DeMinimisFilter - Toggle and configure de minimis threshold.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Switch, Slider, Box, FormControlLabel, Alert } from '@mui/material';

interface DeMinimisFilterProps {
  onThresholdChange?: (threshold: number, enabled: boolean) => void;
}

const DeMinimisFilter: React.FC<DeMinimisFilterProps> = ({ onThresholdChange }) => {
  const [enabled, setEnabled] = useState(false);
  const [threshold, setThreshold] = useState(5);

  const handleToggle = () => {
    const newEnabled = !enabled;
    setEnabled(newEnabled);
    onThresholdChange?.(threshold, newEnabled);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
          De Minimis Filter
        </Typography>
        <FormControlLabel
          control={<Switch checked={enabled} onChange={handleToggle} color="primary" />}
          label="Apply de minimis threshold"
        />
        {enabled && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" gutterBottom>
              Exclude activities below {threshold}% of turnover
            </Typography>
            <Slider
              value={threshold}
              onChange={(_, val) => { setThreshold(val as number); onThresholdChange?.(val as number, true); }}
              min={1}
              max={20}
              step={1}
              marks={[{ value: 1, label: '1%' }, { value: 5, label: '5%' }, { value: 10, label: '10%' }, { value: 20, label: '20%' }]}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => `${v}%`}
              color="primary"
            />
            <Alert severity="info" sx={{ mt: 1 }}>
              Activities contributing less than {threshold}% will be excluded from the eligibility count.
            </Alert>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default DeMinimisFilter;
