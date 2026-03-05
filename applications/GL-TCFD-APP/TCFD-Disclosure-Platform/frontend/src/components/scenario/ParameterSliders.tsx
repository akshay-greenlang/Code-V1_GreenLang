import React from 'react';
import { Card, CardContent, Typography, Box, Slider, Grid } from '@mui/material';
import type { ScenarioParameter } from '../../types';

interface ParameterSlidersProps {
  parameters: ScenarioParameter[];
  onValueChange: (id: string, value: number) => void;
}

const ParameterSliders: React.FC<ParameterSlidersProps> = ({ parameters, onValueChange }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Scenario Parameters</Typography>
      <Grid container spacing={3}>
        {parameters.map((param) => (
          <Grid item xs={12} sm={6} md={4} key={param.id}>
            <Box>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 500 }}>{param.name}</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700, color: 'primary.main' }}>
                  {param.current_value} {param.unit}
                </Typography>
              </Box>
              <Slider
                value={param.current_value}
                min={param.min_value}
                max={param.max_value}
                step={(param.max_value - param.min_value) / 100}
                onChange={(_, value) => onValueChange(param.id, value as number)}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => `${v} ${param.unit}`}
                size="small"
              />
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">{param.min_value}</Typography>
                <Typography variant="caption" color="text.secondary">{param.max_value}</Typography>
              </Box>
            </Box>
          </Grid>
        ))}
      </Grid>
      {parameters.length === 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
          Select a scenario to configure parameters
        </Typography>
      )}
    </CardContent>
  </Card>
);

export default ParameterSliders;
