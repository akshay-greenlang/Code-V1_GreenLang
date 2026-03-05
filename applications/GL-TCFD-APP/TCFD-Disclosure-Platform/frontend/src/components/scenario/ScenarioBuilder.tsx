import React, { useState } from 'react';
import { Card, CardContent, Typography, TextField, Select, MenuItem, FormControl, InputLabel, Grid, Button, Box, Stepper, Step, StepLabel, SelectChangeEvent } from '@mui/material';
import type { ScenarioType, TemperatureTarget } from '../../types';

interface ScenarioBuilderProps {
  onSubmit: (data: { name: string; description: string; scenario_type: ScenarioType; temperature_target: TemperatureTarget; time_horizon_years: number; base_year: number }) => void;
}

const ScenarioBuilder: React.FC<ScenarioBuilderProps> = ({ onSubmit }) => {
  const [step, setStep] = useState(0);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [scenarioType, setScenarioType] = useState<ScenarioType>('orderly_transition');
  const [tempTarget, setTempTarget] = useState<TemperatureTarget>('2.0C');
  const [horizon, setHorizon] = useState(30);
  const [baseYear, setBaseYear] = useState(2025);

  const steps = ['Basic Info', 'Configuration', 'Review'];

  const handleSubmit = () => {
    onSubmit({ name, description, scenario_type: scenarioType, temperature_target: tempTarget, time_horizon_years: horizon, base_year: baseYear });
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Create Custom Scenario</Typography>
        <Stepper activeStep={step} sx={{ mb: 3 }}>
          {steps.map((label) => <Step key={label}><StepLabel>{label}</StepLabel></Step>)}
        </Stepper>
        {step === 0 && (
          <Grid container spacing={2}>
            <Grid item xs={12}><TextField fullWidth label="Scenario Name" value={name} onChange={(e) => setName(e.target.value)} /></Grid>
            <Grid item xs={12}><TextField fullWidth multiline rows={3} label="Description" value={description} onChange={(e) => setDescription(e.target.value)} /></Grid>
          </Grid>
        )}
        {step === 1 && (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth><InputLabel>Scenario Type</InputLabel>
                <Select value={scenarioType} label="Scenario Type" onChange={(e: SelectChangeEvent) => setScenarioType(e.target.value as ScenarioType)}>
                  <MenuItem value="orderly_transition">Orderly Transition</MenuItem>
                  <MenuItem value="disorderly_transition">Disorderly Transition</MenuItem>
                  <MenuItem value="hot_house">Hot House World</MenuItem>
                  <MenuItem value="net_zero_2050">Net Zero 2050</MenuItem>
                  <MenuItem value="delayed_transition">Delayed Transition</MenuItem>
                  <MenuItem value="current_policies">Current Policies</MenuItem>
                  <MenuItem value="custom">Custom</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth><InputLabel>Temperature Target</InputLabel>
                <Select value={tempTarget} label="Temperature Target" onChange={(e: SelectChangeEvent) => setTempTarget(e.target.value as TemperatureTarget)}>
                  {(['1.5C', '2.0C', '2.5C', '3.0C', '4.0C'] as const).map((t) => (
                    <MenuItem key={t} value={t}>{t}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}><TextField fullWidth type="number" label="Time Horizon (years)" value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} /></Grid>
            <Grid item xs={12} sm={6}><TextField fullWidth type="number" label="Base Year" value={baseYear} onChange={(e) => setBaseYear(Number(e.target.value))} /></Grid>
          </Grid>
        )}
        {step === 2 && (
          <Box sx={{ p: 2, bgcolor: '#FAFAFA', borderRadius: 1 }}>
            <Typography variant="subtitle2">Name: {name}</Typography>
            <Typography variant="body2">Type: {scenarioType.replace(/_/g, ' ')}</Typography>
            <Typography variant="body2">Temperature: {tempTarget}</Typography>
            <Typography variant="body2">Horizon: {horizon} years from {baseYear}</Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>{description}</Typography>
          </Box>
        )}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 3 }}>
          {step > 0 && <Button onClick={() => setStep(step - 1)}>Back</Button>}
          {step < 2
            ? <Button variant="contained" onClick={() => setStep(step + 1)} disabled={step === 0 && !name}>Next</Button>
            : <Button variant="contained" onClick={handleSubmit}>Create Scenario</Button>
          }
        </Box>
      </CardContent>
    </Card>
  );
};

export default ScenarioBuilder;
