/**
 * ClimateRiskWizard - Step-by-step climate risk assessment for DNSH CCA.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Stepper, Step, StepLabel, Box, Button, FormControl, InputLabel, Select, MenuItem, Chip, Grid } from '@mui/material';

const PHYSICAL_RISKS = [
  { hazard: 'Heat stress', category: 'Chronic', likelihood: 'High', impact: 'Medium' },
  { hazard: 'Flooding', category: 'Acute', likelihood: 'Medium', impact: 'High' },
  { hazard: 'Drought', category: 'Chronic', likelihood: 'Low', impact: 'Low' },
  { hazard: 'Wildfire', category: 'Acute', likelihood: 'Low', impact: 'Medium' },
];

const STEPS = ['Identify Risks', 'Assess Vulnerability', 'Adaptation Plan', 'Review'];

const ClimateRiskWizard: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Climate Risk Assessment (DNSH - CCA)
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {STEPS.map(label => <Step key={label}><StepLabel>{label}</StepLabel></Step>)}
        </Stepper>

        {activeStep === 0 && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 2 }}>Physical Climate Risks Identified</Typography>
            <Grid container spacing={2}>
              {PHYSICAL_RISKS.map((risk, idx) => (
                <Grid item xs={12} sm={6} key={idx}>
                  <Box sx={{ p: 2, border: '1px solid #E0E0E0', borderRadius: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{risk.hazard}</Typography>
                    <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                      <Chip label={risk.category} size="small" variant="outlined" />
                      <Chip label={`Likelihood: ${risk.likelihood}`} size="small" color={risk.likelihood === 'High' ? 'error' : risk.likelihood === 'Medium' ? 'warning' : 'success'} />
                      <Chip label={`Impact: ${risk.impact}`} size="small" color={risk.impact === 'High' ? 'error' : risk.impact === 'Medium' ? 'warning' : 'success'} />
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {activeStep === 1 && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 2 }}>Vulnerability Assessment</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>Sensitivity</InputLabel>
                  <Select defaultValue="medium" label="Sensitivity">
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={6}>
                <FormControl fullWidth size="small">
                  <InputLabel>Adaptive Capacity</InputLabel>
                  <Select defaultValue="high" label="Adaptive Capacity">
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </Box>
        )}

        {activeStep === 2 && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 2 }}>Adaptation Measures</Typography>
            <Typography variant="body2" color="text.secondary">
              Define adaptation measures for identified risks. Each measure should address specific physical climate risks.
            </Typography>
          </Box>
        )}

        {activeStep === 3 && (
          <Box>
            <Typography variant="subtitle2" sx={{ mb: 2 }}>Review and Confirm</Typography>
            <Chip label="Overall: PASS" color="success" />
          </Box>
        )}

        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
          <Button disabled={activeStep === 0} onClick={() => setActiveStep(s => s - 1)}>
            Back
          </Button>
          <Button
            variant="contained"
            onClick={() => setActiveStep(s => Math.min(s + 1, STEPS.length - 1))}
          >
            {activeStep === STEPS.length - 1 ? 'Submit' : 'Next'}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ClimateRiskWizard;
