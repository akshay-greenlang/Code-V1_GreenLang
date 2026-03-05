/**
 * AlignmentStepper - Step-by-step workflow for full taxonomy alignment.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Stepper, Step, StepLabel, StepContent, Button, Box, Chip } from '@mui/material';

const STEPS = [
  { label: 'Eligibility Screening', desc: 'Determine if the activity is covered by the EU Taxonomy Delegated Acts.', done: true },
  { label: 'Substantial Contribution', desc: 'Assess whether the activity meets Technical Screening Criteria for at least one objective.', done: true },
  { label: 'DNSH Assessment', desc: 'Verify the activity does no significant harm to the remaining five objectives.', done: true },
  { label: 'Minimum Safeguards', desc: 'Confirm compliance with OECD Guidelines, UN Guiding Principles, ILO Core Conventions.', done: false },
  { label: 'Taxonomy Aligned', desc: 'Activity passes all four conditions and is considered taxonomy-aligned.', done: false },
];

const AlignmentStepper: React.FC = () => {
  const activeStep = STEPS.findIndex(s => !s.done);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Alignment Workflow</Typography>
        <Stepper activeStep={activeStep} orientation="vertical">
          {STEPS.map((step, idx) => (
            <Step key={step.label} completed={step.done}>
              <StepLabel>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {step.label}
                  {step.done && <Chip label="Complete" size="small" color="success" />}
                </Box>
              </StepLabel>
              <StepContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  {step.desc}
                </Typography>
                <Button variant="contained" size="small">
                  {idx === STEPS.length - 1 ? 'Finalize' : 'Proceed'}
                </Button>
              </StepContent>
            </Step>
          ))}
        </Stepper>
      </CardContent>
    </Card>
  );
};

export default AlignmentStepper;
