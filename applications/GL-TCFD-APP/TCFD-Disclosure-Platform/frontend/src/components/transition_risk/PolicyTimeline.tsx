import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Stepper, Step, StepLabel, StepContent } from '@mui/material';
import type { PolicyRisk } from '../../types';
import RiskBadge from '../common/RiskBadge';
import { formatDate, formatCurrency } from '../../utils/formatters';

interface PolicyTimelineProps { policies: PolicyRisk[]; }

const PolicyTimeline: React.FC<PolicyTimelineProps> = ({ policies }) => {
  const sorted = [...policies].sort((a, b) => a.effective_date.localeCompare(b.effective_date));
  return (
    <Card><CardContent>
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Regulatory Policy Timeline</Typography>
      <Stepper orientation="vertical" activeStep={-1}>
        {sorted.map((policy) => (
          <Step key={policy.id} active expanded>
            <StepLabel>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{policy.name}</Typography>
                <RiskBadge level={policy.risk_level} size="small" />
                <Chip label={policy.jurisdiction} size="small" variant="outlined" />
              </Box>
            </StepLabel>
            <StepContent>
              <Typography variant="body2" color="text.secondary">{policy.description}</Typography>
              <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                <Typography variant="caption">Effective: {formatDate(policy.effective_date)}</Typography>
                <Typography variant="caption">Impact: {formatCurrency(policy.financial_impact, 'USD', true)}</Typography>
                <Chip label={policy.compliance_status.replace(/_/g, ' ')} size="small"
                  color={policy.compliance_status === 'compliant' ? 'success' : policy.compliance_status === 'partial' ? 'warning' : 'error'} />
              </Box>
            </StepContent>
          </Step>
        ))}
      </Stepper>
      {policies.length === 0 && <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>No policy risks identified</Typography>}
    </CardContent></Card>
  );
};

export default PolicyTimeline;
