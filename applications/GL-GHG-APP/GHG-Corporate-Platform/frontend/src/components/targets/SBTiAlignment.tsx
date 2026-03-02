/**
 * SBTiAlignment - Science Based Targets initiative alignment checker
 *
 * Displays SBTi alignment status with a requirements checklist
 * (near-term, long-term, scope coverage), alignment badge, required
 * reduction rate, and comparison to SBTi pathways (1.5C, well-below 2C).
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Chip,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Alert,
  LinearProgress,
} from '@mui/material';
import { CheckCircle, Cancel, RadioButtonUnchecked } from '@mui/icons-material';
import type { SBTiAlignmentCheck, Target } from '../../types';

interface SBTiAlignmentProps {
  target: Target | null;
  sbtiCheck: SBTiAlignmentCheck | null;
}

interface Requirement {
  label: string;
  met: boolean | null;
  description: string;
}

const SBTiAlignment: React.FC<SBTiAlignmentProps> = ({ target, sbtiCheck }) => {
  if (!target) {
    return (
      <Alert severity="info">
        Select a target to check SBTi alignment.
      </Alert>
    );
  }

  const isAligned = sbtiCheck?.is_aligned ?? target.is_sbti_aligned;
  const pathway = sbtiCheck?.pathway ?? target.sbti_pathway ?? 'Not set';
  const tempTarget = sbtiCheck?.temperature_target ?? '---';
  const requiredRate = sbtiCheck?.required_annual_reduction_percent ?? target.annual_reduction_rate;
  const actualRate = sbtiCheck?.actual_annual_reduction_percent ?? 0;
  const gap = sbtiCheck?.gap_percent ?? 0;

  const requirements: Requirement[] = [
    {
      label: 'Near-term target (5-10 years)',
      met: target.target_year - target.base_year <= 10 && target.target_year - target.base_year >= 5,
      description: 'Target timeframe between 5 and 10 years from base year.',
    },
    {
      label: 'Scope 1 and 2 coverage',
      met: target.scope_coverage?.some((s) => s === 'scope_1' || s === 'scope_2') ?? false,
      description: 'Target must cover Scope 1 and 2 emissions at minimum.',
    },
    {
      label: 'Minimum ambition level',
      met: isAligned,
      description: `Minimum ${requiredRate.toFixed(1)}% annual linear reduction required for ${pathway} pathway.`,
    },
    {
      label: 'Scope 3 screening complete',
      met: target.scope_coverage?.some((s) => s === 'scope_3') ?? null,
      description: 'Scope 3 screening required; target needed if Scope 3 > 40% of total.',
    },
  ];

  const metCount = requirements.filter((r) => r.met === true).length;
  const completionPercent = (metCount / requirements.length) * 100;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">SBTi Alignment</Typography>
          <Chip
            icon={
              isAligned
                ? <CheckCircle fontSize="small" />
                : <Cancel fontSize="small" />
            }
            label={isAligned ? 'Aligned' : 'Not Aligned'}
            color={isAligned ? 'success' : 'error'}
          />
        </Box>

        {/* Pathway info */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6} sm={3}>
            <Typography variant="caption" color="text.secondary">Pathway</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {String(pathway).replace(/_/g, ' ')}
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="caption" color="text.secondary">Temperature</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>{tempTarget}</Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="caption" color="text.secondary">Required Rate</Typography>
            <Typography variant="body1" sx={{ fontWeight: 600 }}>
              {requiredRate.toFixed(1)}%/yr
            </Typography>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Typography variant="caption" color="text.secondary">Actual Rate</Typography>
            <Typography
              variant="body1"
              sx={{
                fontWeight: 600,
                color: actualRate >= requiredRate ? 'success.main' : 'error.main',
              }}
            >
              {actualRate.toFixed(1)}%/yr
            </Typography>
          </Grid>
        </Grid>

        {/* Gap indicator */}
        {gap !== 0 && (
          <Alert severity={gap <= 0 ? 'success' : 'warning'} sx={{ mb: 2 }}>
            {gap <= 0
              ? 'Annual reduction rate exceeds SBTi minimum requirements.'
              : `Annual reduction rate is ${Math.abs(gap).toFixed(1)} percentage points below SBTi minimum.`}
          </Alert>
        )}

        <Divider sx={{ my: 2 }} />

        {/* Requirements checklist */}
        <Typography variant="subtitle2" gutterBottom>
          Requirements Checklist ({metCount}/{requirements.length})
        </Typography>
        <LinearProgress
          variant="determinate"
          value={completionPercent}
          sx={{ mb: 2, height: 6, borderRadius: 3 }}
        />

        <List dense disablePadding>
          {requirements.map((req, i) => (
            <ListItem key={i} sx={{ px: 0 }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                {req.met === true ? (
                  <CheckCircle color="success" fontSize="small" />
                ) : req.met === false ? (
                  <Cancel color="error" fontSize="small" />
                ) : (
                  <RadioButtonUnchecked color="disabled" fontSize="small" />
                )}
              </ListItemIcon>
              <ListItemText
                primary={req.label}
                secondary={req.description}
                primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                secondaryTypographyProps={{ variant: 'caption' }}
              />
            </ListItem>
          ))}
        </List>

        {/* Recommendations */}
        {sbtiCheck?.recommendations && sbtiCheck.recommendations.length > 0 && (
          <Box sx={{ mt: 2 }}>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="subtitle2" gutterBottom>Recommendations</Typography>
            {sbtiCheck.recommendations.map((rec, i) => (
              <Typography key={i} variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                - {rec}
              </Typography>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SBTiAlignment;
