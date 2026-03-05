/**
 * MilestoneTimeline - Transition plan milestone timeline
 *
 * Displays transition milestones in a vertical timeline layout
 * with status indicators, progress bars, and term labels
 * (short/medium/long-term).
 */
import React from 'react';
import { Card, CardContent, Typography, Box, Chip, LinearProgress } from '@mui/material';
import {
  CheckCircle,
  RadioButtonUnchecked,
  Pending,
  Warning,
} from '@mui/icons-material';
import type { TransitionMilestone } from '../../types';
import { TransitionMilestoneStatus } from '../../types';

interface MilestoneTimelineProps {
  milestones: TransitionMilestone[];
}

function getStatusIcon(status: TransitionMilestoneStatus) {
  switch (status) {
    case TransitionMilestoneStatus.COMPLETED:
      return <CheckCircle sx={{ color: '#2e7d32', fontSize: 24 }} />;
    case TransitionMilestoneStatus.IN_PROGRESS:
      return <Pending sx={{ color: '#1565c0', fontSize: 24 }} />;
    case TransitionMilestoneStatus.DELAYED:
      return <Warning sx={{ color: '#e53935', fontSize: 24 }} />;
    case TransitionMilestoneStatus.NOT_STARTED:
    default:
      return <RadioButtonUnchecked sx={{ color: '#9e9e9e', fontSize: 24 }} />;
  }
}

function getStatusColor(
  status: TransitionMilestoneStatus,
): 'success' | 'info' | 'error' | 'default' {
  switch (status) {
    case TransitionMilestoneStatus.COMPLETED:
      return 'success';
    case TransitionMilestoneStatus.IN_PROGRESS:
      return 'info';
    case TransitionMilestoneStatus.DELAYED:
      return 'error';
    default:
      return 'default';
  }
}

function getTermLabel(milestone: TransitionMilestone): string {
  if (milestone.is_short_term) return 'Short-term';
  if (milestone.is_medium_term) return 'Medium-term';
  if (milestone.is_long_term) return 'Long-term';
  return '';
}

function getTermColor(milestone: TransitionMilestone): string {
  if (milestone.is_short_term) return '#2e7d32';
  if (milestone.is_medium_term) return '#1565c0';
  if (milestone.is_long_term) return '#7b1fa2';
  return '#9e9e9e';
}

const MilestoneTimeline: React.FC<MilestoneTimelineProps> = ({ milestones }) => {
  const sorted = [...milestones].sort((a, b) => a.target_year - b.target_year);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Transition Milestones
        </Typography>

        {sorted.map((ms, idx) => (
          <Box
            key={ms.id}
            sx={{
              display: 'flex',
              gap: 2,
              mb: idx < sorted.length - 1 ? 0 : 0,
              position: 'relative',
            }}
          >
            {/* Timeline connector */}
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                minWidth: 32,
              }}
            >
              {getStatusIcon(ms.status)}
              {idx < sorted.length - 1 && (
                <Box
                  sx={{
                    width: 2,
                    flex: 1,
                    bgcolor: ms.status === TransitionMilestoneStatus.COMPLETED
                      ? '#2e7d32'
                      : '#e0e0e0',
                    my: 0.5,
                  }}
                />
              )}
            </Box>

            {/* Content */}
            <Box sx={{ flex: 1, pb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                <Typography variant="subtitle2" fontWeight={600}>
                  {ms.title}
                </Typography>
                <Chip
                  label={ms.target_year}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: 11 }}
                />
                <Chip
                  label={ms.status.replace('_', ' ')}
                  size="small"
                  color={getStatusColor(ms.status)}
                  sx={{ fontSize: 11 }}
                />
                {getTermLabel(ms) && (
                  <Chip
                    label={getTermLabel(ms)}
                    size="small"
                    sx={{
                      fontSize: 11,
                      bgcolor: getTermColor(ms) + '15',
                      color: getTermColor(ms),
                      border: `1px solid ${getTermColor(ms)}40`,
                    }}
                  />
                )}
              </Box>

              {ms.description && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  {ms.description}
                </Typography>
              )}

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <LinearProgress
                  variant="determinate"
                  value={ms.progress_pct}
                  sx={{
                    flex: 1,
                    height: 6,
                    borderRadius: 3,
                    bgcolor: '#e0e0e0',
                    '& .MuiLinearProgress-bar': {
                      bgcolor: ms.status === TransitionMilestoneStatus.DELAYED
                        ? '#e53935'
                        : ms.status === TransitionMilestoneStatus.COMPLETED
                          ? '#2e7d32'
                          : '#1565c0',
                      borderRadius: 3,
                    },
                  }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ minWidth: 32 }}>
                  {ms.progress_pct.toFixed(0)}%
                </Typography>
              </Box>

              <Box sx={{ display: 'flex', gap: 2, mt: 0.5 }}>
                <Typography variant="caption" color="text.secondary">
                  Target: -{ms.target_reduction_pct.toFixed(0)}%
                </Typography>
                {ms.responsible && (
                  <Typography variant="caption" color="text.secondary">
                    Owner: {ms.responsible}
                  </Typography>
                )}
              </Box>
            </Box>
          </Box>
        ))}

        {sorted.length === 0 && (
          <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 3 }}>
            No milestones defined. Add milestones to track transition progress.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default MilestoneTimeline;
