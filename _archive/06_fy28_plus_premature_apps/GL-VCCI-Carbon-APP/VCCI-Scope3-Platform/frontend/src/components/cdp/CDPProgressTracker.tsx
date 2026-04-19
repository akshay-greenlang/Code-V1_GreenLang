/**
 * CDPProgressTracker - Section completion tracking for CDP questionnaire
 *
 * Displays overall and per-section progress with visual indicators,
 * auto-populated vs manual breakdown charts, and data gap alerts.
 */

import React, { useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error as ErrorIcon,
  Info,
  AccessTime,
  AutoFixHigh,
  Edit as EditIcon,
} from '@mui/icons-material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  Legend,
} from 'recharts';
import type { CDPProgress } from '../../store/slices/cdpSlice';

// =============================================================================
// Props Interface
// =============================================================================

interface CDPProgressTrackerProps {
  progress: CDPProgress;
  deadline?: string;
}

// =============================================================================
// Constants
// =============================================================================

const PROGRESS_COLORS = {
  high: '#2e7d32',
  medium: '#ed6c02',
  low: '#d32f2f',
};

const PIE_COLORS = ['#2e7d32', '#ed6c02', '#9e9e9e'];

const SEVERITY_CONFIG = {
  critical: { color: 'error' as const, icon: <ErrorIcon fontSize="small" /> },
  warning: { color: 'warning' as const, icon: <Warning fontSize="small" /> },
  info: { color: 'info' as const, icon: <Info fontSize="small" /> },
};

// =============================================================================
// Helper Functions
// =============================================================================

function getProgressColor(percentage: number): string {
  if (percentage >= 80) return PROGRESS_COLORS.high;
  if (percentage >= 50) return PROGRESS_COLORS.medium;
  return PROGRESS_COLORS.low;
}

function computeDaysRemaining(deadline: string): number {
  const now = new Date();
  const deadlineDate = new Date(deadline);
  const diffMs = deadlineDate.getTime() - now.getTime();
  return Math.ceil(diffMs / (1000 * 60 * 60 * 24));
}

// =============================================================================
// Donut Chart Component
// =============================================================================

interface CompletionDonutProps {
  percentage: number;
  autoFilled: number;
  manual: number;
  unanswered: number;
}

const CompletionDonut: React.FC<CompletionDonutProps> = ({
  percentage,
  autoFilled,
  manual,
  unanswered,
}) => {
  const data = [
    { name: 'Auto-filled', value: autoFilled },
    { name: 'Manual', value: manual },
    { name: 'Unanswered', value: unanswered },
  ].filter((d) => d.value > 0);

  return (
    <Box sx={{ position: 'relative', width: '100%', height: 200 }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={80}
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={PIE_COLORS[['Auto-filled', 'Manual', 'Unanswered'].indexOf(entry.name)] || PIE_COLORS[2]}
              />
            ))}
          </Pie>
          <RechartsTooltip
            formatter={(value: number, name: string) => [
              `${value} questions`,
              name,
            ]}
          />
          <Legend
            verticalAlign="bottom"
            height={30}
            iconSize={10}
            formatter={(value: string) => (
              <span style={{ fontSize: '0.75rem', color: '#666' }}>{value}</span>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
      {/* Center label */}
      <Box
        sx={{
          position: 'absolute',
          top: '40%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
        }}
      >
        <Typography variant="h5" fontWeight="bold" color={getProgressColor(percentage)}>
          {percentage.toFixed(0)}%
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Complete
        </Typography>
      </Box>
    </Box>
  );
};

// =============================================================================
// Main Component
// =============================================================================

const CDPProgressTracker: React.FC<CDPProgressTrackerProps> = ({
  progress,
  deadline,
}) => {
  const unanswered = progress.totalQuestions - progress.answeredQuestions;

  const daysRemaining = useMemo(
    () => (deadline ? computeDaysRemaining(deadline) : null),
    [deadline]
  );

  const sortedGaps = useMemo(
    () =>
      [...progress.dataGaps].sort((a, b) => {
        const severityOrder = { critical: 0, warning: 1, info: 2 };
        return severityOrder[a.severity] - severityOrder[b.severity];
      }),
    [progress.dataGaps]
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Deadline countdown */}
      {deadline && daysRemaining !== null && (
        <Chip
          icon={<AccessTime />}
          label={
            daysRemaining > 0
              ? `${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} until deadline`
              : daysRemaining === 0
              ? 'Deadline is today'
              : `${Math.abs(daysRemaining)} day${Math.abs(daysRemaining) !== 1 ? 's' : ''} overdue`
          }
          color={
            daysRemaining > 30
              ? 'success'
              : daysRemaining > 7
              ? 'warning'
              : 'error'
          }
          variant="outlined"
          sx={{ justifyContent: 'flex-start' }}
        />
      )}

      {/* Overall completion donut */}
      <Card variant="outlined">
        <CardContent sx={{ pb: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Overall Completion
          </Typography>
          <CompletionDonut
            percentage={progress.overallCompletion}
            autoFilled={progress.autoFilledQuestions}
            manual={progress.manualQuestions}
            unanswered={unanswered}
          />

          {/* Summary stats */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              mt: 1,
              px: 1,
            }}
          >
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="success.main">
                {progress.autoFilledQuestions}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Auto-filled
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="warning.main">
                {progress.manualQuestions}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Manual
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" color="text.secondary">
                {unanswered}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Remaining
              </Typography>
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Section-by-section progress */}
      <Card variant="outlined">
        <CardContent sx={{ pb: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Section Progress
          </Typography>
          <List dense disablePadding>
            {progress.sectionProgress.map((section, index) => (
              <React.Fragment key={section.sectionId}>
                {index > 0 && <Divider />}
                <ListItem disableGutters sx={{ py: 0.75, flexDirection: 'column', alignItems: 'stretch' }}>
                  <Box
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      mb: 0.5,
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      {section.isValid ? (
                        <CheckCircle
                          sx={{ fontSize: 16, color: 'success.main' }}
                        />
                      ) : section.completionPercentage >= 50 ? (
                        <Warning sx={{ fontSize: 16, color: 'warning.main' }} />
                      ) : (
                        <ErrorIcon sx={{ fontSize: 16, color: 'error.main' }} />
                      )}
                      <Typography variant="body2" noWrap sx={{ maxWidth: 120 }}>
                        {section.sectionCode}
                      </Typography>
                    </Box>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ minWidth: 70, textAlign: 'right' }}
                    >
                      {section.answeredQuestions}/{section.totalQuestions}
                    </Typography>
                  </Box>

                  <LinearProgress
                    variant="determinate"
                    value={section.completionPercentage}
                    sx={{
                      height: 4,
                      borderRadius: 2,
                      backgroundColor: 'grey.200',
                      '& .MuiLinearProgress-bar': {
                        borderRadius: 2,
                        backgroundColor: getProgressColor(
                          section.completionPercentage
                        ),
                      },
                    }}
                  />

                  {/* Auto vs manual breakdown */}
                  <Box
                    sx={{
                      display: 'flex',
                      gap: 1,
                      mt: 0.25,
                      justifyContent: 'flex-end',
                    }}
                  >
                    {section.autoFilledQuestions > 0 && (
                      <Tooltip title="Auto-filled questions">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                          <AutoFixHigh sx={{ fontSize: 12, color: 'success.main' }} />
                          <Typography variant="caption" color="success.main">
                            {section.autoFilledQuestions}
                          </Typography>
                        </Box>
                      </Tooltip>
                    )}
                    {section.manualQuestions > 0 && (
                      <Tooltip title="Manual entries">
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.25 }}>
                          <EditIcon sx={{ fontSize: 12, color: 'warning.main' }} />
                          <Typography variant="caption" color="warning.main">
                            {section.manualQuestions}
                          </Typography>
                        </Box>
                      </Tooltip>
                    )}
                  </Box>
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        </CardContent>
      </Card>

      {/* Data gap alerts */}
      {sortedGaps.length > 0 && (
        <Card variant="outlined">
          <CardContent sx={{ pb: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Data Gaps ({sortedGaps.length})
            </Typography>
            <List dense disablePadding>
              {sortedGaps.map((gap, index) => {
                const config = SEVERITY_CONFIG[gap.severity];
                return (
                  <React.Fragment key={gap.id}>
                    {index > 0 && <Divider />}
                    <ListItem disableGutters sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 28 }}>
                        {config.icon}
                      </ListItemIcon>
                      <ListItemText
                        primary={
                          <Typography variant="body2" noWrap>
                            {gap.description}
                          </Typography>
                        }
                        secondary={
                          gap.suggestedAction && (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              noWrap
                            >
                              {gap.suggestedAction}
                            </Typography>
                          )
                        }
                      />
                      <Chip
                        label={gap.severity}
                        size="small"
                        color={config.color}
                        variant="outlined"
                        sx={{
                          height: 20,
                          fontSize: '0.65rem',
                          textTransform: 'capitalize',
                          ml: 0.5,
                        }}
                      />
                    </ListItem>
                  </React.Fragment>
                );
              })}
            </List>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default CDPProgressTracker;
