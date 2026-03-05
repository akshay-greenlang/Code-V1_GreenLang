/**
 * ActionTimeline - Gantt-style timeline showing remediation action plan across quarters.
 *
 * Visualizes gap remediation actions as horizontal bars on a quarterly timeline,
 * color-coded by pillar and status, with milestones and dependencies.
 */

import React, { useMemo } from 'react';
import { Card, CardContent, Typography, Box, Chip, Tooltip as MUITooltip, LinearProgress } from '@mui/material';
import { CheckCircle, RadioButtonUnchecked, Schedule, Flag } from '@mui/icons-material';

interface ActionItem {
  id: string;
  action: string;
  pillar: string;
  owner: string;
  start_date: string;
  end_date: string;
  status: 'not_started' | 'planning' | 'in_progress' | 'completed';
  progress: number;
  priority: 'critical' | 'high' | 'medium' | 'low';
  milestones?: { date: string; label: string; completed: boolean }[];
}

interface ActionTimelineProps {
  actions: ActionItem[];
  startQuarter?: string;
  endQuarter?: string;
}

const PILLAR_COLORS: Record<string, string> = {
  Governance: '#1B5E20',
  Strategy: '#0D47A1',
  'Risk Management': '#E65100',
  'Metrics & Targets': '#6A1B9A',
};

const STATUS_ICONS: Record<string, React.ReactNode> = {
  completed: <CheckCircle sx={{ color: '#2E7D32', fontSize: 14 }} />,
  in_progress: <Schedule sx={{ color: '#EF6C00', fontSize: 14 }} />,
  planning: <Schedule sx={{ color: '#0D47A1', fontSize: 14 }} />,
  not_started: <RadioButtonUnchecked sx={{ color: '#9E9E9E', fontSize: 14 }} />,
};

const generateQuarters = (start: string, end: string): string[] => {
  const quarters: string[] = [];
  const [startYear, startQ] = start.split(' Q').map(Number);
  const [endYear, endQ] = end.split(' Q').map(Number);

  let y = startYear;
  let q = startQ;
  while (y < endYear || (y === endYear && q <= endQ)) {
    quarters.push(`${y} Q${q}`);
    q++;
    if (q > 4) {
      q = 1;
      y++;
    }
  }
  return quarters;
};

const dateToQuarter = (dateStr: string): string => {
  const d = new Date(dateStr);
  const q = Math.ceil((d.getMonth() + 1) / 3);
  return `${d.getFullYear()} Q${q}`;
};

const ActionTimeline: React.FC<ActionTimelineProps> = ({
  actions,
  startQuarter = '2025 Q1',
  endQuarter = '2026 Q2',
}) => {
  const quarters = useMemo(() => generateQuarters(startQuarter, endQuarter), [startQuarter, endQuarter]);

  const getBarPosition = (start: string, end: string): { left: number; width: number } => {
    const startQ = dateToQuarter(start);
    const endQ = dateToQuarter(end);
    const startIdx = Math.max(0, quarters.indexOf(startQ));
    const endIdx = Math.min(quarters.length - 1, quarters.indexOf(endQ));
    const totalWidth = quarters.length;
    const left = (startIdx / totalWidth) * 100;
    const width = Math.max(((endIdx - startIdx + 1) / totalWidth) * 100, 100 / totalWidth);
    return { left, width };
  };

  const sortedActions = useMemo(
    () => [...actions].sort((a, b) => {
      const pillarOrder: Record<string, number> = { Governance: 0, Strategy: 1, 'Risk Management': 2, 'Metrics & Targets': 3 };
      const pillarDiff = (pillarOrder[a.pillar] ?? 4) - (pillarOrder[b.pillar] ?? 4);
      if (pillarDiff !== 0) return pillarDiff;
      return new Date(a.start_date).getTime() - new Date(b.start_date).getTime();
    }),
    [actions]
  );

  const completedCount = actions.filter((a) => a.status === 'completed').length;
  const overallProgress = actions.length > 0
    ? Math.round(actions.reduce((s, a) => s + a.progress, 0) / actions.length)
    : 0;

  const todayQuarter = dateToQuarter(new Date().toISOString());
  const todayIdx = quarters.indexOf(todayQuarter);
  const todayLeft = todayIdx >= 0 ? ((todayIdx + 0.5) / quarters.length) * 100 : -1;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              Remediation Action Timeline
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {completedCount}/{actions.length} actions completed -- Overall progress: {overallProgress}%
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            {Object.entries(PILLAR_COLORS).map(([pillar, color]) => (
              <Chip
                key={pillar}
                label={pillar.split(' ')[0]}
                size="small"
                sx={{ backgroundColor: `${color}15`, color, fontSize: '0.65rem', fontWeight: 600, height: 22 }}
              />
            ))}
          </Box>
        </Box>

        {/* Overall Progress */}
        <LinearProgress
          variant="determinate"
          value={overallProgress}
          sx={{ height: 6, borderRadius: 3, mb: 3 }}
          color={overallProgress >= 75 ? 'success' : overallProgress >= 40 ? 'warning' : 'error'}
        />

        {/* Timeline Header */}
        <Box sx={{ display: 'flex', mb: 1, pl: '280px', position: 'relative' }}>
          {quarters.map((q, idx) => (
            <Box
              key={q}
              sx={{
                flex: 1,
                textAlign: 'center',
                borderLeft: '1px solid #E0E0E0',
                px: 0.5,
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  fontWeight: q === todayQuarter ? 700 : 400,
                  color: q === todayQuarter ? 'primary.main' : 'text.secondary',
                  fontSize: '0.7rem',
                }}
              >
                {q}
              </Typography>
            </Box>
          ))}
        </Box>

        {/* Timeline Rows */}
        <Box sx={{ position: 'relative' }}>
          {/* Today Marker */}
          {todayLeft >= 0 && (
            <Box
              sx={{
                position: 'absolute',
                left: `calc(280px + ${todayLeft}%)`,
                top: 0,
                bottom: 0,
                width: 2,
                backgroundColor: 'error.main',
                zIndex: 2,
                opacity: 0.6,
              }}
            />
          )}

          {sortedActions.map((action) => {
            const { left, width } = getBarPosition(action.start_date, action.end_date);
            const color = PILLAR_COLORS[action.pillar] || '#9E9E9E';

            return (
              <Box
                key={action.id}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  mb: 0.5,
                  minHeight: 36,
                  '&:hover': { backgroundColor: '#FAFAFA' },
                }}
              >
                {/* Label */}
                <Box
                  sx={{
                    width: 280,
                    flexShrink: 0,
                    pr: 1,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 0.5,
                    overflow: 'hidden',
                  }}
                >
                  {STATUS_ICONS[action.status]}
                  <MUITooltip title={`${action.action} (${action.owner})`} placement="right">
                    <Typography
                      variant="body2"
                      sx={{
                        fontSize: '0.75rem',
                        fontWeight: 500,
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        textDecoration: action.status === 'completed' ? 'line-through' : 'none',
                        color: action.status === 'completed' ? 'text.secondary' : 'text.primary',
                      }}
                    >
                      {action.action}
                    </Typography>
                  </MUITooltip>
                </Box>

                {/* Bar */}
                <Box
                  sx={{
                    flex: 1,
                    position: 'relative',
                    height: 24,
                    borderLeft: '1px solid #E0E0E0',
                  }}
                >
                  <MUITooltip
                    title={
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 600 }}>{action.action}</Typography>
                        <Typography variant="caption">Owner: {action.owner}</Typography>
                        <br />
                        <Typography variant="caption">
                          {new Date(action.start_date).toLocaleDateString()} - {new Date(action.end_date).toLocaleDateString()}
                        </Typography>
                        <br />
                        <Typography variant="caption">Progress: {action.progress}%</Typography>
                      </Box>
                    }
                  >
                    <Box
                      sx={{
                        position: 'absolute',
                        left: `${left}%`,
                        width: `${width}%`,
                        top: 2,
                        height: 20,
                        borderRadius: 1,
                        backgroundColor: action.status === 'completed' ? '#C8E6C9' : `${color}30`,
                        border: `1px solid ${action.status === 'completed' ? '#4CAF50' : color}`,
                        overflow: 'hidden',
                      }}
                    >
                      {/* Progress fill */}
                      <Box
                        sx={{
                          height: '100%',
                          width: `${action.progress}%`,
                          backgroundColor: action.status === 'completed' ? '#4CAF50' : color,
                          opacity: 0.6,
                          borderRadius: 0.5,
                        }}
                      />
                      {/* Milestones */}
                      {action.milestones?.map((milestone, idx) => {
                        const msQ = dateToQuarter(milestone.date);
                        const msIdx = quarters.indexOf(msQ);
                        if (msIdx < 0) return null;
                        const msLeft = ((msIdx + 0.5 - (left * quarters.length / 100)) / (width * quarters.length / 100)) * 100;
                        return (
                          <MUITooltip key={idx} title={`${milestone.label} (${milestone.completed ? 'Done' : 'Pending'})`}>
                            <Flag
                              sx={{
                                position: 'absolute',
                                top: -2,
                                left: `${msLeft}%`,
                                fontSize: 12,
                                color: milestone.completed ? '#2E7D32' : '#EF6C00',
                              }}
                            />
                          </MUITooltip>
                        );
                      })}
                    </Box>
                  </MUITooltip>
                </Box>
              </Box>
            );
          })}
        </Box>

        {/* Legend */}
        <Box sx={{ display: 'flex', gap: 2, mt: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Box sx={{ width: 2, height: 12, backgroundColor: 'error.main' }} />
            <Typography variant="caption">Today</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Flag sx={{ fontSize: 12, color: '#2E7D32' }} />
            <Typography variant="caption">Milestone (done)</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Flag sx={{ fontSize: 12, color: '#EF6C00' }} />
            <Typography variant="caption">Milestone (pending)</Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default ActionTimeline;
