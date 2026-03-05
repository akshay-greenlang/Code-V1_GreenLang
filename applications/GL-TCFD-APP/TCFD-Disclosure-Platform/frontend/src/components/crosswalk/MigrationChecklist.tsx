/**
 * MigrationChecklist - Step-by-step checklist for migrating from TCFD to ISSB/IFRS S2.
 *
 * Displays categorized migration tasks with priority, effort, progress tracking,
 * and dependency awareness. Supports interactive completion toggling.
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Checkbox,
  LinearProgress,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  Divider,
  Alert,
  Button,
} from '@mui/material';
import {
  CheckCircle,
  RadioButtonUnchecked,
  Flag,
  Schedule,
  Warning,
  PlayArrow,
} from '@mui/icons-material';

interface MigrationTask {
  id: string;
  task: string;
  description: string;
  category: string;
  pillar: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  effort_weeks: number;
  effort_label: string;
  completed: boolean;
  dependencies: string[];
  assignee: string;
  due_date: string;
  issb_reference: string;
}

interface MigrationChecklistProps {
  tasks: MigrationTask[];
  onToggleComplete?: (taskId: string, completed: boolean) => void;
  onTaskClick?: (taskId: string) => void;
}

const PRIORITY_COLORS: Record<string, { chip: 'error' | 'warning' | 'info' | 'default'; icon: string }> = {
  critical: { chip: 'error', icon: '#C62828' },
  high: { chip: 'warning', icon: '#EF6C00' },
  medium: { chip: 'info', icon: '#0D47A1' },
  low: { chip: 'default', icon: '#9E9E9E' },
};

const CATEGORY_COLORS: Record<string, string> = {
  Planning: '#1B5E20',
  Governance: '#0D47A1',
  Strategy: '#7B1FA2',
  'Risk Management': '#E65100',
  Metrics: '#00838F',
  Reporting: '#795548',
};

const MigrationChecklist: React.FC<MigrationChecklistProps> = ({
  tasks,
  onToggleComplete,
  onTaskClick,
}) => {
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [priorityFilter, setPriorityFilter] = useState('all');
  const [showCompletedOnly, setShowCompletedOnly] = useState(false);

  const categories = useMemo(
    () => Array.from(new Set(tasks.map((t) => t.category))),
    [tasks]
  );

  const filteredTasks = useMemo(() => {
    let result = tasks;
    if (categoryFilter !== 'all') result = result.filter((t) => t.category === categoryFilter);
    if (priorityFilter !== 'all') result = result.filter((t) => t.priority === priorityFilter);
    if (showCompletedOnly) result = result.filter((t) => !t.completed);
    return result;
  }, [tasks, categoryFilter, priorityFilter, showCompletedOnly]);

  const completedCount = tasks.filter((t) => t.completed).length;
  const totalCount = tasks.length;
  const progressPct = totalCount > 0 ? Math.round((completedCount / totalCount) * 100) : 0;
  const totalEffort = tasks.filter((t) => !t.completed).reduce((s, t) => s + t.effort_weeks, 0);
  const criticalRemaining = tasks.filter((t) => !t.completed && t.priority === 'critical').length;

  const groupedByCategory = useMemo(() => {
    const groups: Record<string, MigrationTask[]> = {};
    filteredTasks.forEach((task) => {
      if (!groups[task.category]) groups[task.category] = [];
      groups[task.category].push(task);
    });
    return groups;
  }, [filteredTasks]);

  const hasBlockedTasks = useMemo(() => {
    return tasks.some((task) => {
      if (task.completed) return false;
      return task.dependencies.some((depId) => {
        const dep = tasks.find((t) => t.id === depId);
        return dep && !dep.completed;
      });
    });
  }, [tasks]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              ISSB Migration Checklist
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Step-by-step guide to achieving ISSB/IFRS S2 compliance from TCFD baseline
            </Typography>
          </Box>
          <Button
            variant="outlined"
            size="small"
            startIcon={<PlayArrow />}
            onClick={() => setShowCompletedOnly(!showCompletedOnly)}
          >
            {showCompletedOnly ? 'Show All' : 'Hide Completed'}
          </Button>
        </Box>

        {/* Progress Summary */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <Box sx={{ p: 2, backgroundColor: '#FAFAFA', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }}>Overall Progress</Typography>
                <Typography variant="body2" sx={{ fontWeight: 700 }}>{progressPct}%</Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={progressPct}
                sx={{ height: 10, borderRadius: 5 }}
                color={progressPct >= 75 ? 'success' : progressPct >= 40 ? 'warning' : 'error'}
              />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                {completedCount} of {totalCount} tasks completed
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} md={2}>
            <Box sx={{ p: 2, textAlign: 'center', backgroundColor: '#FFF3E0', borderRadius: 1 }}>
              <Schedule sx={{ color: '#EF6C00' }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{totalEffort}</Typography>
              <Typography variant="caption">Weeks remaining</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} md={2}>
            <Box sx={{ p: 2, textAlign: 'center', backgroundColor: criticalRemaining > 0 ? '#FFEBEE' : '#E8F5E9', borderRadius: 1 }}>
              <Warning sx={{ color: criticalRemaining > 0 ? '#C62828' : '#2E7D32' }} />
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{criticalRemaining}</Typography>
              <Typography variant="caption">Critical tasks left</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} md={4}>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {categories.map((cat) => {
                const catTasks = tasks.filter((t) => t.category === cat);
                const catDone = catTasks.filter((t) => t.completed).length;
                return (
                  <Chip
                    key={cat}
                    label={`${cat}: ${catDone}/${catTasks.length}`}
                    size="small"
                    sx={{
                      backgroundColor: `${CATEGORY_COLORS[cat] || '#9E9E9E'}15`,
                      color: CATEGORY_COLORS[cat] || '#9E9E9E',
                      fontWeight: 600,
                      fontSize: '0.65rem',
                    }}
                    onClick={() => setCategoryFilter(categoryFilter === cat ? 'all' : cat)}
                    variant={categoryFilter === cat ? 'filled' : 'outlined'}
                  />
                );
              })}
            </Box>
          </Grid>
        </Grid>

        {/* Filters */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={categoryFilter}
              label="Category"
              onChange={(e: SelectChangeEvent) => setCategoryFilter(e.target.value)}
            >
              <MenuItem value="all">All Categories</MenuItem>
              {categories.map((c) => (
                <MenuItem key={c} value={c}>{c}</MenuItem>
              ))}
            </Select>
          </FormControl>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Priority</InputLabel>
            <Select
              value={priorityFilter}
              label="Priority"
              onChange={(e: SelectChangeEvent) => setPriorityFilter(e.target.value)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="critical">Critical</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Blocked Task Warning */}
        {hasBlockedTasks && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Some tasks have unmet dependencies. Complete prerequisite tasks first to unblock downstream work.
          </Alert>
        )}

        {/* Task List by Category */}
        {Object.entries(groupedByCategory).map(([category, catTasks]) => (
          <Box key={category} sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Box
                sx={{
                  width: 4,
                  height: 20,
                  borderRadius: 2,
                  backgroundColor: CATEGORY_COLORS[category] || '#9E9E9E',
                }}
              />
              <Typography variant="subtitle2" sx={{ fontWeight: 700, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: 0.5 }}>
                {category}
              </Typography>
              <Chip
                label={`${catTasks.filter((t) => t.completed).length}/${catTasks.length}`}
                size="small"
                variant="outlined"
                sx={{ height: 20, fontSize: '0.65rem' }}
              />
            </Box>

            {catTasks
              .sort((a, b) => {
                const pOrder: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };
                return (pOrder[a.priority] ?? 4) - (pOrder[b.priority] ?? 4);
              })
              .map((task) => {
                const isBlocked = task.dependencies.some((depId) => {
                  const dep = tasks.find((t) => t.id === depId);
                  return dep && !dep.completed;
                });

                return (
                  <Box
                    key={task.id}
                    sx={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 1,
                      py: 1,
                      px: 1,
                      borderBottom: '1px solid #F5F5F5',
                      opacity: task.completed ? 0.6 : 1,
                      backgroundColor: isBlocked ? '#FFF8E1' : 'transparent',
                      borderRadius: 0.5,
                      cursor: onTaskClick ? 'pointer' : 'default',
                      '&:hover': { backgroundColor: task.completed ? undefined : '#FAFAFA' },
                    }}
                    onClick={() => onTaskClick?.(task.id)}
                  >
                    <Checkbox
                      checked={task.completed}
                      disabled={isBlocked && !task.completed}
                      size="small"
                      onChange={(e) => {
                        e.stopPropagation();
                        onToggleComplete?.(task.id, !task.completed);
                      }}
                      sx={{ mt: -0.5 }}
                    />

                    <Box sx={{ flex: 1 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.3 }}>
                        <Typography
                          variant="body2"
                          sx={{
                            fontWeight: 500,
                            textDecoration: task.completed ? 'line-through' : 'none',
                            fontSize: '0.85rem',
                          }}
                        >
                          {task.task}
                        </Typography>
                        {isBlocked && !task.completed && (
                          <Chip label="Blocked" size="small" color="warning" sx={{ height: 18, fontSize: '0.6rem' }} />
                        )}
                      </Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                        {task.description}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                        <Chip
                          label={task.priority}
                          size="small"
                          color={PRIORITY_COLORS[task.priority].chip}
                          sx={{ height: 18, fontSize: '0.6rem', textTransform: 'capitalize' }}
                        />
                        <Chip
                          label={task.effort_label}
                          size="small"
                          variant="outlined"
                          sx={{ height: 18, fontSize: '0.6rem' }}
                        />
                        <Chip
                          label={task.issb_reference}
                          size="small"
                          sx={{ height: 18, fontSize: '0.6rem', backgroundColor: '#E3F2FD', color: '#0D47A1' }}
                        />
                        {task.assignee && (
                          <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                            {task.assignee}
                          </Typography>
                        )}
                        {task.due_date && (
                          <Typography
                            variant="caption"
                            sx={{
                              ml: 'auto',
                              color: new Date(task.due_date) < new Date() && !task.completed ? 'error.main' : 'text.secondary',
                              fontWeight: new Date(task.due_date) < new Date() && !task.completed ? 600 : 400,
                            }}
                          >
                            Due: {new Date(task.due_date).toLocaleDateString()}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  </Box>
                );
              })}
          </Box>
        ))}

        {filteredTasks.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <CheckCircle sx={{ fontSize: 48, color: 'success.main', mb: 1 }} />
            <Typography variant="body2" color="text.secondary">
              {showCompletedOnly
                ? 'All tasks are completed! Migration to ISSB is fully ready.'
                : 'No tasks match the current filters.'}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default MigrationChecklist;
