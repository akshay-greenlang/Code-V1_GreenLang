/**
 * ChangeLog - Historical response change log
 *
 * Lists changes between two reporting years with change type
 * (added/modified/removed), affected questions, and score impact.
 */
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Divider,
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  History,
  TrendingUp,
  TrendingDown,
} from '@mui/icons-material';
import type { ChangeLogEntry } from '../../types';
import { CDP_MODULE_NAMES, CDPModule } from '../../types';

interface ChangeLogProps {
  changes: ChangeLogEntry[];
  yearA: number;
  yearB: number;
}

function getChangeIcon(type: ChangeLogEntry['change_type']) {
  switch (type) {
    case 'added':
      return <Add sx={{ fontSize: 18, color: '#2e7d32' }} />;
    case 'modified':
      return <Edit sx={{ fontSize: 18, color: '#1565c0' }} />;
    case 'removed':
      return <Delete sx={{ fontSize: 18, color: '#e53935' }} />;
  }
}

function getChangeColor(
  type: ChangeLogEntry['change_type'],
): 'success' | 'info' | 'error' {
  switch (type) {
    case 'added': return 'success';
    case 'modified': return 'info';
    case 'removed': return 'error';
  }
}

const ChangeLog: React.FC<ChangeLogProps> = ({ changes, yearA, yearB }) => {
  const addedCount = changes.filter((c) => c.change_type === 'added').length;
  const modifiedCount = changes.filter((c) => c.change_type === 'modified').length;
  const removedCount = changes.filter((c) => c.change_type === 'removed').length;

  const totalImpact = changes.reduce((s, c) => s + c.impact_on_score, 0);

  // Group by module
  const moduleGroups = new Map<string, ChangeLogEntry[]>();
  changes.forEach((c) => {
    const key = c.module_code;
    const existing = moduleGroups.get(key) || [];
    existing.push(c);
    moduleGroups.set(key, existing);
  });

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <History sx={{ color: '#1565c0' }} />
            <Typography variant="h6">Change Log</Typography>
          </Box>
          <Typography variant="caption" color="text.secondary">
            {yearA} to {yearB}
          </Typography>
        </Box>

        {/* Summary chips */}
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip
            icon={<Add sx={{ fontSize: 14 }} />}
            label={`${addedCount} added`}
            size="small"
            color="success"
            variant="outlined"
          />
          <Chip
            icon={<Edit sx={{ fontSize: 14 }} />}
            label={`${modifiedCount} modified`}
            size="small"
            color="info"
            variant="outlined"
          />
          <Chip
            icon={<Delete sx={{ fontSize: 14 }} />}
            label={`${removedCount} removed`}
            size="small"
            color="error"
            variant="outlined"
          />
          <Chip
            icon={totalImpact >= 0
              ? <TrendingUp sx={{ fontSize: 14 }} />
              : <TrendingDown sx={{ fontSize: 14 }} />}
            label={`Net impact: ${totalImpact > 0 ? '+' : ''}${totalImpact.toFixed(1)} pts`}
            size="small"
            color={totalImpact >= 0 ? 'success' : 'error'}
          />
        </Box>

        <Divider sx={{ mb: 2 }} />

        {/* Changes grouped by module */}
        {Array.from(moduleGroups.entries()).map(([moduleCode, moduleChanges]) => (
          <Box key={moduleCode} sx={{ mb: 2 }}>
            <Typography variant="subtitle2" color="primary" gutterBottom>
              {CDP_MODULE_NAMES[moduleCode as CDPModule] || moduleCode}
            </Typography>

            {moduleChanges.map((change) => (
              <Box
                key={change.id}
                sx={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 1.5,
                  py: 0.75,
                  px: 1,
                  borderRadius: 1,
                  mb: 0.5,
                  '&:hover': { bgcolor: '#fafafa' },
                }}
              >
                {getChangeIcon(change.change_type)}

                <Box sx={{ flex: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" fontWeight={500}>
                      Q{change.question_number}
                    </Typography>
                    <Chip
                      label={change.change_type}
                      size="small"
                      color={getChangeColor(change.change_type)}
                      sx={{ fontSize: 10 }}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    {change.description}
                  </Typography>
                </Box>

                {change.impact_on_score !== 0 && (
                  <Typography
                    variant="body2"
                    fontWeight={600}
                    color={change.impact_on_score > 0 ? 'success.main' : 'error.main'}
                    sx={{ whiteSpace: 'nowrap' }}
                  >
                    {change.impact_on_score > 0 ? '+' : ''}{change.impact_on_score.toFixed(1)} pts
                  </Typography>
                )}
              </Box>
            ))}
          </Box>
        ))}

        {changes.length === 0 && (
          <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ py: 3 }}>
            No changes recorded between {yearA} and {yearB}.
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default ChangeLog;
