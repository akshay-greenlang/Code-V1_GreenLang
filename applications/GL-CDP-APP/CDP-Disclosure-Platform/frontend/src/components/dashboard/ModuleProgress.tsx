/**
 * ModuleProgress - Module completion progress bars
 *
 * Displays progress bars for all 13 CDP modules with
 * answered/reviewed/approved counts.
 */

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';
import type { ModuleProgress as ModuleProgressData } from '../../types';
import ProgressBar from '../common/ProgressBar';
import { MODULE_COLORS, CDPModule } from '../../types';

interface ModuleProgressProps {
  modules: ModuleProgressData[];
}

const ModuleProgressComponent: React.FC<ModuleProgressProps> = ({ modules }) => {
  const applicableModules = modules.filter((m) => m.is_applicable);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Module Completion
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
          {applicableModules.map((mod) => (
            <Box key={mod.module_code}>
              <ProgressBar
                label={`${mod.module_code} - ${mod.module_name}`}
                value={mod.completion_pct}
                color={MODULE_COLORS[mod.module_code as CDPModule]}
              />
              <Typography variant="caption" color="text.secondary">
                {mod.answered}/{mod.total_questions} answered | {mod.reviewed} reviewed | {mod.approved} approved
              </Typography>
            </Box>
          ))}
        </Box>
      </CardContent>
    </Card>
  );
};

export default ModuleProgressComponent;
