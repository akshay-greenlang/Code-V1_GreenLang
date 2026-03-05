/**
 * ModuleNav - Module navigation sidebar for questionnaire
 *
 * Displays all 13 CDP modules with completion status indicators
 * and allows navigation between modules.
 */

import React from 'react';
import { List, ListItem, ListItemButton, ListItemText, Typography, Box, Chip } from '@mui/material';
import { CheckCircle, RadioButtonUnchecked, HourglassEmpty } from '@mui/icons-material';
import type { Module } from '../../types';
import { CDP_MODULE_NAMES, CDPModule, MODULE_COLORS } from '../../types';

interface ModuleNavProps {
  modules: Module[];
  currentModuleId: string | null;
  onSelectModule: (moduleId: string) => void;
}

const ModuleNav: React.FC<ModuleNavProps> = ({ modules, currentModuleId, onSelectModule }) => {
  const getIcon = (mod: Module) => {
    if (!mod.is_applicable) return <RadioButtonUnchecked fontSize="small" sx={{ color: '#bdbdbd' }} />;
    if (mod.completion_pct >= 100) return <CheckCircle fontSize="small" sx={{ color: '#2e7d32' }} />;
    if (mod.completion_pct > 0) return <HourglassEmpty fontSize="small" sx={{ color: '#ef6c00' }} />;
    return <RadioButtonUnchecked fontSize="small" sx={{ color: '#9e9e9e' }} />;
  };

  return (
    <Box sx={{ width: 280, borderRight: '1px solid #e0e0e0', height: '100%', overflow: 'auto' }}>
      <Typography variant="subtitle2" sx={{ p: 2, pb: 1, fontWeight: 600, color: 'text.secondary' }}>
        MODULES
      </Typography>
      <List dense disablePadding>
        {modules.map((mod) => {
          const isActive = mod.id === currentModuleId;
          const color = MODULE_COLORS[mod.module_code as CDPModule] || '#546e7a';
          return (
            <ListItem key={mod.id} disablePadding>
              <ListItemButton
                selected={isActive}
                onClick={() => onSelectModule(mod.id)}
                disabled={!mod.is_applicable}
                sx={{
                  px: 2,
                  py: 1,
                  '&.Mui-selected': {
                    backgroundColor: color + '10',
                    borderLeft: `3px solid ${color}`,
                  },
                }}
              >
                <Box sx={{ mr: 1.5 }}>{getIcon(mod)}</Box>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2" fontWeight={isActive ? 600 : 400}>
                        {mod.module_code} - {CDP_MODULE_NAMES[mod.module_code as CDPModule] || mod.name}
                      </Typography>
                      {mod.is_applicable && (
                        <Chip
                          label={`${mod.completion_pct.toFixed(0)}%`}
                          size="small"
                          sx={{
                            height: 20,
                            fontSize: '0.65rem',
                            backgroundColor: mod.completion_pct >= 100 ? '#e8f5e9' : '#fff3e0',
                            color: mod.completion_pct >= 100 ? '#2e7d32' : '#ef6c00',
                          }}
                        />
                      )}
                    </Box>
                  }
                  secondary={
                    mod.is_applicable
                      ? `${mod.answered_count}/${mod.question_count} questions`
                      : 'Not applicable'
                  }
                  secondaryTypographyProps={{ variant: 'caption' }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
    </Box>
  );
};

export default ModuleNav;
