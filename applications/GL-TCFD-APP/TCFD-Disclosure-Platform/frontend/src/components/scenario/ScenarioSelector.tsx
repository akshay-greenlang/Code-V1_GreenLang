import React from 'react';
import { Card, CardContent, Typography, Grid, Box, Chip, Checkbox, Button } from '@mui/material';
import type { ScenarioDefinition } from '../../types';
import { getScenarioColor, getTemperatureColor } from '../../utils/scenarioHelpers';

interface ScenarioSelectorProps {
  scenarios: ScenarioDefinition[];
  selectedIds: string[];
  onToggle: (id: string) => void;
  onCompare: () => void;
}

const ScenarioSelector: React.FC<ScenarioSelectorProps> = ({ scenarios, selectedIds, onToggle, onCompare }) => (
  <Card>
    <CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600 }}>Select Scenarios for Analysis</Typography>
        <Button variant="contained" onClick={onCompare} disabled={selectedIds.length < 2}>
          Compare ({selectedIds.length})
        </Button>
      </Box>
      <Grid container spacing={2}>
        {scenarios.map((s) => (
          <Grid item xs={12} sm={6} md={3} key={s.id}>
            <Box
              onClick={() => onToggle(s.id)}
              sx={{
                p: 2, borderRadius: 2, cursor: 'pointer',
                border: selectedIds.includes(s.id)
                  ? `2px solid ${getScenarioColor(s.scenario_type)}`
                  : '2px solid #E0E0E0',
                bgcolor: selectedIds.includes(s.id)
                  ? `${getScenarioColor(s.scenario_type)}08`
                  : 'white',
                transition: 'all 0.2s',
                '&:hover': { borderColor: getScenarioColor(s.scenario_type) },
              }}
            >
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>{s.name}</Typography>
                <Checkbox checked={selectedIds.includes(s.id)} size="small" sx={{ p: 0 }} />
              </Box>
              <Box sx={{ display: 'flex', gap: 0.5, mt: 1, flexWrap: 'wrap' }}>
                <Chip label={s.temperature_target} size="small"
                  sx={{ bgcolor: getTemperatureColor(s.temperature_target), color: 'white', fontSize: 11 }} />
                <Chip label={`${s.time_horizon_years}yr`} size="small" variant="outlined" sx={{ fontSize: 11 }} />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                {s.description?.slice(0, 80)}{s.description && s.description.length > 80 ? '...' : ''}
              </Typography>
            </Box>
          </Grid>
        ))}
      </Grid>
      {scenarios.length === 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
          No scenarios defined. Create scenarios to begin analysis.
        </Typography>
      )}
    </CardContent>
  </Card>
);

export default ScenarioSelector;
