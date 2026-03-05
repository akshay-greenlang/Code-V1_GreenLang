/**
 * SectorFilter - Multi-select filter for sectors and objectives.
 */

import React, { useState } from 'react';
import { Box, FormControl, InputLabel, Select, MenuItem, Chip, OutlinedInput, SelectChangeEvent } from '@mui/material';

const SECTORS = ['Energy', 'Manufacturing', 'Transport', 'Construction', 'ICT', 'Water', 'Forestry', 'Professional Services'];
const OBJECTIVES = ['CCM', 'CCA', 'WTR', 'CE', 'PPC', 'BIO'];

interface SectorFilterProps {
  onSectorChange?: (sectors: string[]) => void;
  onObjectiveChange?: (objectives: string[]) => void;
}

const SectorFilter: React.FC<SectorFilterProps> = ({ onSectorChange, onObjectiveChange }) => {
  const [sectors, setSectors] = useState<string[]>([]);
  const [objectives, setObjectives] = useState<string[]>([]);

  const handleSectorChange = (event: SelectChangeEvent<string[]>) => {
    const val = event.target.value as string[];
    setSectors(val);
    onSectorChange?.(val);
  };

  const handleObjectiveChange = (event: SelectChangeEvent<string[]>) => {
    const val = event.target.value as string[];
    setObjectives(val);
    onObjectiveChange?.(val);
  };

  return (
    <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
      <FormControl sx={{ minWidth: 200 }} size="small">
        <InputLabel>Sectors</InputLabel>
        <Select
          multiple
          value={sectors}
          onChange={handleSectorChange}
          input={<OutlinedInput label="Sectors" />}
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {selected.map(v => <Chip key={v} label={v} size="small" />)}
            </Box>
          )}
        >
          {SECTORS.map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
        </Select>
      </FormControl>

      <FormControl sx={{ minWidth: 200 }} size="small">
        <InputLabel>Objectives</InputLabel>
        <Select
          multiple
          value={objectives}
          onChange={handleObjectiveChange}
          input={<OutlinedInput label="Objectives" />}
          renderValue={(selected) => (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
              {selected.map(v => <Chip key={v} label={v} size="small" color="primary" />)}
            </Box>
          )}
        >
          {OBJECTIVES.map(o => <MenuItem key={o} value={o}>{o}</MenuItem>)}
        </Select>
      </FormControl>
    </Box>
  );
};

export default SectorFilter;
