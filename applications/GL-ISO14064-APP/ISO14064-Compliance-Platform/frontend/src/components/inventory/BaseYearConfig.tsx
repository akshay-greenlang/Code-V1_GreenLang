/**
 * GL-ISO14064-APP v1.0 - Base Year Configuration
 *
 * Form component for setting and managing the base year per
 * ISO 14064-1 Clause 5.3.  Displays current base year, emissions,
 * recalculation policy, and recent triggers.
 */

import React, { useState } from 'react';
import {
  Card, CardContent, Typography, Box, TextField, Button,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Chip, Divider, Alert,
} from '@mui/material';
import type { BaseYearRecord, BaseYearTrigger } from '../../types';

interface Props {
  baseYear: BaseYearRecord | null;
  triggers: BaseYearTrigger[];
  onSetBaseYear: (data: { base_year: number; emissions_tco2e: number; recalculation_policy?: string }) => void;
  onRecalculate: (data: { trigger_type: string; description: string; new_emissions_tco2e: number }) => void;
}

const BaseYearConfig: React.FC<Props> = ({ baseYear, triggers, onSetBaseYear, onRecalculate }) => {
  const [year, setYear] = useState(baseYear?.base_year ?? new Date().getFullYear() - 1);
  const [emissions, setEmissions] = useState(baseYear?.original_emissions_tco2e ?? 0);
  const [policy, setPolicy] = useState(baseYear?.recalculation_policy ?? 'Recalculate when structural changes exceed 5% of total emissions.');
  const [showRecalc, setShowRecalc] = useState(false);
  const [triggerType, setTriggerType] = useState('');
  const [triggerDesc, setTriggerDesc] = useState('');
  const [newEmissions, setNewEmissions] = useState(0);

  const handleSave = () => {
    onSetBaseYear({ base_year: year, emissions_tco2e: emissions, recalculation_policy: policy });
  };

  const handleRecalculate = () => {
    onRecalculate({ trigger_type: triggerType, description: triggerDesc, new_emissions_tco2e: newEmissions });
    setShowRecalc(false);
    setTriggerType('');
    setTriggerDesc('');
    setNewEmissions(0);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Base Year Configuration (Clause 5.3)
        </Typography>

        {baseYear && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Current base year: <strong>{baseYear.base_year}</strong> |
            Original: {baseYear.original_emissions_tco2e.toLocaleString()} tCO2e
            {baseYear.recalculated_emissions_tco2e != null && (
              <> | Recalculated: {baseYear.recalculated_emissions_tco2e.toLocaleString()} tCO2e</>
            )}
          </Alert>
        )}

        <Box display="flex" gap={2} flexWrap="wrap" mb={2}>
          <TextField
            label="Base Year"
            type="number"
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            inputProps={{ min: 1990, max: new Date().getFullYear() }}
            size="small"
            sx={{ width: 140 }}
          />
          <TextField
            label="Base Year Emissions (tCO2e)"
            type="number"
            value={emissions}
            onChange={(e) => setEmissions(Number(e.target.value))}
            inputProps={{ min: 0, step: 0.1 }}
            size="small"
            sx={{ width: 240 }}
          />
          <TextField
            label="Recalculation Policy"
            value={policy}
            onChange={(e) => setPolicy(e.target.value)}
            size="small"
            multiline
            maxRows={2}
            sx={{ flex: 1, minWidth: 300 }}
          />
          <Button variant="contained" onClick={handleSave} sx={{ alignSelf: 'flex-start' }}>
            Save
          </Button>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
          <Typography variant="subtitle2">Recalculation Triggers</Typography>
          <Button size="small" variant="outlined" onClick={() => setShowRecalc(!showRecalc)}>
            {showRecalc ? 'Cancel' : 'New Recalculation'}
          </Button>
        </Box>

        {showRecalc && (
          <Box display="flex" gap={2} flexWrap="wrap" mb={2} p={2} sx={{ bgcolor: 'background.default', borderRadius: 1 }}>
            <TextField
              label="Trigger Type"
              value={triggerType}
              onChange={(e) => setTriggerType(e.target.value)}
              size="small"
              placeholder="e.g., acquisition, methodology_change"
              sx={{ width: 200 }}
            />
            <TextField
              label="Description"
              value={triggerDesc}
              onChange={(e) => setTriggerDesc(e.target.value)}
              size="small"
              sx={{ flex: 1, minWidth: 200 }}
            />
            <TextField
              label="New Emissions (tCO2e)"
              type="number"
              value={newEmissions}
              onChange={(e) => setNewEmissions(Number(e.target.value))}
              size="small"
              sx={{ width: 200 }}
            />
            <Button variant="contained" color="warning" onClick={handleRecalculate} disabled={!triggerType}>
              Recalculate
            </Button>
          </Box>
        )}

        {triggers.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No recalculation triggers recorded.
          </Typography>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 600 }}>Type</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Description</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 600 }}>Impact (tCO2e)</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 600 }}>Recalc?</TableCell>
                  <TableCell sx={{ fontWeight: 600 }}>Date</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {triggers.map((t) => (
                  <TableRow key={t.id} hover>
                    <TableCell>{t.trigger_type}</TableCell>
                    <TableCell>{t.description}</TableCell>
                    <TableCell align="right">
                      {t.impact_tco2e?.toLocaleString() ?? '-'}
                    </TableCell>
                    <TableCell align="center">
                      <Chip
                        label={t.requires_recalculation ? 'Yes' : 'No'}
                        size="small"
                        color={t.requires_recalculation ? 'warning' : 'default'}
                      />
                    </TableCell>
                    <TableCell>{new Date(t.triggered_at).toLocaleDateString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default BaseYearConfig;
