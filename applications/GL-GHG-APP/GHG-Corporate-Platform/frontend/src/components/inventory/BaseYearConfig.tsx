/**
 * BaseYearConfig - Base year configuration and recalculation panel
 *
 * Displays the current base year, significance threshold slider,
 * recalculation history timeline, and lock/unlock controls.
 */

import React from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Slider,
  Chip,
  Divider,
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot,
  Alert,
} from '@mui/material';
import { Lock, LockOpen, Refresh, CheckCircle, Warning } from '@mui/icons-material';
import type { BaseYear, Recalculation } from '../../types';
import { formatNumber, formatDate } from '../../utils/formatters';

interface BaseYearConfigProps {
  baseYear: BaseYear;
  onRecalculate: () => void;
  onLock: (locked: boolean) => void;
  onThresholdChange?: (threshold: number) => void;
}

const BaseYearConfig: React.FC<BaseYearConfigProps> = ({
  baseYear,
  onRecalculate,
  onLock,
  onThresholdChange,
}) => {
  const recalculations = baseYear.recalculation_history || [];

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Base Year Configuration</Typography>
          <Chip
            icon={baseYear.locked ? <Lock fontSize="small" /> : <LockOpen fontSize="small" />}
            label={baseYear.locked ? 'Locked' : 'Unlocked'}
            color={baseYear.locked ? 'default' : 'warning'}
            variant="outlined"
          />
        </Box>

        {/* Base year overview */}
        <Box sx={{ display: 'flex', gap: 4, mb: 3 }}>
          <Box>
            <Typography variant="body2" color="text.secondary">Base Year</Typography>
            <Typography variant="h4" sx={{ fontWeight: 700 }}>{baseYear.year}</Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">Total Emissions</Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {formatNumber(baseYear.total_emissions_tco2e)} tCO2e
            </Typography>
          </Box>
        </Box>

        {/* Scope breakdown */}
        <Box sx={{ display: 'flex', gap: 3, mb: 3, flexWrap: 'wrap' }}>
          <Box>
            <Typography variant="caption" color="text.secondary">Scope 1</Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatNumber(baseYear.scope1_emissions_tco2e)} tCO2e
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Scope 2 (Location)</Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatNumber(baseYear.scope2_location_emissions_tco2e)} tCO2e
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Scope 2 (Market)</Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatNumber(baseYear.scope2_market_emissions_tco2e)} tCO2e
            </Typography>
          </Box>
          <Box>
            <Typography variant="caption" color="text.secondary">Scope 3</Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatNumber(baseYear.scope3_emissions_tco2e)} tCO2e
            </Typography>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Significance threshold */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Significance Threshold for Recalculation
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            Structural changes exceeding this threshold trigger a base year recalculation.
          </Typography>
          <Box sx={{ px: 2 }}>
            <Slider
              value={baseYear.significance_threshold_percent}
              min={1}
              max={10}
              step={0.5}
              marks={[
                { value: 1, label: '1%' },
                { value: 5, label: '5%' },
                { value: 10, label: '10%' },
              ]}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => `${v}%`}
              disabled={baseYear.locked}
              onChange={(_, value) => onThresholdChange?.(value as number)}
              sx={{ maxWidth: 400 }}
            />
          </Box>
        </Box>

        {/* Recalculation policy */}
        {baseYear.recalculation_policy && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2">
              <strong>Recalculation Policy:</strong> {baseYear.recalculation_policy}
            </Typography>
          </Alert>
        )}

        {/* Actions */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={onRecalculate}
            disabled={baseYear.locked}
          >
            Trigger Recalculation
          </Button>
          <Button
            variant="outlined"
            startIcon={baseYear.locked ? <LockOpen /> : <Lock />}
            onClick={() => onLock(!baseYear.locked)}
            color={baseYear.locked ? 'warning' : 'default'}
          >
            {baseYear.locked ? 'Unlock Base Year' : 'Lock Base Year'}
          </Button>
        </Box>

        {/* Recalculation history */}
        {recalculations.length > 0 && (
          <Box>
            <Divider sx={{ mb: 2 }} />
            <Typography variant="subtitle2" gutterBottom>
              Recalculation History
            </Typography>
            {recalculations.map((recalc: Recalculation) => (
              <Box
                key={recalc.id}
                sx={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 2,
                  py: 1.5,
                  borderBottom: '1px solid',
                  borderColor: 'divider',
                }}
              >
                <CheckCircle fontSize="small" color="success" sx={{ mt: 0.25 }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {recalc.trigger_reason}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatDate(recalc.recalculation_date)} | Approved by {recalc.approved_by}
                  </Typography>
                  <Typography variant="body2">
                    {formatNumber(recalc.original_emissions_tco2e)} {' -> '}
                    {formatNumber(recalc.recalculated_emissions_tco2e)} tCO2e
                    ({recalc.change_percent > 0 ? '+' : ''}{recalc.change_percent.toFixed(1)}%)
                  </Typography>
                </Box>
              </Box>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default BaseYearConfig;
