/**
 * SatelliteOverlay - Satellite assessment results for a specific plot.
 *
 * Shows side-by-side baseline vs current imagery (simulated as color fields),
 * NDVI change indicator, forest cover percentage, deforestation detection
 * status, and a timeline of satellite assessments.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  Stack,
  Divider,
  Paper,
  ToggleButton,
  ToggleButtonGroup,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
} from '@mui/material';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import ForestIcon from '@mui/icons-material/Forest';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import TimelineIcon from '@mui/icons-material/Timeline';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SatelliteAssessment {
  id: string;
  plot_id: string;
  assessment_date: string;
  satellite_source: string;
  ndvi_baseline: number;
  ndvi_current: number;
  ndvi_change: number;
  forest_cover_baseline_pct: number;
  forest_cover_current_pct: number;
  tree_loss_pct: number;
  deforestation_detected: boolean;
  confidence_score: number;
  classification: 'no_change' | 'degradation' | 'deforestation';
  reference_date: string;
  imagery_resolution_m: number;
  notes: string;
}

interface SatelliteOverlayProps {
  plotId: string;
  assessments: SatelliteAssessment[];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function classificationColor(c: string): 'success' | 'warning' | 'error' {
  switch (c) {
    case 'no_change': return 'success';
    case 'degradation': return 'warning';
    case 'deforestation': return 'error';
    default: return 'warning';
  }
}

function classificationLabel(c: string): string {
  switch (c) {
    case 'no_change': return 'No Change';
    case 'degradation': return 'Degradation';
    case 'deforestation': return 'Deforestation Detected';
    default: return c;
  }
}

function ndviToColor(ndvi: number): string {
  if (ndvi >= 0.6) return '#2e7d32';
  if (ndvi >= 0.4) return '#4caf50';
  if (ndvi >= 0.2) return '#8bc34a';
  if (ndvi >= 0.1) return '#cddc39';
  if (ndvi >= 0) return '#ffeb3b';
  return '#795548';
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Simulated satellite imagery panel using NDVI-derived coloring. */
function ImageryPanel({
  label,
  ndvi,
  forestCover,
  date,
}: {
  label: string;
  ndvi: number;
  forestCover: number;
  date: string;
}) {
  const baseColor = ndviToColor(ndvi);
  // Create a gradient to simulate land-cover variation
  const darkerShade = ndviToColor(ndvi - 0.1);
  const lighterShade = ndviToColor(ndvi + 0.05);

  return (
    <Paper
      variant="outlined"
      sx={{
        p: 0,
        overflow: 'hidden',
        borderRadius: 2,
        flex: 1,
      }}
    >
      <Box sx={{ px: 1.5, py: 0.75, backgroundColor: 'grey.100' }}>
        <Typography variant="subtitle2">{label}</Typography>
        <Typography variant="caption" color="text.secondary">
          {formatDate(date)}
        </Typography>
      </Box>
      {/* Simulated imagery */}
      <Box
        sx={{
          height: 160,
          background: `
            radial-gradient(ellipse at 30% 40%, ${lighterShade} 0%, transparent 60%),
            radial-gradient(ellipse at 70% 60%, ${darkerShade} 0%, transparent 50%),
            linear-gradient(135deg, ${baseColor} 0%, ${darkerShade} 100%)
          `,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
      >
        <SatelliteAltIcon
          sx={{ fontSize: 40, color: 'rgba(255,255,255,0.25)' }}
        />
        <Box
          sx={{
            position: 'absolute',
            bottom: 4,
            right: 8,
            backgroundColor: 'rgba(0,0,0,0.6)',
            color: '#fff',
            px: 1,
            py: 0.25,
            borderRadius: 1,
            fontSize: 11,
          }}
        >
          NDVI: {ndvi.toFixed(3)}
        </Box>
      </Box>
      <Box sx={{ px: 1.5, py: 1 }}>
        <Stack direction="row" alignItems="center" spacing={1}>
          <ForestIcon fontSize="small" color="success" />
          <Typography variant="body2">
            Forest Cover: <strong>{forestCover.toFixed(1)}%</strong>
          </Typography>
        </Stack>
      </Box>
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

const SatelliteOverlay: React.FC<SatelliteOverlayProps> = ({
  plotId,
  assessments,
}) => {
  const sorted = useMemo(
    () =>
      [...assessments].sort(
        (a, b) =>
          new Date(b.assessment_date).getTime() -
          new Date(a.assessment_date).getTime()
      ),
    [assessments]
  );

  const [selectedIdx, setSelectedIdx] = useState(0);
  const current = sorted[selectedIdx] ?? null;

  if (!current) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary" align="center">
            No satellite assessments available for plot {plotId}.
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const ndviChangePct = current.ndvi_change * 100;
  const ndviChangeColor = ndviChangePct >= 0 ? 'success' : ndviChangePct >= -10 ? 'warning' : 'error';

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6">Satellite Assessment</Typography>
          <Chip
            icon={
              current.classification === 'no_change' ? (
                <CheckCircleIcon />
              ) : current.classification === 'deforestation' ? (
                <CancelIcon />
              ) : (
                <WarningAmberIcon />
              )
            }
            label={classificationLabel(current.classification)}
            color={classificationColor(current.classification)}
            variant="outlined"
          />
        </Stack>

        {/* Side-by-side imagery */}
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} mb={2}>
          <ImageryPanel
            label="Baseline (Pre-2020)"
            ndvi={current.ndvi_baseline}
            forestCover={current.forest_cover_baseline_pct}
            date={current.reference_date}
          />
          <ImageryPanel
            label="Current Assessment"
            ndvi={current.ndvi_current}
            forestCover={current.forest_cover_current_pct}
            date={current.assessment_date}
          />
        </Stack>

        {/* NDVI Change Bar */}
        <Paper variant="outlined" sx={{ p: 1.5, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            NDVI Change
          </Typography>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Typography variant="body2" sx={{ minWidth: 60 }}>
              {current.ndvi_baseline.toFixed(3)}
            </Typography>
            <Box sx={{ flex: 1, position: 'relative' }}>
              <LinearProgress
                variant="determinate"
                value={Math.max(0, Math.min(100, 50 + ndviChangePct * 5))}
                color={ndviChangeColor as 'success' | 'warning' | 'error'}
                sx={{ height: 10, borderRadius: 1 }}
              />
              <Box
                sx={{
                  position: 'absolute',
                  top: -2,
                  left: '50%',
                  width: 2,
                  height: 14,
                  backgroundColor: '#333',
                }}
              />
            </Box>
            <Typography variant="body2" sx={{ minWidth: 60, textAlign: 'right' }}>
              {current.ndvi_current.toFixed(3)}
            </Typography>
          </Stack>
          <Typography
            variant="caption"
            color={`${ndviChangeColor}.main`}
            sx={{ mt: 0.5, display: 'block', textAlign: 'center' }}
          >
            Change: {ndviChangePct >= 0 ? '+' : ''}{ndviChangePct.toFixed(1)}%
          </Typography>
        </Paper>

        {/* Key Metrics */}
        <Grid container spacing={2} mb={2}>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Forest Cover
              </Typography>
              <Typography variant="h6">
                {current.forest_cover_current_pct.toFixed(1)}%
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Tree Loss
              </Typography>
              <Typography
                variant="h6"
                color={current.tree_loss_pct > 5 ? 'error.main' : 'text.primary'}
              >
                {current.tree_loss_pct.toFixed(1)}%
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Confidence
              </Typography>
              <Typography variant="h6">
                {(current.confidence_score * 100).toFixed(0)}%
              </Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <Typography variant="caption" color="text.secondary">
                Resolution
              </Typography>
              <Typography variant="h6">
                {current.imagery_resolution_m}m
              </Typography>
            </Paper>
          </Grid>
        </Grid>

        <Divider sx={{ my: 1.5 }} />

        {/* Assessment Timeline */}
        <Stack direction="row" alignItems="center" spacing={1} mb={1}>
          <TimelineIcon fontSize="small" color="action" />
          <Typography variant="subtitle2">Assessment History</Typography>
        </Stack>

        <List dense disablePadding>
          {sorted.map((assessment, idx) => (
            <ListItem
              key={assessment.id}
              button
              selected={idx === selectedIdx}
              onClick={() => setSelectedIdx(idx)}
              sx={{
                borderRadius: 1,
                mb: 0.5,
                border: idx === selectedIdx ? '1px solid' : '1px solid transparent',
                borderColor: idx === selectedIdx ? 'primary.main' : 'transparent',
              }}
            >
              <ListItemIcon sx={{ minWidth: 36 }}>
                <CalendarTodayIcon fontSize="small" />
              </ListItemIcon>
              <ListItemText
                primary={formatDate(assessment.assessment_date)}
                secondary={`${assessment.satellite_source} | Confidence: ${(assessment.confidence_score * 100).toFixed(0)}%`}
              />
              <Chip
                size="small"
                label={classificationLabel(assessment.classification)}
                color={classificationColor(assessment.classification)}
                variant={idx === selectedIdx ? 'filled' : 'outlined'}
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default SatelliteOverlay;
