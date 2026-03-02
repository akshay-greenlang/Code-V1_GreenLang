/**
 * EvidenceViewer - Side-by-side satellite evidence viewer.
 *
 * Shows "Before" (baseline 2020) and "After" (latest assessment) panels,
 * NDVI comparison bar, key metrics (forest cover %, tree loss %, date range),
 * confidence score, and evidence classification badge.
 */

import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Stack,
  Chip,
  LinearProgress,
  Paper,
  Divider,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import CancelIcon from '@mui/icons-material/Cancel';
import SatelliteAltIcon from '@mui/icons-material/SatelliteAlt';
import ForestIcon from '@mui/icons-material/Forest';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SatelliteEvidence {
  plot_id: string;
  plot_name: string;
  baseline_date: string;
  assessment_date: string;
  ndvi_baseline: number;
  ndvi_current: number;
  ndvi_change: number;
  forest_cover_baseline_pct: number;
  forest_cover_current_pct: number;
  tree_loss_pct: number;
  confidence_score: number;
  classification: 'no_change' | 'degradation' | 'deforestation';
  satellite_source: string;
  resolution_m: number;
}

interface EvidenceViewerProps {
  evidence: SatelliteEvidence;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function classificationMeta(c: string): {
  label: string;
  color: 'success' | 'warning' | 'error';
  icon: React.ReactNode;
} {
  switch (c) {
    case 'no_change':
      return { label: 'No Change', color: 'success', icon: <CheckCircleIcon /> };
    case 'degradation':
      return { label: 'Degradation', color: 'warning', icon: <WarningAmberIcon /> };
    case 'deforestation':
      return { label: 'Deforestation Detected', color: 'error', icon: <CancelIcon /> };
    default:
      return { label: c, color: 'warning', icon: <WarningAmberIcon /> };
  }
}

function ndviToGradient(ndvi: number): string {
  if (ndvi >= 0.6) return 'linear-gradient(135deg, #1b5e20, #2e7d32, #388e3c)';
  if (ndvi >= 0.4) return 'linear-gradient(135deg, #388e3c, #4caf50, #66bb6a)';
  if (ndvi >= 0.2) return 'linear-gradient(135deg, #66bb6a, #8bc34a, #aed581)';
  if (ndvi >= 0.1) return 'linear-gradient(135deg, #aed581, #cddc39, #dce775)';
  return 'linear-gradient(135deg, #cddc39, #ffeb3b, #fff176)';
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
  });
}

// ---------------------------------------------------------------------------
// Imagery Panel
// ---------------------------------------------------------------------------

function ImagePanel({
  label,
  date,
  ndvi,
  forestCover,
}: {
  label: string;
  date: string;
  ndvi: number;
  forestCover: number;
}) {
  return (
    <Paper variant="outlined" sx={{ overflow: 'hidden', borderRadius: 2, flex: 1 }}>
      <Box sx={{ px: 1.5, py: 0.75, backgroundColor: 'grey.50' }}>
        <Typography variant="subtitle2">{label}</Typography>
        <Typography variant="caption" color="text.secondary">
          {formatDate(date)}
        </Typography>
      </Box>
      <Box
        sx={{
          height: 140,
          background: ndviToGradient(ndvi),
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
      >
        <SatelliteAltIcon sx={{ fontSize: 36, color: 'rgba(255,255,255,0.2)' }} />
        <Box
          sx={{
            position: 'absolute',
            bottom: 4,
            right: 6,
            backgroundColor: 'rgba(0,0,0,0.55)',
            color: '#fff',
            px: 0.75,
            py: 0.25,
            borderRadius: 0.5,
            fontSize: 11,
          }}
        >
          NDVI: {ndvi.toFixed(3)}
        </Box>
      </Box>
      <Stack direction="row" alignItems="center" spacing={0.5} sx={{ px: 1.5, py: 0.75 }}>
        <ForestIcon fontSize="small" color="success" />
        <Typography variant="body2">
          Forest: <strong>{forestCover.toFixed(1)}%</strong>
        </Typography>
      </Stack>
    </Paper>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const EvidenceViewer: React.FC<EvidenceViewerProps> = ({ evidence }) => {
  const meta = classificationMeta(evidence.classification);
  const ndviChangePct = evidence.ndvi_change * 100;

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Stack direction="row" alignItems="center" justifyContent="space-between" mb={2}>
          <Box>
            <Typography variant="h6">Satellite Evidence</Typography>
            <Typography variant="body2" color="text.secondary">
              {evidence.plot_name} | {evidence.satellite_source} ({evidence.resolution_m}m)
            </Typography>
          </Box>
          <Chip
            icon={meta.icon as React.ReactElement}
            label={meta.label}
            color={meta.color}
            variant="outlined"
          />
        </Stack>

        {/* Side-by-side panels */}
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} mb={2}>
          <ImagePanel
            label="Before (Baseline 2020)"
            date={evidence.baseline_date}
            ndvi={evidence.ndvi_baseline}
            forestCover={evidence.forest_cover_baseline_pct}
          />
          <ImagePanel
            label="After (Latest Assessment)"
            date={evidence.assessment_date}
            ndvi={evidence.ndvi_current}
            forestCover={evidence.forest_cover_current_pct}
          />
        </Stack>

        {/* NDVI Comparison Bar */}
        <Paper variant="outlined" sx={{ p: 1.5, mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            NDVI Change
          </Typography>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="body2" sx={{ minWidth: 56 }}>
              {evidence.ndvi_baseline.toFixed(3)}
            </Typography>
            <Box sx={{ flex: 1, position: 'relative' }}>
              <LinearProgress
                variant="determinate"
                value={Math.max(0, Math.min(100, 50 + ndviChangePct * 5))}
                color={ndviChangePct >= 0 ? 'success' : ndviChangePct >= -10 ? 'warning' : 'error'}
                sx={{ height: 10, borderRadius: 1 }}
              />
              {/* Center marker */}
              <Box
                sx={{
                  position: 'absolute',
                  top: -1,
                  left: '50%',
                  width: 2,
                  height: 12,
                  backgroundColor: '#333',
                }}
              />
            </Box>
            <Typography variant="body2" sx={{ minWidth: 56, textAlign: 'right' }}>
              {evidence.ndvi_current.toFixed(3)}
            </Typography>
          </Stack>
          <Typography
            variant="caption"
            sx={{ display: 'block', textAlign: 'center', mt: 0.5 }}
            color={ndviChangePct >= 0 ? 'success.main' : 'error.main'}
          >
            {ndviChangePct >= 0 ? '+' : ''}
            {ndviChangePct.toFixed(1)}% change
          </Typography>
        </Paper>

        {/* Key Metrics */}
        <Grid container spacing={1.5} mb={1}>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <ForestIcon fontSize="small" color="success" />
              <Typography variant="h6">{evidence.forest_cover_current_pct.toFixed(1)}%</Typography>
              <Typography variant="caption" color="text.secondary">Forest Cover</Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <TrendingDownIcon
                fontSize="small"
                color={evidence.tree_loss_pct > 5 ? 'error' : 'action'}
              />
              <Typography
                variant="h6"
                color={evidence.tree_loss_pct > 5 ? 'error.main' : 'text.primary'}
              >
                {evidence.tree_loss_pct.toFixed(1)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">Tree Loss</Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <Typography variant="h6">
                {(evidence.confidence_score * 100).toFixed(0)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">Confidence</Typography>
            </Paper>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Paper variant="outlined" sx={{ p: 1, textAlign: 'center' }}>
              <Typography variant="body2" fontWeight={600}>
                {formatDate(evidence.baseline_date)}
              </Typography>
              <Typography variant="body2">to</Typography>
              <Typography variant="body2" fontWeight={600}>
                {formatDate(evidence.assessment_date)}
              </Typography>
              <Typography variant="caption" color="text.secondary">Date Range</Typography>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default EvidenceViewer;
