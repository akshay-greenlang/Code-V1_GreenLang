/**
 * ComplianceScorecard - Multi-standard compliance dashboard
 *
 * Displays compliance coverage across 5 standards (GHG Protocol Scope 3,
 * ESRS E1, CDP Climate Change, IFRS S2, ISO 14083) with circular progress
 * gauges, requirement checklists, and gap analysis summary.
 */

import React, { useMemo, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActions,
  Grid,
  Button,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  Alert,
  AlertTitle,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
} from '@mui/material';
import {
  CheckCircle,
  Cancel,
  Download,
  Visibility,
  Close,
} from '@mui/icons-material';
import type {
  ComplianceStandard,
  ComplianceRequirement,
} from '../../store/slices/complianceSlice';

// =============================================================================
// Props Interface
// =============================================================================

interface ComplianceScorecardProps {
  standards: ComplianceStandard[];
  overallScore?: number;
  onExportReport?: () => void;
}

// =============================================================================
// Constants
// =============================================================================

// =============================================================================
// Helper Functions
// =============================================================================

function getScoreColor(percentage: number): string {
  if (percentage >= 80) return '#2e7d32';
  if (percentage >= 60) return '#ed6c02';
  if (percentage >= 40) return '#f57c00';
  return '#d32f2f';
}

function getScoreMUIColor(percentage: number): 'success' | 'warning' | 'error' {
  if (percentage >= 80) return 'success';
  if (percentage >= 50) return 'warning';
  return 'error';
}

// =============================================================================
// Circular Progress Gauge
// =============================================================================

interface ProgressGaugeProps {
  value: number;
  size?: number;
  label?: string;
  sublabel?: string;
}

const ProgressGauge: React.FC<ProgressGaugeProps> = ({
  value,
  size = 100,
  label,
  sublabel,
}) => {
  const color = getScoreColor(value);

  return (
    <Box sx={{ position: 'relative', display: 'inline-flex' }}>
      {/* Background circle */}
      <CircularProgress
        variant="determinate"
        value={100}
        size={size}
        thickness={4}
        sx={{ color: 'grey.200', position: 'absolute' }}
      />
      {/* Value circle */}
      <CircularProgress
        variant="determinate"
        value={value}
        size={size}
        thickness={4}
        sx={{ color }}
      />
      {/* Center text */}
      <Box
        sx={{
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          position: 'absolute',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography variant="h6" fontWeight="bold" sx={{ color, lineHeight: 1 }}>
          {label || `${value.toFixed(0)}%`}
        </Typography>
        {sublabel && (
          <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.2, mt: 0.25 }}>
            {sublabel}
          </Typography>
        )}
      </Box>
    </Box>
  );
};

// =============================================================================
// Standard Card Component
// =============================================================================

interface StandardCardProps {
  standard: ComplianceStandard;
}

const StandardCard: React.FC<StandardCardProps> = ({ standard }) => {
  const [detailsOpen, setDetailsOpen] = useState(false);
  const displayRequirements = standard.requirements.slice(0, 5);
  const remainingCount = Math.max(0, standard.requirements.length - 5);

  const metCount = standard.requirements.filter((r) => r.met).length;
  const criticalGaps = standard.requirements.filter(
    (r) => !r.met && r.priority === 'critical'
  ).length;

  return (
    <>
      <Card
        variant="outlined"
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          borderColor:
            standard.coveragePercentage >= 80
              ? 'success.light'
              : standard.coveragePercentage >= 50
              ? 'warning.light'
              : 'error.light',
          borderWidth: 1,
        }}
      >
        <CardContent sx={{ flexGrow: 1, pb: 1 }}>
          {/* Header */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              mb: 1.5,
            }}
          >
            <Box>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ lineHeight: 1.2 }}>
                {standard.shortName}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {standard.name}
              </Typography>
            </Box>
            <ProgressGauge
              value={standard.coveragePercentage}
              size={70}
              label={
                standard.predictedScore
                  ? standard.predictedScore
                  : `${standard.coveragePercentage.toFixed(0)}%`
              }
              sublabel={standard.predictedScore ? 'Predicted' : 'Coverage'}
            />
          </Box>

          {/* Stats */}
          <Box sx={{ display: 'flex', gap: 1, mb: 1.5, flexWrap: 'wrap' }}>
            <Chip
              label={`${standard.requirementsMet}/${standard.requirementsTotal} met`}
              size="small"
              color={getScoreMUIColor(
                (standard.requirementsMet / Math.max(standard.requirementsTotal, 1)) * 100
              )}
              variant="outlined"
              sx={{ fontSize: '0.7rem' }}
            />
            <Chip
              label={`${standard.completionPercentage.toFixed(0)}% complete`}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.7rem' }}
            />
            {criticalGaps > 0 && (
              <Chip
                label={`${criticalGaps} critical gap${criticalGaps > 1 ? 's' : ''}`}
                size="small"
                color="error"
                sx={{ fontSize: '0.7rem' }}
              />
            )}
          </Box>

          {/* Requirements checklist */}
          <List dense disablePadding>
            {displayRequirements.map((req) => (
              <ListItem key={req.id} disableGutters sx={{ py: 0.25 }}>
                <ListItemIcon sx={{ minWidth: 28 }}>
                  {req.met ? (
                    <CheckCircle sx={{ fontSize: 18, color: 'success.main' }} />
                  ) : (
                    <Cancel sx={{ fontSize: 18, color: 'error.main' }} />
                  )}
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Typography
                      variant="body2"
                      sx={{
                        textDecoration: req.met ? 'none' : 'none',
                        color: req.met ? 'text.primary' : 'text.primary',
                        fontSize: '0.8rem',
                      }}
                      noWrap
                      title={req.name}
                    >
                      {req.code}: {req.name}
                    </Typography>
                  }
                />
              </ListItem>
            ))}
          </List>
        </CardContent>

        <CardActions sx={{ px: 2, pb: 1.5, pt: 0 }}>
          <Button
            size="small"
            onClick={() => setDetailsOpen(true)}
            startIcon={<Visibility />}
          >
            View All{remainingCount > 0 ? ` (+${remainingCount})` : ''}
          </Button>
        </CardActions>
      </Card>

      {/* Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Box>
              <Typography variant="h6">{standard.name}</Typography>
              <Typography variant="body2" color="text.secondary">
                {metCount} of {standard.requirements.length} requirements met
              </Typography>
            </Box>
            <IconButton onClick={() => setDetailsOpen(false)} size="small">
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent dividers>
          <List dense>
            {standard.requirements.map((req, index) => (
              <React.Fragment key={req.id}>
                {index > 0 && <Divider />}
                <ListItem
                  sx={{
                    py: 1,
                    backgroundColor: !req.met ? 'rgba(211, 47, 47, 0.04)' : 'inherit',
                  }}
                >
                  <ListItemIcon sx={{ minWidth: 32 }}>
                    {req.met ? (
                      <CheckCircle color="success" fontSize="small" />
                    ) : (
                      <Cancel color="error" fontSize="small" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="body2" fontWeight="medium">
                          {req.code}: {req.name}
                        </Typography>
                        <Chip
                          label={req.priority}
                          size="small"
                          color={
                            req.priority === 'critical'
                              ? 'error'
                              : req.priority === 'high'
                              ? 'warning'
                              : req.priority === 'medium'
                              ? 'info'
                              : 'default'
                          }
                          sx={{ fontSize: '0.65rem', height: 18, textTransform: 'capitalize' }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          {req.description}
                        </Typography>
                        <br />
                        <Typography variant="caption" color="text.secondary">
                          Data points: {req.dataPointsFilled}/{req.dataPointsRequired}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

// =============================================================================
// Main Component
// =============================================================================

const ComplianceScorecard: React.FC<ComplianceScorecardProps> = ({
  standards,
  overallScore,
  onExportReport,
}) => {
  // Compute overall score if not provided
  const computedOverallScore = useMemo(() => {
    if (overallScore !== undefined) return overallScore;
    if (standards.length === 0) return 0;

    const totalWeight = standards.length;
    const weightedSum = standards.reduce(
      (sum, s) => sum + s.coveragePercentage,
      0
    );
    return weightedSum / totalWeight;
  }, [standards, overallScore]);

  // Gap analysis
  const gapAnalysis = useMemo(() => {
    const allGaps = standards.flatMap((s) =>
      s.requirements
        .filter((r) => !r.met)
        .map((r) => ({
          ...r,
          standardName: s.shortName,
        }))
    );

    const critical = allGaps.filter((g) => g.priority === 'critical').length;
    const high = allGaps.filter((g) => g.priority === 'high').length;
    const medium = allGaps.filter((g) => g.priority === 'medium').length;
    const low = allGaps.filter((g) => g.priority === 'low').length;

    return { total: allGaps.length, critical, high, medium, low };
  }, [standards]);

  return (
    <Box>
      {/* Overall score and gap summary */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
          <ProgressGauge
            value={computedOverallScore}
            size={90}
            sublabel="Overall"
          />
          <Box>
            <Typography variant="h6">Compliance Overview</Typography>
            <Typography variant="body2" color="text.secondary">
              Weighted average across {standards.length} reporting standards
            </Typography>
          </Box>
        </Box>

        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
          {gapAnalysis.critical > 0 && (
            <Chip
              label={`${gapAnalysis.critical} Critical`}
              color="error"
              size="small"
            />
          )}
          {gapAnalysis.high > 0 && (
            <Chip
              label={`${gapAnalysis.high} High`}
              color="warning"
              size="small"
            />
          )}
          {gapAnalysis.medium > 0 && (
            <Chip
              label={`${gapAnalysis.medium} Medium`}
              color="info"
              size="small"
              variant="outlined"
            />
          )}
          {gapAnalysis.low > 0 && (
            <Chip
              label={`${gapAnalysis.low} Low`}
              size="small"
              variant="outlined"
            />
          )}
          <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
            {gapAnalysis.total} total gap{gapAnalysis.total !== 1 ? 's' : ''}
          </Typography>

          {onExportReport && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<Download />}
              onClick={onExportReport}
              sx={{ ml: 1 }}
            >
              Export Report
            </Button>
          )}
        </Box>
      </Box>

      {/* Critical gap alert */}
      {gapAnalysis.critical > 0 && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <AlertTitle>Critical Compliance Gaps</AlertTitle>
          {gapAnalysis.critical} critical requirement{gapAnalysis.critical > 1 ? 's are' : ' is'} not
          met. These must be addressed before submission deadlines.
        </Alert>
      )}

      {/* Standard cards grid */}
      <Grid container spacing={2}>
        {standards.map((standard) => (
          <Grid item xs={12} sm={6} md={4} key={standard.id}>
            <StandardCard standard={standard} />
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default ComplianceScorecard;
