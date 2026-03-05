/**
 * BoundaryConfig - Organizational and operational boundary configuration
 *
 * Two sections:
 * 1. Organizational Boundary: consolidation approach selection + entity selection
 * 2. Operational Boundary: 6 ISO category inclusion toggles with significance
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Button,
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Switch,
  TextField,
  Chip,
  Divider,
} from '@mui/material';
import { Save } from '@mui/icons-material';
import {
  ConsolidationApproach,
  ISOCategory,
  ISO_CATEGORY_SHORT_NAMES,
  SignificanceLevel,
} from '../../types';
import type {
  Entity,
  CategoryInclusion,
  OrganizationalBoundary,
  OperationalBoundary,
  SetOrganizationalBoundaryRequest,
  SetOperationalBoundaryRequest,
} from '../../types';
import { getStatusColor } from '../../utils/formatters';

interface BoundaryConfigProps {
  entities: Entity[];
  orgBoundary: OrganizationalBoundary | null;
  opBoundary: OperationalBoundary | null;
  onSaveOrgBoundary: (data: SetOrganizationalBoundaryRequest) => void;
  onSaveOpBoundary: (data: SetOperationalBoundaryRequest) => void;
  loading?: boolean;
}

const CONSOLIDATION_OPTIONS: { value: ConsolidationApproach; label: string; description: string }[] = [
  {
    value: ConsolidationApproach.OPERATIONAL_CONTROL,
    label: 'Operational Control',
    description: '100% of emissions from operations over which the org has operational control',
  },
  {
    value: ConsolidationApproach.FINANCIAL_CONTROL,
    label: 'Financial Control',
    description: '100% of emissions from operations over which the org has financial control',
  },
  {
    value: ConsolidationApproach.EQUITY_SHARE,
    label: 'Equity Share',
    description: 'Emissions proportional to ownership percentage in each operation',
  },
];

const ALL_CATEGORIES = Object.values(ISOCategory);

const BoundaryConfig: React.FC<BoundaryConfigProps> = ({
  entities,
  orgBoundary,
  opBoundary,
  onSaveOrgBoundary,
  onSaveOpBoundary,
  loading = false,
}) => {
  // Organizational boundary state
  const [approach, setApproach] = useState<ConsolidationApproach>(
    ConsolidationApproach.OPERATIONAL_CONTROL,
  );
  const [selectedEntityIds, setSelectedEntityIds] = useState<string[]>([]);

  // Operational boundary state
  const [categories, setCategories] = useState<CategoryInclusion[]>(
    ALL_CATEGORIES.map((cat) => ({
      category: cat,
      included: cat === ISOCategory.CATEGORY_1_DIRECT || cat === ISOCategory.CATEGORY_2_ENERGY,
      significance: SignificanceLevel.UNDER_REVIEW,
      justification: null,
    })),
  );

  useEffect(() => {
    if (orgBoundary) {
      setApproach(orgBoundary.consolidation_approach);
      setSelectedEntityIds(orgBoundary.entity_ids);
    }
  }, [orgBoundary]);

  useEffect(() => {
    if (opBoundary?.categories?.length) {
      setCategories(opBoundary.categories);
    }
  }, [opBoundary]);

  const toggleEntity = (entityId: string) => {
    setSelectedEntityIds((prev) =>
      prev.includes(entityId)
        ? prev.filter((id) => id !== entityId)
        : [...prev, entityId],
    );
  };

  const toggleCategory = (idx: number) => {
    setCategories((prev) => {
      const updated = [...prev];
      updated[idx] = { ...updated[idx], included: !updated[idx].included };
      return updated;
    });
  };

  const updateCategorySignificance = (idx: number, significance: SignificanceLevel) => {
    setCategories((prev) => {
      const updated = [...prev];
      updated[idx] = { ...updated[idx], significance };
      return updated;
    });
  };

  const updateCategoryJustification = (idx: number, justification: string) => {
    setCategories((prev) => {
      const updated = [...prev];
      updated[idx] = { ...updated[idx], justification: justification || null };
      return updated;
    });
  };

  return (
    <Grid container spacing={3}>
      {/* Organizational Boundary */}
      <Grid item xs={12}>
        <Card>
          <CardHeader
            title="Organizational Boundary"
            subheader="Select the consolidation approach and which entities are included (ISO 14064-1 Clause 5.1)"
          />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Consolidation Approach</InputLabel>
                  <Select
                    value={approach}
                    label="Consolidation Approach"
                    onChange={(e) =>
                      setApproach(e.target.value as ConsolidationApproach)
                    }
                  >
                    {CONSOLIDATION_OPTIONS.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        <Box>
                          <Typography variant="body2" fontWeight={600}>
                            {opt.label}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {opt.description}
                          </Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Select Entities to Include
                </Typography>
                {entities.length === 0 ? (
                  <Typography variant="body2" color="text.secondary">
                    No entities defined. Add entities above first.
                  </Typography>
                ) : (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {entities.map((entity) => (
                      <FormControlLabel
                        key={entity.id}
                        control={
                          <Checkbox
                            checked={selectedEntityIds.includes(entity.id)}
                            onChange={() => toggleEntity(entity.id)}
                          />
                        }
                        label={`${entity.name} (${entity.ownership_pct}%)`}
                      />
                    ))}
                  </Box>
                )}
              </Grid>
              <Grid item xs={12}>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <Button
                    variant="contained"
                    startIcon={<Save />}
                    disabled={loading || selectedEntityIds.length === 0}
                    onClick={() =>
                      onSaveOrgBoundary({
                        consolidation_approach: approach,
                        entity_ids: selectedEntityIds,
                      })
                    }
                    sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
                  >
                    Save Organizational Boundary
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      {/* Operational Boundary */}
      <Grid item xs={12}>
        <Card>
          <CardHeader
            title="Operational Boundary"
            subheader="Configure which ISO 14064-1 categories are included and their significance (Clause 5.2)"
          />
          <CardContent>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Category</TableCell>
                    <TableCell align="center">Included</TableCell>
                    <TableCell>Significance</TableCell>
                    <TableCell>Justification</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {categories.map((cat, idx) => {
                    const isMandatory =
                      cat.category === ISOCategory.CATEGORY_1_DIRECT ||
                      cat.category === ISOCategory.CATEGORY_2_ENERGY;
                    return (
                      <TableRow key={cat.category}>
                        <TableCell>
                          <Typography variant="body2" fontWeight={500}>
                            {ISO_CATEGORY_SHORT_NAMES[cat.category]}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Switch
                            checked={cat.included}
                            onChange={() => toggleCategory(idx)}
                            disabled={isMandatory}
                            color="success"
                          />
                          {isMandatory && (
                            <Chip label="Mandatory" size="small" color="primary" sx={{ ml: 1 }} />
                          )}
                        </TableCell>
                        <TableCell>
                          {!isMandatory ? (
                            <FormControl size="small" sx={{ minWidth: 160 }}>
                              <Select
                                value={cat.significance}
                                onChange={(e) =>
                                  updateCategorySignificance(
                                    idx,
                                    e.target.value as SignificanceLevel,
                                  )
                                }
                              >
                                <MenuItem value={SignificanceLevel.SIGNIFICANT}>
                                  Significant
                                </MenuItem>
                                <MenuItem value={SignificanceLevel.NOT_SIGNIFICANT}>
                                  Not Significant
                                </MenuItem>
                                <MenuItem value={SignificanceLevel.UNDER_REVIEW}>
                                  Under Review
                                </MenuItem>
                              </Select>
                            </FormControl>
                          ) : (
                            <Chip
                              label="Significant"
                              color={getStatusColor('significant')}
                              size="small"
                            />
                          )}
                        </TableCell>
                        <TableCell>
                          {!isMandatory && (
                            <TextField
                              size="small"
                              placeholder="Justification for exclusion..."
                              value={cat.justification || ''}
                              onChange={(e) =>
                                updateCategoryJustification(idx, e.target.value)
                              }
                              fullWidth
                              disabled={cat.included}
                            />
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </TableContainer>
            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
              <Button
                variant="contained"
                startIcon={<Save />}
                disabled={loading}
                onClick={() => onSaveOpBoundary({ categories })}
                sx={{ bgcolor: '#1b5e20', '&:hover': { bgcolor: '#2e7d32' } }}
              >
                Save Operational Boundary
              </Button>
            </Box>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default BoundaryConfig;
