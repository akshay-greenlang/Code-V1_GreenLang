/**
 * CategoryOverview - Grid view of all 15 Scope 3 categories
 *
 * Displays 15 cards (3x5 grid) showing category number, name, emissions
 * total, percentage, materiality badge, and data quality tier. Supports
 * sorting by magnitude or category number, and filtering by materiality.
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Grid,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import type { Scope3CategoryBreakdown } from '../../types';
import { SCOPE3_CATEGORY_NAMES } from '../../types';
import StatusBadge from '../common/StatusBadge';
import { formatNumber, formatPercent } from '../../utils/formatters';

interface CategoryOverviewProps {
  categories: Scope3CategoryBreakdown[];
  onCategoryClick?: (category: Scope3CategoryBreakdown) => void;
}

type SortMode = 'magnitude' | 'number';
type FilterMode = 'all' | 'material' | 'immaterial';

const getMaterialityStatus = (cat: Scope3CategoryBreakdown): string => {
  if (cat.is_excluded) return 'not_calculated';
  return cat.is_material ? 'material' : 'immaterial';
};

const getBorderColor = (cat: Scope3CategoryBreakdown): string => {
  if (cat.is_excluded) return '#c62828';
  if (cat.is_material) return '#2e7d32';
  return '#9e9e9e';
};

const CategoryOverview: React.FC<CategoryOverviewProps> = ({ categories, onCategoryClick }) => {
  const [sortMode, setSortMode] = useState<SortMode>('magnitude');
  const [filterMode, setFilterMode] = useState<FilterMode>('all');

  const filteredAndSorted = useMemo(() => {
    let result = [...categories];

    if (filterMode === 'material') {
      result = result.filter((c) => c.is_material && !c.is_excluded);
    } else if (filterMode === 'immaterial') {
      result = result.filter((c) => !c.is_material && !c.is_excluded);
    }

    if (sortMode === 'magnitude') {
      result.sort((a, b) => b.emissions_tco2e - a.emissions_tco2e);
    } else {
      result.sort((a, b) => a.category_number - b.category_number);
    }

    return result;
  }, [categories, sortMode, filterMode]);

  const totalScope3 = categories.reduce((sum, c) => sum + c.emissions_tco2e, 0);
  const materialCount = categories.filter((c) => c.is_material).length;

  return (
    <Box>
      {/* Controls */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap', gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="body2" color="text.secondary">
            {materialCount} of {categories.length} categories material | Total: {formatNumber(totalScope3)} tCO2e
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Filter</InputLabel>
            <Select
              value={filterMode}
              label="Filter"
              onChange={(e) => setFilterMode(e.target.value as FilterMode)}
            >
              <MenuItem value="all">All</MenuItem>
              <MenuItem value="material">Material</MenuItem>
              <MenuItem value="immaterial">Immaterial</MenuItem>
            </Select>
          </FormControl>
          <ToggleButtonGroup
            value={sortMode}
            exclusive
            onChange={(_, v) => v && setSortMode(v)}
            size="small"
          >
            <ToggleButton value="magnitude">By Size</ToggleButton>
            <ToggleButton value="number">By Cat #</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {/* Category grid */}
      <Grid container spacing={2}>
        {filteredAndSorted.map((cat) => (
          <Grid item xs={12} sm={6} md={4} key={cat.category}>
            <Card
              sx={{
                height: '100%',
                borderLeft: `4px solid ${getBorderColor(cat)}`,
                opacity: cat.is_excluded ? 0.6 : 1,
              }}
            >
              <CardActionArea
                onClick={() => onCategoryClick?.(cat)}
                sx={{ height: '100%' }}
                disabled={cat.is_excluded}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                    <Chip
                      label={`Cat ${cat.category_number}`}
                      size="small"
                      variant="outlined"
                      sx={{ fontWeight: 600 }}
                    />
                    <StatusBadge status={getMaterialityStatus(cat)} />
                  </Box>

                  <Typography variant="subtitle2" sx={{ mb: 1, minHeight: 40 }}>
                    {cat.category_name || SCOPE3_CATEGORY_NAMES[cat.category]}
                  </Typography>

                  <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5 }}>
                    {formatNumber(cat.emissions_tco2e)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    tCO2e ({formatPercent(cat.percentage_of_total)})
                  </Typography>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <StatusBadge status={cat.data_quality_tier} />
                    <Typography variant="caption" color="text.secondary">
                      {cat.calculation_method?.replace(/_/g, ' ')}
                    </Typography>
                  </Box>
                </CardContent>
              </CardActionArea>
            </Card>
          </Grid>
        ))}
      </Grid>

      {filteredAndSorted.length === 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
          No categories match the current filter.
        </Typography>
      )}
    </Box>
  );
};

export default CategoryOverview;
