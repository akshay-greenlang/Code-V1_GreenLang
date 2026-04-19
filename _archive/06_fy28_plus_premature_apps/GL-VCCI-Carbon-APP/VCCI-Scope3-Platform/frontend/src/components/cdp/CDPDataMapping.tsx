/**
 * CDPDataMapping - Visual data source mapping for CDP questionnaire
 *
 * Displays a filterable, searchable table showing how each CDP question
 * maps to a data source (ERP, Calculated, Manual, Unmapped) with
 * confidence indicators and summary statistics.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Paper,
  InputAdornment,
  SelectChangeEvent,
} from '@mui/material';
import {
  Search,
  Storage,
  Calculate,
  Edit as EditIcon,
  LinkOff,
} from '@mui/icons-material';
import type { CDPDataMappingItem } from '../../store/slices/cdpSlice';

// =============================================================================
// Props Interface
// =============================================================================

interface CDPDataMappingProps {
  mappings: CDPDataMappingItem[];
  onUpdateMapping?: (id: string, source: CDPDataMappingItem['dataSource']) => void;
}

// =============================================================================
// Constants
// =============================================================================

type DataSourceType = CDPDataMappingItem['dataSource'];

const DATA_SOURCE_CONFIG: Record<
  DataSourceType,
  { label: string; color: 'primary' | 'success' | 'warning' | 'error'; icon: React.ReactElement }
> = {
  erp: { label: 'ERP', color: 'primary', icon: <Storage fontSize="small" /> },
  calculated: {
    label: 'Calculated',
    color: 'success',
    icon: <Calculate fontSize="small" />,
  },
  manual: {
    label: 'Manual',
    color: 'warning',
    icon: <EditIcon fontSize="small" />,
  },
  unmapped: {
    label: 'Unmapped',
    color: 'error',
    icon: <LinkOff fontSize="small" />,
  },
};

const CONFIDENCE_CONFIG: Record<
  string,
  { color: 'success' | 'warning' | 'error' }
> = {
  high: { color: 'success' },
  medium: { color: 'warning' },
  low: { color: 'error' },
};

type SortableField = 'questionNumber' | 'sectionName' | 'dataSource' | 'confidence';
type SortDirection = 'asc' | 'desc';

// =============================================================================
// Main Component
// =============================================================================

const CDPDataMapping: React.FC<CDPDataMappingProps> = ({
  mappings,
  onUpdateMapping: _onUpdateMapping,
}) => {
  // State
  const [searchTerm, setSearchTerm] = useState('');
  const [sectionFilter, setSectionFilter] = useState<string>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');
  const [confidenceFilter, setConfidenceFilter] = useState<string>('all');
  const [sortField, setSortField] = useState<SortableField>('questionNumber');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');

  // Derived data
  const uniqueSections = useMemo(
    () => Array.from(new Set(mappings.map((m) => m.sectionName))).sort(),
    [mappings]
  );

  // Filter and sort
  const filteredMappings = useMemo(() => {
    let filtered = [...mappings];

    // Search filter
    if (searchTerm) {
      const lower = searchTerm.toLowerCase();
      filtered = filtered.filter(
        (m) =>
          m.questionText.toLowerCase().includes(lower) ||
          m.questionNumber.toLowerCase().includes(lower) ||
          m.displayValue.toLowerCase().includes(lower)
      );
    }

    // Section filter
    if (sectionFilter !== 'all') {
      filtered = filtered.filter((m) => m.sectionName === sectionFilter);
    }

    // Source filter
    if (sourceFilter !== 'all') {
      filtered = filtered.filter((m) => m.dataSource === sourceFilter);
    }

    // Confidence filter
    if (confidenceFilter !== 'all') {
      filtered = filtered.filter((m) => m.confidence === confidenceFilter);
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      const aVal = a[sortField];
      const bVal = b[sortField];

      if (typeof aVal === 'string' && typeof bVal === 'string') {
        comparison = aVal.localeCompare(bVal);
      }

      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [mappings, searchTerm, sectionFilter, sourceFilter, confidenceFilter, sortField, sortDirection]);

  // Statistics
  const stats = useMemo(() => {
    const total = mappings.length;
    const mapped = mappings.filter((m) => m.dataSource !== 'unmapped').length;
    const unmapped = total - mapped;
    const autoFilled = mappings.filter(
      (m) => m.dataSource === 'erp' || m.dataSource === 'calculated'
    ).length;
    const autoFilledPercent = total > 0 ? ((autoFilled / total) * 100).toFixed(1) : '0.0';

    const bySource: Record<string, number> = { erp: 0, calculated: 0, manual: 0, unmapped: 0 };
    mappings.forEach((m) => {
      bySource[m.dataSource] = (bySource[m.dataSource] || 0) + 1;
    });

    return { total, mapped, unmapped, autoFilled, autoFilledPercent, bySource };
  }, [mappings]);

  // Handlers
  const handleSort = useCallback(
    (field: SortableField) => {
      if (sortField === field) {
        setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortField(field);
        setSortDirection('asc');
      }
    },
    [sortField]
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearchTerm(e.target.value);
    },
    []
  );

  const handleSectionFilterChange = useCallback((e: SelectChangeEvent<string>) => {
    setSectionFilter(e.target.value);
  }, []);

  const handleSourceFilterChange = useCallback((e: SelectChangeEvent<string>) => {
    setSourceFilter(e.target.value);
  }, []);

  const handleConfidenceFilterChange = useCallback((e: SelectChangeEvent<string>) => {
    setConfidenceFilter(e.target.value);
  }, []);

  return (
    <Card variant="outlined">
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Data Source Mapping
        </Typography>

        {/* Stats row */}
        <Box
          sx={{
            display: 'flex',
            gap: 2,
            mb: 2,
            flexWrap: 'wrap',
          }}
        >
          <Chip
            label={`Total: ${stats.total}`}
            variant="outlined"
            size="small"
          />
          <Chip
            label={`Mapped: ${stats.mapped}`}
            color="success"
            variant="outlined"
            size="small"
          />
          <Chip
            label={`Unmapped: ${stats.unmapped}`}
            color={stats.unmapped > 0 ? 'error' : 'default'}
            variant="outlined"
            size="small"
          />
          <Chip
            label={`Auto-filled: ${stats.autoFilledPercent}%`}
            color="primary"
            variant="outlined"
            size="small"
          />
          {Object.entries(stats.bySource).map(([source, count]) => {
            const config = DATA_SOURCE_CONFIG[source as DataSourceType];
            return config && count > 0 ? (
              <Chip
                key={source}
                icon={config.icon}
                label={`${config.label}: ${count}`}
                color={config.color}
                variant="outlined"
                size="small"
              />
            ) : null;
          })}
        </Box>

        {/* Filters */}
        <Box
          sx={{
            display: 'flex',
            gap: 2,
            mb: 2,
            flexWrap: 'wrap',
          }}
        >
          <TextField
            label="Search questions"
            value={searchTerm}
            onChange={handleSearchChange}
            size="small"
            sx={{ minWidth: 200, flexGrow: 1 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <Search />
                </InputAdornment>
              ),
            }}
          />

          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Section</InputLabel>
            <Select
              value={sectionFilter}
              label="Section"
              onChange={handleSectionFilterChange}
            >
              <MenuItem value="all">All Sections</MenuItem>
              {uniqueSections.map((section) => (
                <MenuItem key={section} value={section}>
                  {section}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel>Source</InputLabel>
            <Select
              value={sourceFilter}
              label="Source"
              onChange={handleSourceFilterChange}
            >
              <MenuItem value="all">All Sources</MenuItem>
              {Object.entries(DATA_SOURCE_CONFIG).map(([key, config]) => (
                <MenuItem key={key} value={key}>
                  {config.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel>Confidence</InputLabel>
            <Select
              value={confidenceFilter}
              label="Confidence"
              onChange={handleConfidenceFilterChange}
            >
              <MenuItem value="all">All Levels</MenuItem>
              <MenuItem value="high">High</MenuItem>
              <MenuItem value="medium">Medium</MenuItem>
              <MenuItem value="low">Low</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Data table */}
        <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 500 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontWeight: 'bold', width: 100 }}>
                  <TableSortLabel
                    active={sortField === 'questionNumber'}
                    direction={sortField === 'questionNumber' ? sortDirection : 'asc'}
                    onClick={() => handleSort('questionNumber')}
                  >
                    Question ID
                  </TableSortLabel>
                </TableCell>
                <TableCell sx={{ fontWeight: 'bold' }}>Question Text</TableCell>
                <TableCell sx={{ fontWeight: 'bold', width: 110 }}>
                  <TableSortLabel
                    active={sortField === 'sectionName'}
                    direction={sortField === 'sectionName' ? sortDirection : 'asc'}
                    onClick={() => handleSort('sectionName')}
                  >
                    Section
                  </TableSortLabel>
                </TableCell>
                <TableCell sx={{ fontWeight: 'bold', width: 130 }}>
                  <TableSortLabel
                    active={sortField === 'dataSource'}
                    direction={sortField === 'dataSource' ? sortDirection : 'asc'}
                    onClick={() => handleSort('dataSource')}
                  >
                    Data Source
                  </TableSortLabel>
                </TableCell>
                <TableCell sx={{ fontWeight: 'bold', width: 160 }}>Value</TableCell>
                <TableCell sx={{ fontWeight: 'bold', width: 100 }}>
                  <TableSortLabel
                    active={sortField === 'confidence'}
                    direction={sortField === 'confidence' ? sortDirection : 'asc'}
                    onClick={() => handleSort('confidence')}
                  >
                    Confidence
                  </TableSortLabel>
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredMappings.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} align="center" sx={{ py: 3 }}>
                    <Typography variant="body2" color="text.secondary">
                      No mappings match the current filters.
                    </Typography>
                  </TableCell>
                </TableRow>
              ) : (
                filteredMappings.map((mapping) => {
                  const sourceConfig = DATA_SOURCE_CONFIG[mapping.dataSource];
                  const confidenceConfig = CONFIDENCE_CONFIG[mapping.confidence];

                  return (
                    <TableRow
                      key={mapping.id}
                      hover
                      sx={{
                        backgroundColor:
                          mapping.dataSource === 'unmapped'
                            ? 'rgba(211, 47, 47, 0.04)'
                            : 'inherit',
                      }}
                    >
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace" fontSize="0.8rem">
                          {mapping.questionNumber}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{
                            maxWidth: 300,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={mapping.questionText}
                        >
                          {mapping.questionText}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {mapping.sectionName}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={sourceConfig.icon}
                          label={sourceConfig.label}
                          color={sourceConfig.color}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.75rem' }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          sx={{
                            maxWidth: 140,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={mapping.displayValue}
                        >
                          {mapping.displayValue || '--'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {mapping.dataSource !== 'unmapped' && confidenceConfig && (
                          <Chip
                            label={mapping.confidence}
                            size="small"
                            color={confidenceConfig.color}
                            sx={{
                              fontSize: '0.7rem',
                              textTransform: 'capitalize',
                              height: 22,
                            }}
                          />
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </TableContainer>

        {/* Results count */}
        <Box sx={{ mt: 1, display: 'flex', justifyContent: 'flex-end' }}>
          <Typography variant="caption" color="text.secondary">
            Showing {filteredMappings.length} of {mappings.length} mappings
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default CDPDataMapping;
