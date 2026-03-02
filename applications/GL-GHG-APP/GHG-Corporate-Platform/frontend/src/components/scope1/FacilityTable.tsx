/**
 * FacilityTable - Per-facility Scope 1 emissions data table
 *
 * MUI-based sortable, searchable table showing facility-level Scope 1
 * emissions. Columns: Facility, Country, Source Categories, Total tCO2e,
 * % of Scope 1, Data Quality. Supports row expansion for source detail.
 */

import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Paper,
  TextField,
  Box,
  Chip,
  Collapse,
  IconButton,
  Typography,
  InputAdornment,
} from '@mui/material';
import { Search, KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';
import type { FacilityEmissions } from '../../types';
import { formatNumber, getQualityColor } from '../../utils/formatters';

interface FacilityTableProps {
  facilities: FacilityEmissions[];
}

type SortKey = 'facility_name' | 'country' | 'total' | 'percent_of_scope' | 'data_quality';

const FacilityRow: React.FC<{ facility: FacilityEmissions }> = ({ facility }) => {
  const [open, setOpen] = useState(false);
  const categories = Object.entries(facility.source_categories);

  return (
    <>
      <TableRow hover>
        <TableCell padding="checkbox">
          <IconButton size="small" onClick={() => setOpen(!open)}>
            {open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
          </IconButton>
        </TableCell>
        <TableCell>{facility.facility_name}</TableCell>
        <TableCell>{facility.country}</TableCell>
        <TableCell>{categories.length} categories</TableCell>
        <TableCell align="right">{formatNumber(facility.total, 1)}</TableCell>
        <TableCell align="right">{facility.percent_of_scope.toFixed(1)}%</TableCell>
        <TableCell align="center">
          <Chip
            label={`${facility.data_quality.toFixed(0)}%`}
            size="small"
            color={getQualityColor(facility.data_quality)}
          />
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={7}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ py: 1, px: 2, mb: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Source Category Breakdown
              </Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Category</TableCell>
                    <TableCell align="right">Emissions (tCO2e)</TableCell>
                    <TableCell align="right">% of Facility</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {categories
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value]) => (
                      <TableRow key={name}>
                        <TableCell>{name.replace(/_/g, ' ')}</TableCell>
                        <TableCell align="right">{formatNumber(value, 1)}</TableCell>
                        <TableCell align="right">
                          {facility.total > 0 ? ((value / facility.total) * 100).toFixed(1) : 0}%
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
};

const FacilityTable: React.FC<FacilityTableProps> = ({ facilities }) => {
  const [search, setSearch] = useState('');
  const [orderBy, setOrderBy] = useState<SortKey>('total');
  const [order, setOrder] = useState<'asc' | 'desc'>('desc');

  const handleSort = (key: SortKey) => {
    setOrder(orderBy === key && order === 'asc' ? 'desc' : 'asc');
    setOrderBy(key);
  };

  const filtered = facilities.filter(
    (f) =>
      f.facility_name.toLowerCase().includes(search.toLowerCase()) ||
      f.country.toLowerCase().includes(search.toLowerCase())
  );

  const sorted = [...filtered].sort((a, b) => {
    const aVal = a[orderBy as keyof FacilityEmissions] ?? '';
    const bVal = b[orderBy as keyof FacilityEmissions] ?? '';
    const cmp = typeof aVal === 'number' ? aVal - (bVal as number) : String(aVal).localeCompare(String(bVal));
    return order === 'asc' ? cmp : -cmp;
  });

  return (
    <Paper>
      <Box sx={{ p: 2, pb: 1 }}>
        <TextField
          size="small"
          placeholder="Search facilities..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search fontSize="small" />
              </InputAdornment>
            ),
          }}
          sx={{ minWidth: 280 }}
        />
      </Box>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox" />
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'facility_name'}
                  direction={orderBy === 'facility_name' ? order : 'asc'}
                  onClick={() => handleSort('facility_name')}
                >
                  Facility
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'country'}
                  direction={orderBy === 'country' ? order : 'asc'}
                  onClick={() => handleSort('country')}
                >
                  Country
                </TableSortLabel>
              </TableCell>
              <TableCell>Source Categories</TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'total'}
                  direction={orderBy === 'total' ? order : 'asc'}
                  onClick={() => handleSort('total')}
                >
                  Total (tCO2e)
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'percent_of_scope'}
                  direction={orderBy === 'percent_of_scope' ? order : 'asc'}
                  onClick={() => handleSort('percent_of_scope')}
                >
                  % of Scope 1
                </TableSortLabel>
              </TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={orderBy === 'data_quality'}
                  direction={orderBy === 'data_quality' ? order : 'asc'}
                  onClick={() => handleSort('data_quality')}
                >
                  Data Quality
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sorted.map((f) => (
              <FacilityRow key={f.facility_id} facility={f} />
            ))}
            {sorted.length === 0 && (
              <TableRow>
                <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                  No facilities found
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default FacilityTable;
