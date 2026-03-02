/**
 * DataTable Component
 *
 * Generic MUI Table with sorting, pagination, search, row click,
 * and selectable rows. Accepts any data type via generics.
 */

import React, { useState, useMemo } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Paper,
  TextField,
  Box,
  Checkbox,
  Typography,
  InputAdornment,
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';

type OrderDirection = 'asc' | 'desc';

export interface ColumnDef<T> {
  id: string;
  label: string;
  accessor: keyof T | ((row: T) => React.ReactNode);
  sortable?: boolean;
  align?: 'left' | 'center' | 'right';
  width?: string | number;
  render?: (value: unknown, row: T) => React.ReactNode;
}

interface DataTableProps<T> {
  columns: ColumnDef<T>[];
  data: T[];
  totalCount: number;
  page: number;
  rowsPerPage: number;
  onPageChange: (page: number) => void;
  onRowsPerPageChange: (rowsPerPage: number) => void;
  onRowClick?: (row: T) => void;
  onSortChange?: (sortBy: string, sortOrder: OrderDirection) => void;
  onSearchChange?: (search: string) => void;
  selectable?: boolean;
  selectedIds?: string[];
  onSelectionChange?: (ids: string[]) => void;
  getRowId: (row: T) => string;
  searchPlaceholder?: string;
  loading?: boolean;
  emptyMessage?: string;
}

function DataTable<T>({
  columns,
  data,
  totalCount,
  page,
  rowsPerPage,
  onPageChange,
  onRowsPerPageChange,
  onRowClick,
  onSortChange,
  onSearchChange,
  selectable = false,
  selectedIds = [],
  onSelectionChange,
  getRowId,
  searchPlaceholder = 'Search...',
  loading = false,
  emptyMessage = 'No data available',
}: DataTableProps<T>): React.ReactElement {
  const [orderBy, setOrderBy] = useState<string>('');
  const [order, setOrder] = useState<OrderDirection>('asc');
  const [search, setSearch] = useState('');

  const handleSort = (columnId: string) => {
    const isAsc = orderBy === columnId && order === 'asc';
    const newOrder: OrderDirection = isAsc ? 'desc' : 'asc';
    setOrderBy(columnId);
    setOrder(newOrder);
    onSortChange?.(columnId, newOrder);
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = event.target.value;
    setSearch(value);
    onSearchChange?.(value);
  };

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const allIds = data.map(getRowId);
      onSelectionChange?.(allIds);
    } else {
      onSelectionChange?.([]);
    }
  };

  const handleSelectRow = (id: string) => {
    const newSelection = selectedIds.includes(id)
      ? selectedIds.filter((s) => s !== id)
      : [...selectedIds, id];
    onSelectionChange?.(newSelection);
  };

  const getCellValue = (row: T, column: ColumnDef<T>): React.ReactNode => {
    const rawValue =
      typeof column.accessor === 'function'
        ? column.accessor(row)
        : row[column.accessor];

    if (column.render) {
      return column.render(rawValue, row);
    }

    if (rawValue === null || rawValue === undefined) return '-';
    return rawValue as React.ReactNode;
  };

  const allSelected =
    data.length > 0 && selectedIds.length === data.length;
  const someSelected =
    selectedIds.length > 0 && selectedIds.length < data.length;

  return (
    <Paper sx={{ width: '100%' }}>
      {/* Search bar */}
      {onSearchChange && (
        <Box sx={{ p: 2, pb: 1 }}>
          <TextField
            fullWidth
            size="small"
            placeholder={searchPlaceholder}
            value={search}
            onChange={handleSearchChange}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon color="action" />
                </InputAdornment>
              ),
            }}
          />
        </Box>
      )}

      {/* Table */}
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    indeterminate={someSelected}
                    checked={allSelected}
                    onChange={handleSelectAll}
                    inputProps={{ 'aria-label': 'Select all rows' }}
                  />
                </TableCell>
              )}
              {columns.map((col) => (
                <TableCell
                  key={col.id}
                  align={col.align || 'left'}
                  sx={{
                    fontWeight: 600,
                    width: col.width,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {col.sortable !== false ? (
                    <TableSortLabel
                      active={orderBy === col.id}
                      direction={orderBy === col.id ? order : 'asc'}
                      onClick={() => handleSort(col.id)}
                    >
                      {col.label}
                    </TableSortLabel>
                  ) : (
                    col.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>

          <TableBody>
            {data.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length + (selectable ? 1 : 0)}
                  align="center"
                  sx={{ py: 4 }}
                >
                  <Typography variant="body2" color="text.secondary">
                    {loading ? 'Loading...' : emptyMessage}
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              data.map((row) => {
                const rowId = getRowId(row);
                const isSelected = selectedIds.includes(rowId);
                return (
                  <TableRow
                    key={rowId}
                    hover
                    selected={isSelected}
                    onClick={() => onRowClick?.(row)}
                    sx={{
                      cursor: onRowClick ? 'pointer' : 'default',
                    }}
                  >
                    {selectable && (
                      <TableCell padding="checkbox">
                        <Checkbox
                          checked={isSelected}
                          onChange={() => handleSelectRow(rowId)}
                          onClick={(e) => e.stopPropagation()}
                          inputProps={{ 'aria-label': `Select row ${rowId}` }}
                        />
                      </TableCell>
                    )}
                    {columns.map((col) => (
                      <TableCell key={col.id} align={col.align || 'left'}>
                        {getCellValue(row, col)}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Pagination */}
      <TablePagination
        component="div"
        count={totalCount}
        page={page}
        rowsPerPage={rowsPerPage}
        onPageChange={(_, newPage) => onPageChange(newPage)}
        onRowsPerPageChange={(e) =>
          onRowsPerPageChange(parseInt(e.target.value, 10))
        }
        rowsPerPageOptions={[10, 25, 50, 100]}
      />
    </Paper>
  );
}

export default DataTable;
