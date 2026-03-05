/**
 * DataTable - Reusable sortable, filterable, paginated table.
 */

import React, { useState, useMemo } from 'react';
import {
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  TableSortLabel, TablePagination, Paper, TextField, Box, InputAdornment,
} from '@mui/material';
import { Search } from '@mui/icons-material';

interface Column<T> {
  key: keyof T;
  label: string;
  align?: 'left' | 'center' | 'right';
  format?: (value: unknown, row: T) => React.ReactNode;
  sortable?: boolean;
  width?: number | string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyField: keyof T;
  title?: string;
  searchable?: boolean;
  searchFields?: (keyof T)[];
  defaultSortBy?: keyof T;
  defaultSortDir?: 'asc' | 'desc';
  rowsPerPageOptions?: number[];
  onRowClick?: (row: T) => void;
  dense?: boolean;
}

function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  keyField,
  searchable = true,
  searchFields,
  defaultSortBy,
  defaultSortDir = 'asc',
  rowsPerPageOptions = [10, 25, 50],
  onRowClick,
  dense = false,
}: DataTableProps<T>) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(rowsPerPageOptions[0] || 10);
  const [orderBy, setOrderBy] = useState<keyof T | undefined>(defaultSortBy);
  const [order, setOrder] = useState<'asc' | 'desc'>(defaultSortDir);
  const [filter, setFilter] = useState('');

  const filtered = useMemo(() => {
    if (!filter) return data;
    const lower = filter.toLowerCase();
    const fields = searchFields || columns.map(c => c.key);
    return data.filter(row =>
      fields.some(f => String(row[f] ?? '').toLowerCase().includes(lower))
    );
  }, [data, filter, searchFields, columns]);

  const sorted = useMemo(() => {
    if (!orderBy) return filtered;
    return [...filtered].sort((a, b) => {
      const aVal = a[orderBy];
      const bVal = b[orderBy];
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return order === 'asc' ? cmp : -cmp;
    });
  }, [filtered, orderBy, order]);

  const paged = sorted.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  const handleSort = (col: keyof T) => {
    const isAsc = orderBy === col && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(col);
  };

  return (
    <Paper sx={{ width: '100%' }}>
      {searchable && (
        <Box sx={{ p: 1.5 }}>
          <TextField
            size="small"
            placeholder="Search..."
            value={filter}
            onChange={(e) => { setFilter(e.target.value); setPage(0); }}
            InputProps={{
              startAdornment: <InputAdornment position="start"><Search fontSize="small" /></InputAdornment>,
            }}
            sx={{ minWidth: 280 }}
          />
        </Box>
      )}

      <TableContainer>
        <Table size={dense ? 'small' : 'medium'}>
          <TableHead>
            <TableRow>
              {columns.map(col => (
                <TableCell key={String(col.key)} align={col.align || 'left'} sx={{ width: col.width }}>
                  {col.sortable !== false ? (
                    <TableSortLabel
                      active={orderBy === col.key}
                      direction={orderBy === col.key ? order : 'asc'}
                      onClick={() => handleSort(col.key)}
                    >
                      {col.label}
                    </TableSortLabel>
                  ) : col.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {paged.map(row => (
              <TableRow
                key={String(row[keyField])}
                hover
                onClick={() => onRowClick?.(row)}
                sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
              >
                {columns.map(col => (
                  <TableCell key={String(col.key)} align={col.align || 'left'}>
                    {col.format ? col.format(row[col.key], row) : String(row[col.key] ?? '')}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        rowsPerPageOptions={rowsPerPageOptions}
        component="div"
        count={filtered.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={(_, p) => setPage(p)}
        onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value)); setPage(0); }}
      />
    </Paper>
  );
}

export default DataTable;
