/**
 * DataTable - Reusable sortable, searchable data table
 *
 * Wraps MUI Table with built-in sorting, search filtering,
 * and pagination.  Generic over row type for type safety.
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
  InputAdornment,
} from '@mui/material';
import { Search } from '@mui/icons-material';

export interface Column<T> {
  id: string;
  label: string;
  align?: 'left' | 'center' | 'right';
  sortable?: boolean;
  render?: (row: T) => React.ReactNode;
  getValue?: (row: T) => string | number;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  searchable?: boolean;
  searchPlaceholder?: string;
  defaultSort?: string;
  defaultOrder?: 'asc' | 'desc';
  rowsPerPageOptions?: number[];
  onRowClick?: (row: T) => void;
  dense?: boolean;
  toolbar?: React.ReactNode;
}

function DataTable<T>({
  columns,
  rows,
  rowKey,
  searchable = true,
  searchPlaceholder = 'Search...',
  defaultSort,
  defaultOrder = 'asc',
  rowsPerPageOptions = [10, 25, 50],
  onRowClick,
  dense = false,
  toolbar,
}: DataTableProps<T>) {
  const [search, setSearch] = useState('');
  const [orderBy, setOrderBy] = useState(defaultSort || '');
  const [order, setOrder] = useState<'asc' | 'desc'>(defaultOrder);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(rowsPerPageOptions[0]);

  const filteredRows = useMemo(() => {
    if (!search.trim()) return rows;
    const q = search.toLowerCase();
    return rows.filter((row) =>
      columns.some((col) => {
        const val = col.getValue
          ? col.getValue(row)
          : (row as Record<string, unknown>)[col.id];
        return String(val ?? '').toLowerCase().includes(q);
      }),
    );
  }, [rows, search, columns]);

  const sortedRows = useMemo(() => {
    if (!orderBy) return filteredRows;
    const col = columns.find((c) => c.id === orderBy);
    return [...filteredRows].sort((a, b) => {
      const aVal = col?.getValue
        ? col.getValue(a)
        : (a as Record<string, unknown>)[orderBy];
      const bVal = col?.getValue
        ? col.getValue(b)
        : (b as Record<string, unknown>)[orderBy];
      const cmp = String(aVal ?? '').localeCompare(String(bVal ?? ''), undefined, {
        numeric: true,
      });
      return order === 'asc' ? cmp : -cmp;
    });
  }, [filteredRows, orderBy, order, columns]);

  const paginatedRows = sortedRows.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage,
  );

  const handleSort = (columnId: string) => {
    const isAsc = orderBy === columnId && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(columnId);
  };

  return (
    <Paper sx={{ width: '100%' }}>
      {(searchable || toolbar) && (
        <Box sx={{ p: 2, pb: 1, display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          {searchable && (
            <TextField
              size="small"
              placeholder={searchPlaceholder}
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(0);
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search fontSize="small" />
                  </InputAdornment>
                ),
              }}
              sx={{ minWidth: 280 }}
            />
          )}
          {toolbar}
        </Box>
      )}
      <TableContainer>
        <Table size={dense ? 'small' : 'medium'}>
          <TableHead>
            <TableRow>
              {columns.map((col) => (
                <TableCell key={col.id} align={col.align || 'left'}>
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
            {paginatedRows.map((row) => (
              <TableRow
                key={rowKey(row)}
                hover
                onClick={() => onRowClick?.(row)}
                sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
              >
                {columns.map((col) => (
                  <TableCell key={col.id} align={col.align || 'left'}>
                    {col.render
                      ? col.render(row)
                      : String(
                          (row as Record<string, unknown>)[col.id] ?? '',
                        )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
            {paginatedRows.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={columns.length}
                  align="center"
                  sx={{ py: 4 }}
                >
                  No data available
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={rowsPerPageOptions}
        component="div"
        count={filteredRows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={(_, newPage) => setPage(newPage)}
        onRowsPerPageChange={(e) => {
          setRowsPerPage(parseInt(e.target.value, 10));
          setPage(0);
        }}
      />
    </Paper>
  );
}

export default DataTable;
