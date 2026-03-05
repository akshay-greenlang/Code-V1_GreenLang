/**
 * DataTable - Reusable sortable, filterable data table with pagination.
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
  Typography,
  InputAdornment,
} from '@mui/material';
import { Search } from '@mui/icons-material';

export interface Column<T> {
  id: string;
  label: string;
  accessor: (row: T) => React.ReactNode;
  sortAccessor?: (row: T) => string | number;
  align?: 'left' | 'center' | 'right';
  width?: number | string;
  filterable?: boolean;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyAccessor: (row: T) => string;
  title?: string;
  searchPlaceholder?: string;
  defaultSortColumn?: string;
  defaultSortDirection?: 'asc' | 'desc';
  onRowClick?: (row: T) => void;
  rowsPerPageOptions?: number[];
  emptyMessage?: string;
  toolbar?: React.ReactNode;
}

function DataTable<T>({
  columns,
  data,
  keyAccessor,
  title,
  searchPlaceholder = 'Search...',
  defaultSortColumn,
  defaultSortDirection = 'asc',
  onRowClick,
  rowsPerPageOptions = [10, 25, 50],
  emptyMessage = 'No data available',
  toolbar,
}: DataTableProps<T>) {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(rowsPerPageOptions[0] || 10);
  const [orderBy, setOrderBy] = useState<string>(defaultSortColumn || '');
  const [order, setOrder] = useState<'asc' | 'desc'>(defaultSortDirection);
  const [search, setSearch] = useState('');

  const handleSort = (columnId: string) => {
    const isAsc = orderBy === columnId && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(columnId);
  };

  const filteredAndSortedData = useMemo(() => {
    let result = [...data];

    if (search.trim()) {
      const lowerSearch = search.toLowerCase();
      result = result.filter((row) =>
        columns.some((col) => {
          const val = col.accessor(row);
          return typeof val === 'string' && val.toLowerCase().includes(lowerSearch);
        })
      );
    }

    if (orderBy) {
      const col = columns.find((c) => c.id === orderBy);
      if (col) {
        const accessor = col.sortAccessor || col.accessor;
        result.sort((a, b) => {
          const aVal = accessor(a);
          const bVal = accessor(b);
          if (typeof aVal === 'number' && typeof bVal === 'number') {
            return order === 'asc' ? aVal - bVal : bVal - aVal;
          }
          const aStr = String(aVal ?? '');
          const bStr = String(bVal ?? '');
          return order === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
        });
      }
    }

    return result;
  }, [data, search, orderBy, order, columns]);

  const paginatedData = filteredAndSortedData.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  return (
    <Paper sx={{ width: '100%' }}>
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        {title && (
          <Typography variant="h6" sx={{ fontWeight: 600, flexShrink: 0 }}>
            {title}
          </Typography>
        )}
        <TextField
          size="small"
          placeholder={searchPlaceholder}
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(0); }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start"><Search fontSize="small" /></InputAdornment>
            ),
          }}
          sx={{ minWidth: 240, flexGrow: 1 }}
        />
        {toolbar}
      </Box>

      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              {columns.map((col) => (
                <TableCell
                  key={col.id}
                  align={col.align || 'left'}
                  sx={{ width: col.width }}
                >
                  {col.sortAccessor !== undefined || typeof col.accessor === 'function' ? (
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
            {paginatedData.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length} align="center" sx={{ py: 4 }}>
                  <Typography variant="body2" color="text.secondary">
                    {emptyMessage}
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              paginatedData.map((row) => (
                <TableRow
                  key={keyAccessor(row)}
                  hover
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  sx={{ cursor: onRowClick ? 'pointer' : 'default' }}
                >
                  {columns.map((col) => (
                    <TableCell key={col.id} align={col.align || 'left'}>
                      {col.accessor(row)}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={filteredAndSortedData.length}
        page={page}
        rowsPerPage={rowsPerPage}
        rowsPerPageOptions={rowsPerPageOptions}
        onPageChange={(_, newPage) => setPage(newPage)}
        onRowsPerPageChange={(e) => { setRowsPerPage(parseInt(e.target.value, 10)); setPage(0); }}
      />
    </Paper>
  );
}

export default DataTable;
