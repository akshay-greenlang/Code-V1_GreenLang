/**
 * Data table widget with sorting, filtering, and pagination.
 */

import React, { useMemo, useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  ColumnDef,
  SortingState,
  ColumnFiltersState
} from '@tanstack/react-table';
import { Metric } from '../MetricService';

interface TableWidgetProps {
  title: string;
  data: Metric[];
  config?: {
    columns?: string[];
    pageSize?: number;
    sortable?: boolean;
    filterable?: boolean;
  };
  onRemove?: () => void;
}

const TableWidget: React.FC<TableWidgetProps> = ({ title, data, config, onRemove }) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [pagination, setPagination] = useState({
    pageIndex: 0,
    pageSize: config?.pageSize || 10
  });

  const columns = useMemo<ColumnDef<any>[]>(() => {
    if (config?.columns) {
      return config.columns.map(col => ({
        accessorKey: col,
        header: col.charAt(0).toUpperCase() + col.slice(1),
        cell: info => info.getValue()
      }));
    }

    // Default columns
    return [
      {
        accessorKey: 'timestamp',
        header: 'Timestamp',
        cell: info => new Date(info.getValue() as string).toLocaleString()
      },
      {
        accessorKey: 'name',
        header: 'Name',
        cell: info => info.getValue()
      },
      {
        accessorKey: 'value',
        header: 'Value',
        cell: info => {
          const value = info.getValue();
          return typeof value === 'number' ? value.toFixed(2) : value;
        }
      }
    ];
  }, [config?.columns]);

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      columnFilters,
      pagination
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel()
  });

  const exportCSV = () => {
    const headers = columns.map(col => (col as any).header).join(',');
    const rows = data.map(row =>
      columns.map(col => row[(col as any).accessorKey]).join(',')
    ).join('\n');

    const csv = `${headers}\n${rows}`;
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `table-${Date.now()}.csv`;
    link.click();
  };

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={exportCSV} style={{ padding: '4px 8px', cursor: 'pointer' }}>Export CSV</button>
          {onRemove && (
            <button onClick={onRemove} style={{ background: 'transparent', border: 'none', cursor: 'pointer', fontSize: '18px', color: '#f44336' }}>
              ×
            </button>
          )}
        </div>
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th
                    key={header.id}
                    style={{
                      padding: '8px',
                      borderBottom: '2px solid #ddd',
                      textAlign: 'left',
                      cursor: config?.sortable !== false ? 'pointer' : 'default'
                    }}
                    onClick={config?.sortable !== false ? header.column.getToggleSortingHandler() : undefined}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {{
                      asc: ' ↑',
                      desc: ' ↓'
                    }[header.column.getIsSorted() as string] ?? null}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map(row => (
              <tr key={row.id} style={{ borderBottom: '1px solid #eee' }}>
                {row.getVisibleCells().map(cell => (
                  <td key={cell.id} style={{ padding: '8px' }}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px', borderTop: '1px solid #ddd' }}>
        <div>
          Page {pagination.pageIndex + 1} of {table.getPageCount()}
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => table.setPageIndex(0)}
            disabled={!table.getCanPreviousPage()}
            style={{ padding: '4px 8px', cursor: 'pointer' }}
          >
            {'<<'}
          </button>
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            style={{ padding: '4px 8px', cursor: 'pointer' }}
          >
            {'<'}
          </button>
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            style={{ padding: '4px 8px', cursor: 'pointer' }}
          >
            {'>'}
          </button>
          <button
            onClick={() => table.setPageIndex(table.getPageCount() - 1)}
            disabled={!table.getCanNextPage()}
            style={{ padding: '4px 8px', cursor: 'pointer' }}
          >
            {'>>'}
          </button>
        </div>
      </div>

      {data.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>No data available</div>
      )}
    </div>
  );
};

export default TableWidget;
