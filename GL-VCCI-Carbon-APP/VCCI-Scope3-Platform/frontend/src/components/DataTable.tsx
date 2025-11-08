import React from 'react';
import { DataGrid, GridColDef, GridPaginationModel, GridRowParams } from '@mui/x-data-grid';
import { Paper } from '@mui/material';

interface DataTableProps {
  rows: any[];
  columns: GridColDef[];
  loading?: boolean;
  pageSize?: number;
  page?: number;
  totalRows?: number;
  onPageChange?: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
  onRowClick?: (params: GridRowParams) => void;
  height?: string | number;
}

const DataTable: React.FC<DataTableProps> = ({
  rows,
  columns,
  loading = false,
  pageSize = 25,
  page = 0,
  totalRows,
  onPageChange,
  onPageSizeChange,
  onRowClick,
  height = 600,
}) => {
  const handlePaginationChange = (model: GridPaginationModel) => {
    if (onPageChange && model.page !== page) {
      onPageChange(model.page);
    }
    if (onPageSizeChange && model.pageSize !== pageSize) {
      onPageSizeChange(model.pageSize);
    }
  };

  return (
    <Paper sx={{ height, width: '100%' }}>
      <DataGrid
        rows={rows}
        columns={columns}
        loading={loading}
        pageSizeOptions={[10, 25, 50, 100]}
        paginationModel={{ page, pageSize }}
        onPaginationModelChange={handlePaginationChange}
        rowCount={totalRows || rows.length}
        paginationMode={totalRows ? 'server' : 'client'}
        onRowClick={onRowClick}
        disableRowSelectionOnClick
        sx={{
          '& .MuiDataGrid-cell:hover': {
            cursor: onRowClick ? 'pointer' : 'default',
          },
        }}
      />
    </Paper>
  );
};

export default DataTable;
