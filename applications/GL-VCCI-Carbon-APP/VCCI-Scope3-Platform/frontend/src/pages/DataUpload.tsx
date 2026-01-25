import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { CloudUpload, Refresh } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { uploadFile, pollUploadStatus, clearUploadStatus, fetchTransactions } from '../store/slices/transactionsSlice';
import { formatNumber, formatRelativeTime } from '../utils/formatters';

const DataUpload: React.FC = () => {
  const dispatch = useAppDispatch();
  const { uploadStatus, uploading, error } = useAppSelector((state) => state.transactions);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileFormat, setFileFormat] = useState<string>('csv');

  // Poll upload status
  useEffect(() => {
    if (uploadStatus && uploadStatus.status === 'processing') {
      const interval = setInterval(() => {
        dispatch(pollUploadStatus(uploadStatus.jobId));
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(interval);
    }
  }, [uploadStatus, dispatch]);

  // Refresh transactions when upload completes
  useEffect(() => {
    if (uploadStatus && uploadStatus.status === 'completed') {
      dispatch(fetchTransactions());
    }
  }, [uploadStatus, dispatch]);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);

      // Auto-detect format from extension
      const fileName = event.target.files[0].name.toLowerCase();
      if (fileName.endsWith('.csv')) setFileFormat('csv');
      else if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) setFileFormat('excel');
      else if (fileName.endsWith('.json')) setFileFormat('json');
      else if (fileName.endsWith('.xml')) setFileFormat('xml');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    await dispatch(uploadFile({ file: selectedFile, format: fileFormat }));
  };

  const handleReset = () => {
    setSelectedFile(null);
    setFileFormat('csv');
    dispatch(clearUploadStatus());
  };

  const getProgressPercentage = (): number => {
    if (!uploadStatus) return 0;
    if (uploadStatus.totalRecords === 0) return 0;
    return (uploadStatus.processedRecords / uploadStatus.totalRecords) * 100;
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Upload
      </Typography>

      <Grid container spacing={3}>
        {/* Upload Form */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Upload Transaction Data
            </Typography>

            <Box sx={{ mt: 2 }}>
              <input
                accept=".csv,.xlsx,.xls,.json,.xml"
                style={{ display: 'none' }}
                id="file-upload"
                type="file"
                onChange={handleFileChange}
              />
              <label htmlFor="file-upload">
                <Button
                  variant="outlined"
                  component="span"
                  startIcon={<CloudUpload />}
                  fullWidth
                  sx={{ mb: 2 }}
                >
                  Choose File
                </Button>
              </label>

              {selectedFile && (
                <Alert severity="info" sx={{ mb: 2 }}>
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(2)} KB)
                </Alert>
              )}

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>File Format</InputLabel>
                <Select
                  value={fileFormat}
                  label="File Format"
                  onChange={(e) => setFileFormat(e.target.value)}
                >
                  <MenuItem value="csv">CSV</MenuItem>
                  <MenuItem value="excel">Excel (XLSX/XLS)</MenuItem>
                  <MenuItem value="json">JSON</MenuItem>
                  <MenuItem value="xml">XML</MenuItem>
                </Select>
              </FormControl>

              <Box sx={{ display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading || (uploadStatus?.status === 'processing')}
                  fullWidth
                >
                  {uploading ? 'Uploading...' : 'Upload & Process'}
                </Button>
                <Button variant="outlined" onClick={handleReset}>
                  Reset
                </Button>
              </Box>
            </Box>

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Paper>

          {/* Instructions */}
          <Paper sx={{ p: 3, mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              File Format Requirements
            </Typography>
            <Typography variant="body2" paragraph>
              Your file should contain the following columns:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemText
                  primary="transaction_id"
                  secondary="Unique identifier for the transaction"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="date"
                  secondary="Transaction date (YYYY-MM-DD)"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="supplier_name"
                  secondary="Supplier name"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="product_name"
                  secondary="Product or service name"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="quantity"
                  secondary="Quantity purchased"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="spend_usd"
                  secondary="Spend amount in USD"
                />
              </ListItem>
              <ListItem>
                <ListItemText
                  primary="ghg_category"
                  secondary="GHG Protocol Category (1-15)"
                />
              </ListItem>
            </List>
          </Paper>
        </Grid>

        {/* Upload Status */}
        <Grid item xs={12} md={6}>
          {uploadStatus && (
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Upload Status</Typography>
                <Chip
                  label={uploadStatus.status.toUpperCase()}
                  color={
                    uploadStatus.status === 'completed'
                      ? 'success'
                      : uploadStatus.status === 'failed'
                      ? 'error'
                      : 'info'
                  }
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  Progress: {uploadStatus.processedRecords} / {uploadStatus.totalRecords} records
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={getProgressPercentage()}
                  sx={{ height: 10, borderRadius: 5 }}
                />
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="body2" color="textSecondary">
                    Total Records
                  </Typography>
                  <Typography variant="h6">{formatNumber(uploadStatus.totalRecords)}</Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2" color="textSecondary">
                    Processed
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {formatNumber(uploadStatus.processedRecords)}
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2" color="textSecondary">
                    Failed
                  </Typography>
                  <Typography variant="h6" color="error.main">
                    {formatNumber(uploadStatus.failedRecords)}
                  </Typography>
                </Grid>
              </Grid>

              {uploadStatus.errors && uploadStatus.errors.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="subtitle2" color="error" gutterBottom>
                    Errors ({uploadStatus.errors.length})
                  </Typography>
                  <List dense sx={{ maxHeight: 200, overflow: 'auto' }}>
                    {uploadStatus.errors.slice(0, 10).map((error, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={`Row ${error.row}: ${error.field}`}
                          secondary={error.message}
                        />
                      </ListItem>
                    ))}
                  </List>
                  {uploadStatus.errors.length > 10 && (
                    <Typography variant="caption" color="textSecondary">
                      ... and {uploadStatus.errors.length - 10} more errors
                    </Typography>
                  )}
                </Box>
              )}

              {uploadStatus.status === 'completed' && (
                <Alert severity="success" sx={{ mt: 2 }}>
                  Upload completed successfully! Data is now available in the dashboard.
                </Alert>
              )}
            </Paper>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default DataUpload;
