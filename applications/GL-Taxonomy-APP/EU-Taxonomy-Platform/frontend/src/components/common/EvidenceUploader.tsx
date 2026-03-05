/**
 * EvidenceUploader - File upload component for evidence documents.
 */

import React, { useState, useRef } from 'react';
import {
  Box, Button, Typography, List, ListItem, ListItemText, ListItemIcon,
  IconButton, Chip, Paper,
} from '@mui/material';
import { CloudUpload, InsertDriveFile, Delete, CheckCircle } from '@mui/icons-material';

interface UploadedFile {
  name: string;
  size: number;
  type: string;
  status: 'pending' | 'uploaded' | 'verified';
}

interface EvidenceUploaderProps {
  onUpload?: (files: File[]) => void;
  accept?: string;
  maxFiles?: number;
  maxSizeMB?: number;
  label?: string;
}

const EvidenceUploader: React.FC<EvidenceUploaderProps> = ({
  onUpload,
  accept = '.pdf,.doc,.docx,.xlsx,.csv,.jpg,.png',
  maxFiles = 10,
  maxSizeMB = 25,
  label = 'Upload Evidence Documents',
}) => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (fileList: FileList) => {
    const newFiles: UploadedFile[] = [];
    for (let i = 0; i < Math.min(fileList.length, maxFiles - files.length); i++) {
      const file = fileList[i];
      if (file.size <= maxSizeMB * 1024 * 1024) {
        newFiles.push({ name: file.name, size: file.size, type: file.type, status: 'pending' });
      }
    }
    setFiles(prev => [...prev, ...newFiles]);
    onUpload?.(Array.from(fileList));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
        {label}
      </Typography>

      <Paper
        variant="outlined"
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        sx={{
          p: 3,
          textAlign: 'center',
          border: dragActive ? '2px dashed #1B5E20' : '2px dashed #BDBDBD',
          backgroundColor: dragActive ? '#E8F5E9' : '#FAFAFA',
          cursor: 'pointer',
          transition: 'all 0.2s',
        }}
        onClick={() => inputRef.current?.click()}
      >
        <CloudUpload sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
        <Typography variant="body2" color="text.secondary">
          Drag and drop files here, or click to browse
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Max {maxSizeMB}MB per file. Accepted: {accept}
        </Typography>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          multiple
          hidden
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
      </Paper>

      {files.length > 0 && (
        <List dense sx={{ mt: 1 }}>
          {files.map((file, idx) => (
            <ListItem
              key={`${file.name}-${idx}`}
              secondaryAction={
                <IconButton edge="end" onClick={() => removeFile(idx)} size="small">
                  <Delete fontSize="small" />
                </IconButton>
              }
            >
              <ListItemIcon sx={{ minWidth: 36 }}>
                {file.status === 'verified' ? (
                  <CheckCircle color="success" fontSize="small" />
                ) : (
                  <InsertDriveFile color="action" fontSize="small" />
                )}
              </ListItemIcon>
              <ListItemText
                primary={file.name}
                secondary={formatSize(file.size)}
                primaryTypographyProps={{ fontSize: '0.85rem' }}
              />
              <Chip
                label={file.status}
                size="small"
                color={file.status === 'verified' ? 'success' : file.status === 'uploaded' ? 'primary' : 'default'}
                sx={{ mr: 1 }}
              />
            </ListItem>
          ))}
        </List>
      )}
    </Box>
  );
};

export default EvidenceUploader;
