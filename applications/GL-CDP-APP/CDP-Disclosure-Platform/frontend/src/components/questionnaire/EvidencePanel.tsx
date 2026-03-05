/**
 * EvidencePanel - Evidence attachment panel
 *
 * Allows uploading and managing evidence documents linked to a
 * CDP response. Supports drag-and-drop file upload.
 */

import React, { useCallback, useRef } from 'react';
import { Box, Typography, List, ListItem, ListItemText, ListItemIcon, IconButton, Button, Chip } from '@mui/material';
import { AttachFile, Delete, CloudUpload, InsertDriveFile } from '@mui/icons-material';
import type { Evidence } from '../../types';
import { useAppDispatch } from '../../store/hooks';
import { uploadEvidence, deleteEvidence } from '../../store/slices/responseSlice';
import { formatDate } from '../../utils/formatters';

interface EvidencePanelProps {
  responseId: string;
  evidence: Evidence[];
}

const EvidencePanel: React.FC<EvidencePanelProps> = ({ responseId, evidence }) => {
  const dispatch = useAppDispatch();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach((file) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('description', file.name);
      dispatch(uploadEvidence({ responseId, formData }));
    });
  }, [dispatch, responseId]);

  const handleDelete = useCallback((evidenceId: string) => {
    dispatch(deleteEvidence({ responseId, evidenceId }));
  }, [dispatch, responseId]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <Box>
      <Typography variant="subtitle2" fontWeight={600} gutterBottom>
        Evidence ({evidence.length})
      </Typography>

      <Box
        sx={{
          border: '2px dashed #e0e0e0',
          borderRadius: 2,
          p: 2,
          textAlign: 'center',
          cursor: 'pointer',
          '&:hover': { backgroundColor: '#f5f7f5', borderColor: '#1b5e20' },
          transition: 'all 0.2s ease',
          mb: 2,
        }}
        onClick={() => fileInputRef.current?.click()}
        onDrop={(e) => {
          e.preventDefault();
          handleFileSelect(e.dataTransfer.files);
        }}
        onDragOver={(e) => e.preventDefault()}
      >
        <CloudUpload sx={{ color: '#9e9e9e', fontSize: 32 }} />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          Drop files here or click to upload
        </Typography>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          hidden
          onChange={(e) => handleFileSelect(e.target.files)}
        />
      </Box>

      {evidence.length > 0 && (
        <List dense disablePadding>
          {evidence.map((ev) => (
            <ListItem
              key={ev.id}
              secondaryAction={
                <IconButton edge="end" size="small" onClick={() => handleDelete(ev.id)}>
                  <Delete fontSize="small" />
                </IconButton>
              }
              sx={{ px: 1, borderRadius: 1, mb: 0.5, '&:hover': { backgroundColor: '#f5f5f5' } }}
            >
              <ListItemIcon sx={{ minWidth: 36 }}>
                <InsertDriveFile fontSize="small" sx={{ color: '#1565c0' }} />
              </ListItemIcon>
              <ListItemText
                primary={ev.file_name}
                secondary={`${formatFileSize(ev.file_size_bytes)} -- uploaded ${formatDate(ev.uploaded_at)}`}
                primaryTypographyProps={{ variant: 'body2', fontWeight: 500 }}
                secondaryTypographyProps={{ variant: 'caption' }}
              />
            </ListItem>
          ))}
        </List>
      )}
    </Box>
  );
};

export default EvidencePanel;
