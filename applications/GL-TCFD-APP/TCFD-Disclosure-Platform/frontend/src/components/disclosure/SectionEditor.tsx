import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, TextField, Box, Button, Chip } from '@mui/material';
import { Save, Visibility } from '@mui/icons-material';
import type { DisclosureSection } from '../../types';

interface SectionEditorProps { section: DisclosureSection | null; onSave: (sectionId: string, content: string) => void; }

const SectionEditor: React.FC<SectionEditorProps> = ({ section, onSave }) => {
  const [content, setContent] = useState('');
  const [dirty, setDirty] = useState(false);

  useEffect(() => { if (section) { setContent(section.content); setDirty(false); } }, [section]);

  if (!section) return <Card><CardContent><Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>Select a disclosure section to edit</Typography></CardContent></Card>;

  return (
    <Card><CardContent>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Box>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>{section.title}</Typography>
          <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
            <Chip label={section.tcfd_code} size="small" color="primary" /><Chip label={section.pillar.replace(/_/g, ' ')} size="small" variant="outlined" />
            <Chip label={section.status.replace(/_/g, ' ')} size="small" color={section.status === 'final' ? 'success' : section.status === 'review' ? 'primary' : 'default'} />
          </Box>
        </Box>
        <Button variant="contained" startIcon={<Save />} onClick={() => { onSave(section.id, content); setDirty(false); }} disabled={!dirty}>Save</Button>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5, fontStyle: 'italic' }}>Recommendation: {section.tcfd_recommendation}</Typography>
      <TextField
        fullWidth multiline minRows={12} maxRows={24} value={content}
        onChange={(e) => { setContent(e.target.value); setDirty(true); }}
        placeholder="Enter disclosure content..."
        sx={{ '& .MuiOutlinedInput-root': { fontFamily: '"Georgia", serif', fontSize: 14, lineHeight: 1.8 } }}
      />
      {section.reviewer_notes && (
        <Box sx={{ mt: 2, p: 1.5, bgcolor: '#FFF3E0', borderRadius: 1 }}>
          <Typography variant="caption" sx={{ fontWeight: 600 }}>Reviewer Notes:</Typography>
          <Typography variant="body2">{section.reviewer_notes}</Typography>
        </Box>
      )}
    </CardContent></Card>
  );
};

export default SectionEditor;
