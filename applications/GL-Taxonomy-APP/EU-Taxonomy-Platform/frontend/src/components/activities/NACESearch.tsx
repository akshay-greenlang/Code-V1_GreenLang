/**
 * NACESearch - NACE code search input with autocomplete results.
 */

import React, { useState } from 'react';
import { TextField, Box, List, ListItem, ListItemText, Paper, Typography, Chip, InputAdornment } from '@mui/material';
import { Search } from '@mui/icons-material';

const DEMO_RESULTS = [
  { nace_code: 'D35.11', description: 'Production of electricity', eligible: true, objectives: ['CCM', 'CCA'] },
  { nace_code: 'C23.1', description: 'Manufacture of glass', eligible: true, objectives: ['CCM', 'CE'] },
  { nace_code: 'F41.1', description: 'Development of building projects', eligible: true, objectives: ['CCM', 'CCA'] },
  { nace_code: 'H49.1', description: 'Passenger rail transport', eligible: true, objectives: ['CCM'] },
  { nace_code: 'J62.0', description: 'Computer programming', eligible: true, objectives: ['CCM'] },
];

interface NACESearchProps {
  onSelect?: (naceCode: string) => void;
}

const NACESearch: React.FC<NACESearchProps> = ({ onSelect }) => {
  const [query, setQuery] = useState('');
  const [showResults, setShowResults] = useState(false);

  const filtered = DEMO_RESULTS.filter(r =>
    r.nace_code.toLowerCase().includes(query.toLowerCase()) ||
    r.description.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <Box sx={{ position: 'relative' }}>
      <TextField
        fullWidth
        label="Search NACE Code"
        placeholder="Enter NACE code (e.g., D35.11) or description..."
        value={query}
        onChange={(e) => { setQuery(e.target.value); setShowResults(true); }}
        onFocus={() => setShowResults(true)}
        InputProps={{
          startAdornment: <InputAdornment position="start"><Search /></InputAdornment>,
        }}
      />
      {showResults && query.length > 0 && (
        <Paper
          sx={{ position: 'absolute', zIndex: 10, width: '100%', maxHeight: 300, overflow: 'auto', mt: 0.5 }}
          elevation={3}
        >
          <List dense>
            {filtered.map(result => (
              <ListItem
                key={result.nace_code}
                component="div"
                onClick={() => { onSelect?.(result.nace_code); setQuery(result.nace_code); setShowResults(false); }}
                sx={{ cursor: 'pointer', '&:hover': { backgroundColor: '#F5F5F5' } }}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="body2" sx={{ fontWeight: 700 }}>{result.nace_code}</Typography>
                      <Typography variant="body2">{result.description}</Typography>
                    </Box>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
                      {result.objectives.map(obj => (
                        <Chip key={obj} label={obj} size="small" variant="outlined" color="primary" />
                      ))}
                      <Chip label={result.eligible ? 'Eligible' : 'Not Eligible'} size="small" color={result.eligible ? 'success' : 'default'} />
                    </Box>
                  }
                />
              </ListItem>
            ))}
            {filtered.length === 0 && (
              <ListItem>
                <ListItemText primary="No matching NACE codes found" />
              </ListItem>
            )}
          </List>
        </Paper>
      )}
    </Box>
  );
};

export default NACESearch;
