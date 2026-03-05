/**
 * CounterpartySearch - Search for counterparties by name or LEI.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, TextField, InputAdornment, List, ListItem, ListItemText, Chip, Box } from '@mui/material';
import { Search } from '@mui/icons-material';

const DEMO_RESULTS = [
  { lei: 'ABCD1234567890ABCDEF', name: 'SolarTech GmbH', country: 'DE', sector: 'Energy', aligned_pct: 85.2 },
  { lei: 'EFGH1234567890ABCDEF', name: 'WindPower SA', country: 'FR', sector: 'Energy', aligned_pct: 78.5 },
  { lei: 'IJKL1234567890ABCDEF', name: 'GreenBuild Corp', country: 'NL', sector: 'Construction', aligned_pct: 45.0 },
];

interface CounterpartySearchProps {
  onSelect?: (lei: string) => void;
}

const CounterpartySearch: React.FC<CounterpartySearchProps> = ({ onSelect }) => {
  const [query, setQuery] = useState('');

  const filtered = query.length > 0
    ? DEMO_RESULTS.filter(r => r.name.toLowerCase().includes(query.toLowerCase()) || r.lei.includes(query.toUpperCase()))
    : [];

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>Counterparty Search</Typography>
        <TextField
          fullWidth
          placeholder="Search by name or LEI..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          InputProps={{ startAdornment: <InputAdornment position="start"><Search /></InputAdornment> }}
          size="small"
        />
        {filtered.length > 0 && (
          <List dense sx={{ mt: 1 }}>
            {filtered.map(r => (
              <ListItem key={r.lei} onClick={() => onSelect?.(r.lei)} sx={{ border: '1px solid #E0E0E0', borderRadius: 1, mb: 0.5, cursor: 'pointer', '&:hover': { backgroundColor: '#F5F5F5' } }}>
                <ListItemText
                  primary={r.name}
                  secondary={`LEI: ${r.lei} | ${r.country} | ${r.sector}`}
                />
                <Chip label={`${r.aligned_pct}% aligned`} size="small" color={r.aligned_pct > 50 ? 'success' : 'warning'} />
              </ListItem>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default CounterpartySearch;
