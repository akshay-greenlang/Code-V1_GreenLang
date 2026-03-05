/**
 * AssignmentPanel - Question assignment to team members
 *
 * Allows assigning questions to team members and viewing current assignments.
 */

import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Chip, MenuItem } from '@mui/material';
import { PersonAdd, PersonOutline } from '@mui/icons-material';

interface AssignmentPanelProps {
  currentAssignee: string | null;
  teamMembers: Array<{ id: string; name: string; email: string }>;
  onAssign: (memberId: string) => void;
}

const AssignmentPanel: React.FC<AssignmentPanelProps> = ({
  currentAssignee,
  teamMembers,
  onAssign,
}) => {
  const [selectedMember, setSelectedMember] = useState('');

  const handleAssign = () => {
    if (selectedMember) {
      onAssign(selectedMember);
      setSelectedMember('');
    }
  };

  return (
    <Box>
      <Typography variant="subtitle2" fontWeight={600} gutterBottom>
        Assignment
      </Typography>

      {currentAssignee ? (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
          <PersonOutline fontSize="small" sx={{ color: '#1565c0' }} />
          <Typography variant="body2">
            Assigned to: <strong>{currentAssignee}</strong>
          </Typography>
        </Box>
      ) : (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Not yet assigned
        </Typography>
      )}

      <Box sx={{ display: 'flex', gap: 1 }}>
        <TextField
          select
          size="small"
          label="Assign to"
          value={selectedMember}
          onChange={(e) => setSelectedMember(e.target.value)}
          sx={{ minWidth: 200 }}
        >
          {teamMembers.map((m) => (
            <MenuItem key={m.id} value={m.id}>
              {m.name}
            </MenuItem>
          ))}
        </TextField>
        <Button
          variant="outlined"
          size="small"
          startIcon={<PersonAdd />}
          onClick={handleAssign}
          disabled={!selectedMember}
        >
          Assign
        </Button>
      </Box>
    </Box>
  );
};

export default AssignmentPanel;
