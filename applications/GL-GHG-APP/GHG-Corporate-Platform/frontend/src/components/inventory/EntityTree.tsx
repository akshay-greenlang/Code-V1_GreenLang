/**
 * EntityTree - Organization entity hierarchy tree view
 *
 * Displays the organizational structure as a collapsible tree using
 * MUI TreeView. Shows parent org at root with subsidiaries, facilities,
 * and operations as children. Supports inline editing of entity name
 * and ownership percentage, plus add/remove actions.
 */

import React, { useState, useCallback } from 'react';
import {
  Box,
  Typography,
  IconButton,
  TextField,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Card,
  CardContent,
  Collapse,
} from '@mui/material';
import {
  ExpandMore,
  ChevronRight,
  Edit,
  Delete,
  Add,
  Business,
  LocationOn,
  AccountBalance,
  Settings,
  Check,
  Close,
} from '@mui/icons-material';
import type { Entity } from '../../types';

interface EntityTreeProps {
  entities: Entity[];
  onUpdate: (entity: Entity) => void;
  onAdd: (parentId: string | null, entity: Partial<Entity>) => void;
  onRemove: (entityId: string) => void;
}

const ENTITY_ICONS: Record<string, React.ReactNode> = {
  parent: <Business fontSize="small" color="primary" />,
  subsidiary: <AccountBalance fontSize="small" color="secondary" />,
  facility: <LocationOn fontSize="small" color="action" />,
  operation: <Settings fontSize="small" color="action" />,
};

const ENTITY_TYPE_OPTIONS = [
  { value: 'subsidiary', label: 'Subsidiary' },
  { value: 'facility', label: 'Facility' },
  { value: 'operation', label: 'Operation' },
];

interface EntityNodeProps {
  entity: Entity;
  depth: number;
  onUpdate: (entity: Entity) => void;
  onAdd: (parentId: string | null, entity: Partial<Entity>) => void;
  onRemove: (entityId: string) => void;
}

const EntityNode: React.FC<EntityNodeProps> = ({
  entity,
  depth,
  onUpdate,
  onAdd,
  onRemove,
}) => {
  const [expanded, setExpanded] = useState(depth < 2);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState(entity.name);
  const [editOwnership, setEditOwnership] = useState(entity.ownership_percentage);
  const hasChildren = entity.children && entity.children.length > 0;

  const handleSave = useCallback(() => {
    onUpdate({
      ...entity,
      name: editName,
      ownership_percentage: editOwnership,
    });
    setEditing(false);
  }, [entity, editName, editOwnership, onUpdate]);

  const handleCancel = useCallback(() => {
    setEditName(entity.name);
    setEditOwnership(entity.ownership_percentage);
    setEditing(false);
  }, [entity]);

  return (
    <Box sx={{ ml: depth * 3 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          py: 0.75,
          px: 1,
          borderRadius: 1,
          '&:hover': { backgroundColor: 'action.hover' },
          gap: 0.5,
        }}
      >
        {/* Expand toggle */}
        <IconButton
          size="small"
          onClick={() => setExpanded(!expanded)}
          sx={{ visibility: hasChildren ? 'visible' : 'hidden' }}
        >
          {expanded ? <ExpandMore fontSize="small" /> : <ChevronRight fontSize="small" />}
        </IconButton>

        {/* Icon */}
        {ENTITY_ICONS[entity.entity_type] || ENTITY_ICONS.operation}

        {/* Name / edit */}
        {editing ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
            <TextField
              size="small"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              sx={{ flex: 1 }}
            />
            <TextField
              size="small"
              type="number"
              value={editOwnership}
              onChange={(e) => setEditOwnership(parseFloat(e.target.value))}
              sx={{ width: 80 }}
              InputProps={{ endAdornment: <Typography variant="caption">%</Typography> }}
            />
            <IconButton size="small" color="success" onClick={handleSave}>
              <Check fontSize="small" />
            </IconButton>
            <IconButton size="small" color="error" onClick={handleCancel}>
              <Close fontSize="small" />
            </IconButton>
          </Box>
        ) : (
          <>
            <Typography variant="body2" sx={{ flex: 1, fontWeight: depth === 0 ? 600 : 400 }}>
              {entity.name}
            </Typography>
            <Chip
              label={`${entity.ownership_percentage}%`}
              size="small"
              variant="outlined"
              sx={{ mr: 1 }}
            />
            <Chip
              label={entity.entity_type}
              size="small"
              color={entity.entity_type === 'parent' ? 'primary' : 'default'}
            />
            <IconButton size="small" onClick={() => setEditing(true)}>
              <Edit fontSize="small" />
            </IconButton>
            <IconButton
              size="small"
              onClick={() => onAdd(entity.id, {})}
            >
              <Add fontSize="small" />
            </IconButton>
            {entity.entity_type !== 'parent' && (
              <IconButton size="small" color="error" onClick={() => onRemove(entity.id)}>
                <Delete fontSize="small" />
              </IconButton>
            )}
          </>
        )}
      </Box>

      {/* Children */}
      <Collapse in={expanded}>
        {entity.children?.map((child) => (
          <EntityNode
            key={child.id}
            entity={child}
            depth={depth + 1}
            onUpdate={onUpdate}
            onAdd={onAdd}
            onRemove={onRemove}
          />
        ))}
      </Collapse>
    </Box>
  );
};

const EntityTree: React.FC<EntityTreeProps> = ({
  entities,
  onUpdate,
  onAdd,
  onRemove,
}) => {
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [addParentId, setAddParentId] = useState<string | null>(null);
  const [newEntity, setNewEntity] = useState({
    name: '',
    entity_type: 'facility',
    country: '',
    ownership_percentage: 100,
  });

  const handleOpenAdd = useCallback((parentId: string | null, _: Partial<Entity>) => {
    setAddParentId(parentId);
    setNewEntity({ name: '', entity_type: 'facility', country: '', ownership_percentage: 100 });
    setAddDialogOpen(true);
  }, []);

  const handleConfirmAdd = useCallback(() => {
    onAdd(addParentId, {
      name: newEntity.name,
      entity_type: newEntity.entity_type as Entity['entity_type'],
      country: newEntity.country,
      ownership_percentage: newEntity.ownership_percentage,
    } as Partial<Entity>);
    setAddDialogOpen(false);
  }, [addParentId, newEntity, onAdd]);

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Organization Structure</Typography>
          <Button
            startIcon={<Add />}
            size="small"
            variant="outlined"
            onClick={() => handleOpenAdd(null, {})}
          >
            Add Root Entity
          </Button>
        </Box>

        {entities.length === 0 ? (
          <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: 'center' }}>
            No entities defined. Add your organization structure to begin.
          </Typography>
        ) : (
          entities
            .filter((e) => e.parent_id === null)
            .map((root) => (
              <EntityNode
                key={root.id}
                entity={root}
                depth={0}
                onUpdate={onUpdate}
                onAdd={handleOpenAdd}
                onRemove={onRemove}
              />
            ))
        )}

        {/* Add entity dialog */}
        <Dialog open={addDialogOpen} onClose={() => setAddDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Add Entity</DialogTitle>
          <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: '16px !important' }}>
            <TextField
              label="Entity Name"
              fullWidth
              value={newEntity.name}
              onChange={(e) => setNewEntity({ ...newEntity, name: e.target.value })}
            />
            <FormControl fullWidth>
              <InputLabel>Entity Type</InputLabel>
              <Select
                value={newEntity.entity_type}
                label="Entity Type"
                onChange={(e) => setNewEntity({ ...newEntity, entity_type: e.target.value })}
              >
                {ENTITY_TYPE_OPTIONS.map((opt) => (
                  <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Country"
              fullWidth
              value={newEntity.country}
              onChange={(e) => setNewEntity({ ...newEntity, country: e.target.value })}
            />
            <TextField
              label="Ownership %"
              type="number"
              fullWidth
              value={newEntity.ownership_percentage}
              onChange={(e) =>
                setNewEntity({ ...newEntity, ownership_percentage: parseFloat(e.target.value) })
              }
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setAddDialogOpen(false)}>Cancel</Button>
            <Button variant="contained" onClick={handleConfirmAdd} disabled={!newEntity.name}>
              Add
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default EntityTree;
