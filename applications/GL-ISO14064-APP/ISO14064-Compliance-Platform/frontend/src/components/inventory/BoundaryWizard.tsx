/**
 * GL-ISO14064-APP v1.0 - Boundary Setup Wizard
 *
 * Multi-step wizard for defining organizational and operational
 * boundaries per ISO 14064-1 Clause 5.1.  Steps:
 *   1. Select consolidation approach
 *   2. Select entities to include
 *   3. Define operational boundary (categories 1-6)
 *   4. Review and confirm
 */

import React, { useState } from 'react';
import {
  Box, Stepper, Step, StepLabel, Button, Typography,
  Card, CardContent, RadioGroup, FormControlLabel, Radio,
  Checkbox, List, ListItem, ListItemText, ListItemIcon,
  Divider, Alert,
} from '@mui/material';
import {
  ConsolidationApproach, ISOCategory,
  ISO_CATEGORY_NAMES,
} from '../../types';

interface Entity {
  id: string;
  name: string;
  entity_type: string;
  ownership_pct: number;
}

interface Props {
  entities: Entity[];
  onComplete: (data: {
    consolidation_approach: ConsolidationApproach;
    entity_ids: string[];
    included_categories: ISOCategory[];
  }) => void;
  onCancel: () => void;
}

const STEPS = ['Consolidation Approach', 'Select Entities', 'Operational Boundary', 'Review'];

const APPROACH_DESCRIPTIONS: Record<ConsolidationApproach, string> = {
  [ConsolidationApproach.OPERATIONAL_CONTROL]:
    'Include 100% of emissions from operations over which the organization has operational control.',
  [ConsolidationApproach.FINANCIAL_CONTROL]:
    'Include 100% of emissions from operations over which the organization has financial control.',
  [ConsolidationApproach.EQUITY_SHARE]:
    'Include emissions proportional to the organization\'s equity share in each operation.',
};

const ALL_CATEGORIES = Object.values(ISOCategory);

const BoundaryWizard: React.FC<Props> = ({ entities, onComplete, onCancel }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [approach, setApproach] = useState<ConsolidationApproach>(ConsolidationApproach.OPERATIONAL_CONTROL);
  const [selectedEntities, setSelectedEntities] = useState<string[]>(entities.map((e) => e.id));
  const [selectedCategories, setSelectedCategories] = useState<ISOCategory[]>(ALL_CATEGORIES);

  const toggleEntity = (id: string) => {
    setSelectedEntities((prev) =>
      prev.includes(id) ? prev.filter((e) => e !== id) : [...prev, id],
    );
  };

  const toggleCategory = (cat: ISOCategory) => {
    setSelectedCategories((prev) =>
      prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat],
    );
  };

  const handleNext = () => {
    if (activeStep === STEPS.length - 1) {
      onComplete({
        consolidation_approach: approach,
        entity_ids: selectedEntities,
        included_categories: selectedCategories,
      });
    } else {
      setActiveStep((s) => s + 1);
    }
  };

  const canProceed =
    activeStep === 0 ||
    (activeStep === 1 && selectedEntities.length > 0) ||
    (activeStep === 2 && selectedCategories.length >= 2) ||
    activeStep === 3;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Boundary Setup Wizard
        </Typography>
        <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
          {STEPS.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {/* Step 1: Consolidation Approach */}
        {activeStep === 0 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Select the consolidation approach per ISO 14064-1 Clause 5.1:
            </Typography>
            <RadioGroup
              value={approach}
              onChange={(e) => setApproach(e.target.value as ConsolidationApproach)}
            >
              {Object.values(ConsolidationApproach).map((a) => (
                <FormControlLabel
                  key={a}
                  value={a}
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="body2" fontWeight={600}>
                        {a.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {APPROACH_DESCRIPTIONS[a]}
                      </Typography>
                    </Box>
                  }
                />
              ))}
            </RadioGroup>
          </Box>
        )}

        {/* Step 2: Entity Selection */}
        {activeStep === 1 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Select entities to include in the organizational boundary:
            </Typography>
            {entities.length === 0 ? (
              <Alert severity="info">No entities defined yet. Add entities in Organization Setup first.</Alert>
            ) : (
              <List dense>
                {entities.map((ent) => (
                  <ListItem key={ent.id} button onClick={() => toggleEntity(ent.id)}>
                    <ListItemIcon>
                      <Checkbox edge="start" checked={selectedEntities.includes(ent.id)} />
                    </ListItemIcon>
                    <ListItemText
                      primary={ent.name}
                      secondary={`${ent.entity_type} | ${ent.ownership_pct}% ownership`}
                    />
                  </ListItem>
                ))}
              </List>
            )}
          </Box>
        )}

        {/* Step 3: Operational Boundary */}
        {activeStep === 2 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Select ISO 14064-1 categories to include (Categories 1 and 2 are mandatory):
            </Typography>
            <List dense>
              {ALL_CATEGORIES.map((cat) => {
                const isMandatory = cat === ISOCategory.CATEGORY_1_DIRECT || cat === ISOCategory.CATEGORY_2_ENERGY;
                return (
                  <ListItem key={cat} button onClick={() => !isMandatory && toggleCategory(cat)}>
                    <ListItemIcon>
                      <Checkbox
                        edge="start"
                        checked={selectedCategories.includes(cat)}
                        disabled={isMandatory}
                      />
                    </ListItemIcon>
                    <ListItemText
                      primary={ISO_CATEGORY_NAMES[cat]}
                      secondary={isMandatory ? 'Mandatory per ISO 14064-1' : 'Optional - subject to significance assessment'}
                    />
                  </ListItem>
                );
              })}
            </List>
          </Box>
        )}

        {/* Step 4: Review */}
        {activeStep === 3 && (
          <Box>
            <Typography variant="subtitle2" gutterBottom>
              Review your boundary configuration:
            </Typography>
            <Box sx={{ pl: 2 }}>
              <Typography variant="body2">
                <strong>Approach:</strong>{' '}
                {approach.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
              </Typography>
              <Typography variant="body2" mt={1}>
                <strong>Entities:</strong> {selectedEntities.length} of {entities.length} included
              </Typography>
              <Typography variant="body2" mt={1}>
                <strong>Categories:</strong> {selectedCategories.length} of 6 included
              </Typography>
              <List dense>
                {selectedCategories.map((cat) => (
                  <ListItem key={cat} sx={{ py: 0 }}>
                    <ListItemText primary={ISO_CATEGORY_NAMES[cat]} primaryTypographyProps={{ variant: 'caption' }} />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Box>
        )}

        <Divider sx={{ my: 2 }} />
        <Box display="flex" justifyContent="space-between">
          <Button onClick={activeStep === 0 ? onCancel : () => setActiveStep((s) => s - 1)}>
            {activeStep === 0 ? 'Cancel' : 'Back'}
          </Button>
          <Button variant="contained" onClick={handleNext} disabled={!canProceed}>
            {activeStep === STEPS.length - 1 ? 'Confirm' : 'Next'}
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default BoundaryWizard;
