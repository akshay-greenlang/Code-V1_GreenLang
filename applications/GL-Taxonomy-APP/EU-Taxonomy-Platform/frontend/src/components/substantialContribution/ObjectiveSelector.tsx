/**
 * ObjectiveSelector - Select environmental objective for SC assessment.
 */

import React, { useState } from 'react';
import { Card, CardContent, Typography, Grid, Box, Radio, RadioGroup, FormControlLabel } from '@mui/material';
import { objectiveLabel, objectiveColor, allObjectives } from '../../utils/taxonomyHelpers';
import { EnvironmentalObjective } from '../../types';

interface ObjectiveSelectorProps {
  selected?: EnvironmentalObjective;
  eligibleObjectives?: EnvironmentalObjective[];
  onChange?: (objective: EnvironmentalObjective) => void;
}

const ObjectiveSelector: React.FC<ObjectiveSelectorProps> = ({
  selected,
  eligibleObjectives = [EnvironmentalObjective.CLIMATE_MITIGATION, EnvironmentalObjective.CLIMATE_ADAPTATION],
  onChange,
}) => {
  const [value, setValue] = useState<EnvironmentalObjective | undefined>(selected);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
          Select Environmental Objective
        </Typography>
        <RadioGroup
          value={value || ''}
          onChange={(e) => {
            const obj = e.target.value as EnvironmentalObjective;
            setValue(obj);
            onChange?.(obj);
          }}
        >
          <Grid container spacing={1}>
            {allObjectives.map(obj => {
              const isEligible = eligibleObjectives.includes(obj);
              return (
                <Grid item xs={12} sm={6} key={obj}>
                  <Box
                    sx={{
                      p: 1.5,
                      border: '1px solid',
                      borderColor: value === obj ? objectiveColor(obj) : '#E0E0E0',
                      borderRadius: 1,
                      opacity: isEligible ? 1 : 0.4,
                      backgroundColor: value === obj ? `${objectiveColor(obj)}10` : 'transparent',
                    }}
                  >
                    <FormControlLabel
                      value={obj}
                      control={<Radio size="small" disabled={!isEligible} />}
                      label={
                        <Typography variant="body2" sx={{ fontWeight: value === obj ? 600 : 400 }}>
                          {objectiveLabel(obj)}
                        </Typography>
                      }
                    />
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </RadioGroup>
      </CardContent>
    </Card>
  );
};

export default ObjectiveSelector;
