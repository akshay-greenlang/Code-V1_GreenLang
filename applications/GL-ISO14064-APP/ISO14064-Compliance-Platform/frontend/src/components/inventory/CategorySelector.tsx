/**
 * GL-ISO14064-APP v1.0 - ISO Category Selector
 *
 * Provides a visual selector for the six ISO 14064-1 categories
 * with icons, colours, emission totals, and significance badges.
 * Used for filtering emission sources and navigating category details.
 */

import React from 'react';
import {
  Card, CardContent, Typography, Box, Grid, Chip,
  CardActionArea,
} from '@mui/material';
import FactoryIcon from '@mui/icons-material/Factory';
import BoltIcon from '@mui/icons-material/Bolt';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import ShoppingCartIcon from '@mui/icons-material/ShoppingCart';
import InventoryIcon from '@mui/icons-material/Inventory';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import {
  ISOCategory, ISO_CATEGORY_SHORT_NAMES, CATEGORY_COLORS,
  SignificanceLevel,
} from '../../types';
import type { CategoryBreakdownItem } from '../../types';

interface Props {
  categories: CategoryBreakdownItem[];
  selectedCategory?: ISOCategory | null;
  onSelect: (category: ISOCategory) => void;
}

const CATEGORY_ICONS: Record<ISOCategory, React.ReactNode> = {
  [ISOCategory.CATEGORY_1_DIRECT]: <FactoryIcon />,
  [ISOCategory.CATEGORY_2_ENERGY]: <BoltIcon />,
  [ISOCategory.CATEGORY_3_TRANSPORT]: <LocalShippingIcon />,
  [ISOCategory.CATEGORY_4_PRODUCTS_USED]: <ShoppingCartIcon />,
  [ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG]: <InventoryIcon />,
  [ISOCategory.CATEGORY_6_OTHER]: <MoreHorizIcon />,
};

function significanceColor(level: SignificanceLevel): 'error' | 'success' | 'warning' {
  switch (level) {
    case SignificanceLevel.SIGNIFICANT:
      return 'error';
    case SignificanceLevel.NOT_SIGNIFICANT:
      return 'success';
    default:
      return 'warning';
  }
}

const CategorySelector: React.FC<Props> = ({ categories, selectedCategory, onSelect }) => {
  const categoryMap = new Map(categories.map((c) => [c.category, c]));

  return (
    <Grid container spacing={2}>
      {Object.values(ISOCategory).map((cat) => {
        const data = categoryMap.get(cat);
        const isSelected = selectedCategory === cat;
        const color = CATEGORY_COLORS[cat];

        return (
          <Grid item xs={12} sm={6} md={4} lg={2} key={cat}>
            <Card
              sx={{
                border: isSelected ? `2px solid ${color}` : '1px solid #e0e0e0',
                transition: 'border-color 0.2s',
              }}
            >
              <CardActionArea onClick={() => onSelect(cat)} sx={{ p: 2 }}>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Box sx={{ color }}>{CATEGORY_ICONS[cat]}</Box>
                  <Typography variant="caption" fontWeight={600} noWrap>
                    {ISO_CATEGORY_SHORT_NAMES[cat]}
                  </Typography>
                </Box>
                <Typography variant="h6" fontWeight={700}>
                  {data ? data.emissions_tco2e.toLocaleString(undefined, { maximumFractionDigits: 0 }) : '0'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  tCO2e
                </Typography>
                {data && (
                  <Box mt={1}>
                    <Chip
                      label={data.significance.replace(/_/g, ' ')}
                      size="small"
                      color={significanceColor(data.significance)}
                      variant="outlined"
                    />
                  </Box>
                )}
              </CardActionArea>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
};

export default CategorySelector;
