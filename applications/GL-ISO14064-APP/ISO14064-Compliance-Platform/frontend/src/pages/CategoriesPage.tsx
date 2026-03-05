/**
 * CategoriesPage - ISO 14064-1 category details
 *
 * Shows CategoryOverview grid plus detailed CategoryDetail for each
 * category with gas breakdown and facility breakdown charts.
 */

import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useParams } from 'react-router-dom';
import { Box, Typography, Alert, Tabs, Tab, Grid } from '@mui/material';
import type { AppDispatch, AppRootState } from '../store';
import { fetchCategoryResults } from '../store/slices/inventorySlice';
import LoadingSpinner from '../components/common/LoadingSpinner';
import CategoryOverview from '../components/categories/CategoryOverview';
import CategoryDetail from '../components/categories/CategoryDetail';
import { ISOCategory, ISO_CATEGORY_SHORT_NAMES } from '../types';

const ALL_CATEGORIES = Object.values(ISOCategory);

const CategoriesPage: React.FC = () => {
  const { id: inventoryId } = useParams<{ id: string }>();
  const dispatch = useDispatch<AppDispatch>();
  const { categoryResults, loading, error } = useSelector(
    (s: AppRootState) => s.inventory,
  );
  const [selectedTab, setSelectedTab] = useState(0);

  useEffect(() => {
    if (inventoryId) {
      dispatch(fetchCategoryResults(inventoryId));
    }
  }, [dispatch, inventoryId]);

  if (loading && categoryResults.length === 0) {
    return <LoadingSpinner message="Loading categories..." />;
  }

  const selectedCategory = ALL_CATEGORIES[selectedTab];
  const selectedResult = categoryResults.find((c) => c.category === selectedCategory);

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        ISO 14064-1 Categories
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Six emission/removal categories per ISO 14064-1:2018
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* Overview grid */}
      <Box sx={{ mb: 3 }}>
        <CategoryOverview
          categories={categoryResults}
          onCategoryClick={(cat) => {
            const idx = ALL_CATEGORIES.indexOf(cat);
            if (idx >= 0) setSelectedTab(idx);
          }}
        />
      </Box>

      {/* Tab navigation for detail view */}
      <Tabs
        value={selectedTab}
        onChange={(_, v) => setSelectedTab(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{
          mb: 2,
          '& .MuiTab-root': { textTransform: 'none' },
          '& .Mui-selected': { color: '#1b5e20', fontWeight: 600 },
          '& .MuiTabs-indicator': { backgroundColor: '#1b5e20' },
        }}
      >
        {ALL_CATEGORIES.map((cat) => (
          <Tab key={cat} label={ISO_CATEGORY_SHORT_NAMES[cat]} />
        ))}
      </Tabs>

      {/* Detail panel */}
      {selectedResult ? (
        <CategoryDetail category={selectedResult} />
      ) : (
        <Alert severity="info">
          No data available for {ISO_CATEGORY_SHORT_NAMES[selectedCategory]}.
          Add emission sources to populate this category.
        </Alert>
      )}
    </Box>
  );
};

export default CategoriesPage;
