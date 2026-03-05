/**
 * Taxonomy-specific helper functions for labels, colors, and mappings.
 */

import { EnvironmentalObjective, ActivityType, AlignmentStatus, DNSHStatus, SafeguardTopic, GapCategory, GapSeverity, ExposureType } from '../types';

export const objectiveLabel = (objective: EnvironmentalObjective): string => {
  const labels: Record<EnvironmentalObjective, string> = {
    [EnvironmentalObjective.CLIMATE_MITIGATION]: 'Climate Change Mitigation',
    [EnvironmentalObjective.CLIMATE_ADAPTATION]: 'Climate Change Adaptation',
    [EnvironmentalObjective.WATER_MARINE]: 'Water & Marine Resources',
    [EnvironmentalObjective.CIRCULAR_ECONOMY]: 'Circular Economy',
    [EnvironmentalObjective.POLLUTION_PREVENTION]: 'Pollution Prevention & Control',
    [EnvironmentalObjective.BIODIVERSITY]: 'Biodiversity & Ecosystems',
  };
  return labels[objective] || objective;
};

export const objectiveShortLabel = (objective: EnvironmentalObjective): string => {
  const labels: Record<EnvironmentalObjective, string> = {
    [EnvironmentalObjective.CLIMATE_MITIGATION]: 'CCM',
    [EnvironmentalObjective.CLIMATE_ADAPTATION]: 'CCA',
    [EnvironmentalObjective.WATER_MARINE]: 'WTR',
    [EnvironmentalObjective.CIRCULAR_ECONOMY]: 'CE',
    [EnvironmentalObjective.POLLUTION_PREVENTION]: 'PPC',
    [EnvironmentalObjective.BIODIVERSITY]: 'BIO',
  };
  return labels[objective] || objective;
};

export const objectiveColor = (objective: EnvironmentalObjective): string => {
  const colors: Record<EnvironmentalObjective, string> = {
    [EnvironmentalObjective.CLIMATE_MITIGATION]: '#1B5E20',
    [EnvironmentalObjective.CLIMATE_ADAPTATION]: '#0D47A1',
    [EnvironmentalObjective.WATER_MARINE]: '#01579B',
    [EnvironmentalObjective.CIRCULAR_ECONOMY]: '#E65100',
    [EnvironmentalObjective.POLLUTION_PREVENTION]: '#4A148C',
    [EnvironmentalObjective.BIODIVERSITY]: '#1B5E20',
  };
  return colors[objective] || '#757575';
};

export const activityTypeLabel = (type: ActivityType | null): string => {
  if (!type) return 'Own Performance';
  const labels: Record<ActivityType, string> = {
    [ActivityType.ENABLING]: 'Enabling',
    [ActivityType.TRANSITIONAL]: 'Transitional',
    [ActivityType.OWN_PERFORMANCE]: 'Own Performance',
  };
  return labels[type] || type;
};

export const activityTypeColor = (type: ActivityType | null): string => {
  if (!type) return '#1B5E20';
  const colors: Record<ActivityType, string> = {
    [ActivityType.ENABLING]: '#0D47A1',
    [ActivityType.TRANSITIONAL]: '#E65100',
    [ActivityType.OWN_PERFORMANCE]: '#1B5E20',
  };
  return colors[type] || '#757575';
};

export const alignmentStatusLabel = (status: AlignmentStatus): string => {
  const labels: Record<AlignmentStatus, string> = {
    [AlignmentStatus.NOT_STARTED]: 'Not Started',
    [AlignmentStatus.ELIGIBLE]: 'Eligible',
    [AlignmentStatus.SC_PASS]: 'SC Passed',
    [AlignmentStatus.DNSH_PASS]: 'DNSH Passed',
    [AlignmentStatus.MS_PASS]: 'MS Passed',
    [AlignmentStatus.ALIGNED]: 'Aligned',
    [AlignmentStatus.NOT_ELIGIBLE]: 'Not Eligible',
    [AlignmentStatus.NOT_ALIGNED]: 'Not Aligned',
  };
  return labels[status] || status;
};

export const dnshStatusLabel = (status: DNSHStatus): string => {
  const labels: Record<DNSHStatus, string> = {
    [DNSHStatus.PASS]: 'Pass',
    [DNSHStatus.FAIL]: 'Fail',
    [DNSHStatus.NOT_APPLICABLE]: 'N/A',
    [DNSHStatus.PENDING]: 'Pending',
  };
  return labels[status] || status;
};

export const dnshStatusColor = (status: DNSHStatus): string => {
  const colors: Record<DNSHStatus, string> = {
    [DNSHStatus.PASS]: '#2E7D32',
    [DNSHStatus.FAIL]: '#C62828',
    [DNSHStatus.NOT_APPLICABLE]: '#9E9E9E',
    [DNSHStatus.PENDING]: '#EF6C00',
  };
  return colors[status] || '#757575';
};

export const safeguardTopicLabel = (topic: SafeguardTopic): string => {
  const labels: Record<SafeguardTopic, string> = {
    [SafeguardTopic.HUMAN_RIGHTS]: 'Human Rights',
    [SafeguardTopic.ANTI_CORRUPTION]: 'Anti-Corruption & Bribery',
    [SafeguardTopic.TAXATION]: 'Taxation',
    [SafeguardTopic.FAIR_COMPETITION]: 'Fair Competition',
  };
  return labels[topic] || topic;
};

export const gapCategoryLabel = (category: GapCategory): string => {
  const labels: Record<GapCategory, string> = {
    [GapCategory.SUBSTANTIAL_CONTRIBUTION]: 'Substantial Contribution',
    [GapCategory.DNSH]: 'DNSH',
    [GapCategory.SAFEGUARDS]: 'Minimum Safeguards',
    [GapCategory.DATA]: 'Data Quality',
    [GapCategory.REPORTING]: 'Reporting',
  };
  return labels[category] || category;
};

export const gapSeverityLabel = (severity: GapSeverity): string => {
  const labels: Record<GapSeverity, string> = {
    [GapSeverity.CRITICAL]: 'Critical',
    [GapSeverity.HIGH]: 'High',
    [GapSeverity.MEDIUM]: 'Medium',
    [GapSeverity.LOW]: 'Low',
  };
  return labels[severity] || severity;
};

export const exposureTypeLabel = (type: ExposureType): string => {
  const labels: Record<ExposureType, string> = {
    [ExposureType.GENERAL_LENDING]: 'General Corporate Lending',
    [ExposureType.SPECIALIZED_LENDING]: 'Specialized Lending',
    [ExposureType.PROJECT_FINANCE]: 'Project Finance',
    [ExposureType.EQUITY]: 'Equity Holdings',
    [ExposureType.DEBT_SECURITIES]: 'Debt Securities',
    [ExposureType.DERIVATIVES]: 'Derivatives',
    [ExposureType.MORTGAGE]: 'Mortgages',
    [ExposureType.AUTO_LOAN]: 'Auto Loans',
    [ExposureType.INTERBANK]: 'Interbank Loans',
    [ExposureType.SOVEREIGN]: 'Sovereign Exposures',
  };
  return labels[type] || type;
};

export const epcRatingColor = (rating: string): string => {
  const colors: Record<string, string> = {
    'A': '#1B5E20',
    'B': '#388E3C',
    'C': '#689F38',
    'D': '#AFB42B',
    'E': '#FBC02D',
    'F': '#F57C00',
    'G': '#D32F2F',
  };
  return colors[rating.toUpperCase()] || '#757575';
};

export const alignmentSteps = [
  { key: 'eligibility', label: 'Eligibility', step: 1 },
  { key: 'sc', label: 'Substantial Contribution', step: 2 },
  { key: 'dnsh', label: 'DNSH', step: 3 },
  { key: 'safeguards', label: 'Minimum Safeguards', step: 4 },
  { key: 'aligned', label: 'Aligned', step: 5 },
];

export const allObjectives: EnvironmentalObjective[] = [
  EnvironmentalObjective.CLIMATE_MITIGATION,
  EnvironmentalObjective.CLIMATE_ADAPTATION,
  EnvironmentalObjective.WATER_MARINE,
  EnvironmentalObjective.CIRCULAR_ECONOMY,
  EnvironmentalObjective.POLLUTION_PREVENTION,
  EnvironmentalObjective.BIODIVERSITY,
];

export const allSafeguardTopics: SafeguardTopic[] = [
  SafeguardTopic.HUMAN_RIGHTS,
  SafeguardTopic.ANTI_CORRUPTION,
  SafeguardTopic.TAXATION,
  SafeguardTopic.FAIR_COMPETITION,
];
