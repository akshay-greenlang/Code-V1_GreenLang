/**
 * VerificationTimeline - Stage timeline for verification workflow
 *
 * Displays the 5-stage verification pipeline (draft -> internal_review
 * -> approved -> external_verification -> verified) with the current
 * stage highlighted and completion timestamps.
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Stepper,
  Step,
  StepLabel,
  StepConnector,
  Typography,
  Box,
  styled,
} from '@mui/material';
import {
  Edit,
  RateReview,
  ThumbUp,
  Verified,
  CheckCircle,
} from '@mui/icons-material';
import { VerificationStage } from '../../types';
import type { VerificationStageDetail } from '../../types';
import { formatDate } from '../../utils/formatters';

const STAGE_ORDER: VerificationStage[] = [
  VerificationStage.DRAFT,
  VerificationStage.INTERNAL_REVIEW,
  VerificationStage.APPROVED,
  VerificationStage.EXTERNAL_VERIFICATION,
  VerificationStage.VERIFIED,
];

const STAGE_LABELS: Record<VerificationStage, string> = {
  [VerificationStage.DRAFT]: 'Draft',
  [VerificationStage.INTERNAL_REVIEW]: 'Internal Review',
  [VerificationStage.APPROVED]: 'Approved',
  [VerificationStage.EXTERNAL_VERIFICATION]: 'External Verification',
  [VerificationStage.VERIFIED]: 'Verified',
};

const STAGE_ICONS: Record<VerificationStage, React.ReactElement> = {
  [VerificationStage.DRAFT]: <Edit />,
  [VerificationStage.INTERNAL_REVIEW]: <RateReview />,
  [VerificationStage.APPROVED]: <ThumbUp />,
  [VerificationStage.EXTERNAL_VERIFICATION]: <Verified />,
  [VerificationStage.VERIFIED]: <CheckCircle />,
};

const GreenConnector = styled(StepConnector)(() => ({
  '&.Mui-active .MuiStepConnector-line': {
    borderColor: '#1b5e20',
  },
  '&.Mui-completed .MuiStepConnector-line': {
    borderColor: '#1b5e20',
  },
  '& .MuiStepConnector-line': {
    borderTopWidth: 3,
    borderRadius: 1,
  },
}));

interface VerificationTimelineProps {
  currentStage: VerificationStage;
  stages?: VerificationStageDetail[];
}

const VerificationTimeline: React.FC<VerificationTimelineProps> = ({
  currentStage,
  stages,
}) => {
  const activeStep = STAGE_ORDER.indexOf(currentStage);

  return (
    <Card>
      <CardHeader
        title="Verification Progress"
        subheader={`Current stage: ${STAGE_LABELS[currentStage]}`}
      />
      <CardContent>
        <Stepper
          activeStep={activeStep}
          alternativeLabel
          connector={<GreenConnector />}
        >
          {STAGE_ORDER.map((stage, idx) => {
            const detail = stages?.find((s) => s.stage === stage);
            const isCompleted = idx < activeStep;
            const isCurrent = idx === activeStep;

            return (
              <Step key={stage} completed={isCompleted}>
                <StepLabel
                  StepIconComponent={() => (
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        bgcolor: isCompleted
                          ? '#1b5e20'
                          : isCurrent
                          ? '#2e7d32'
                          : '#e0e0e0',
                        color: isCompleted || isCurrent ? '#fff' : '#757575',
                        transition: 'all 0.3s ease',
                        boxShadow: isCurrent ? '0 0 0 4px rgba(46,125,50,0.2)' : 'none',
                      }}
                    >
                      {React.cloneElement(STAGE_ICONS[stage], { fontSize: 'small' })}
                    </Box>
                  )}
                >
                  <Typography
                    variant="caption"
                    fontWeight={isCurrent ? 700 : 400}
                    color={isCurrent ? '#1b5e20' : 'text.secondary'}
                  >
                    {STAGE_LABELS[stage]}
                  </Typography>
                  {detail?.completed_at && (
                    <Typography variant="caption" display="block" color="text.secondary">
                      {formatDate(detail.completed_at)}
                    </Typography>
                  )}
                  {detail && detail.finding_count > 0 && (
                    <Typography variant="caption" display="block" color="warning.main">
                      {detail.finding_count} finding{detail.finding_count > 1 ? 's' : ''}
                    </Typography>
                  )}
                </StepLabel>
              </Step>
            );
          })}
        </Stepper>
      </CardContent>
    </Card>
  );
};

export default VerificationTimeline;
