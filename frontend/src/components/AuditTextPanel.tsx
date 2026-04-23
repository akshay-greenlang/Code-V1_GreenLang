/**
 * AuditTextPanel — renders the Wave 2.5 `audit_text` narrative with a
 * `[Draft]` banner treatment when `audit_text_draft` is true.
 *
 * Drafts come from unapproved templates; the banner exists to prevent
 * operators from shipping unreviewed copy into regulatory submissions.
 */
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";

export interface AuditTextPanelProps {
  auditText?: string | null;
  /** When true, render the red [Draft] banner. */
  draft?: boolean | null;
  /** Omit the section header (for inline use). */
  hideHeader?: boolean;
}

export function AuditTextPanel({ auditText, draft, hideHeader }: AuditTextPanelProps) {
  if (!auditText || auditText.trim().length === 0) {
    return (
      <Stack spacing={1} data-testid="audit-text-panel">
        {!hideHeader && <Typography variant="subtitle2">Audit narrative</Typography>}
        <Typography variant="caption" color="text.secondary">
          No audit text available for this resolution.
        </Typography>
      </Stack>
    );
  }
  return (
    <Stack spacing={1} data-testid="audit-text-panel">
      {!hideHeader && (
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography variant="subtitle2">Audit narrative</Typography>
          {draft && (
            <Box
              data-testid="audit-text-draft-banner"
              component="span"
              sx={{
                bgcolor: "error.main",
                color: "error.contrastText",
                px: 1,
                py: 0.25,
                borderRadius: 0.5,
                fontWeight: 700,
                fontSize: 11,
                letterSpacing: 1,
              }}
              aria-label="Draft audit text — not safe for regulatory submission"
            >
              [DRAFT]
            </Box>
          )}
        </Stack>
      )}
      {draft && (
        <Alert severity="warning" role="alert" data-testid="audit-text-draft-alert">
          This audit narrative is a <strong>draft</strong> produced from an
          unapproved template. Do not ship into a regulatory submission
          without methodology-lead sign-off.
        </Alert>
      )}
      <Box
        component="blockquote"
        sx={{
          m: 0,
          p: 2,
          borderLeft: "4px solid",
          borderColor: draft ? "error.main" : "success.main",
          bgcolor: "background.paper",
          whiteSpace: "pre-wrap",
          fontFamily: "inherit",
        }}
        data-testid="audit-text-body"
      >
        <Typography variant="body2" component="div">
          {auditText}
        </Typography>
      </Box>
    </Stack>
  );
}

export default AuditTextPanel;
