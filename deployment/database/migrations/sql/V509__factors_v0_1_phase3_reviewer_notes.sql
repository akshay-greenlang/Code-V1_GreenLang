-- =============================================================================
-- V509: GreenLang Factors v0.1 - Phase 3 source_artifacts.reviewer_notes (FORWARD)
-- =============================================================================
-- Description: Adds an additive `reviewer_notes` JSONB column to
--              `factors_v0_1.source_artifacts`. Required by the Phase 3
--              Wave 2.5 PDF/OCR family (Block 2 / Block 3 of the exit
--              checklist) so OCR confidence scores, manual-correction
--              audit trails, and reviewer sign-offs are persisted at the
--              raw-artifact granularity (one source row -> one bytes
--              blob -> one reviewer_notes payload).
--
--              Canonical JSONB shape (validated at the application layer,
--              not by a CHECK because JSONB shape evolves):
--
--                  {
--                    "extraction_method":
--                        "pdfplumber" | "ocrmypdf" | "manual",
--                    "ocr_confidence_min": <float in [0,1]>,
--                    "manual_corrections": [
--                      {
--                        "row_ref": "p<page>.t<table>.r<row>.c<col>",
--                        "field": "<dotted attribute path>",
--                        "before": <prior value (any JSON type)>,
--                        "after":  <new value   (any JSON type)>,
--                        "approver":   "human:<email>",
--                        "approved_at": "<ISO-8601 timestamp>"
--                      },
--                      ...
--                    ],
--                    "reviewer_signoff": {
--                      "by": "human:<email>",
--                      "at": "<ISO-8601 timestamp>"
--                    }
--                  }
--
--              The column is nullable because the vast majority of
--              ingestion runs (Excel, CSV, JSON, API) do NOT need a
--              reviewer trail -- only PDF/OCR + manual-corrected runs
--              produce a reviewer_notes payload. NULL signals "no human
--              review was required" (or "review pending" until populated).
--
--              Idempotency: the ALTER TABLE is wrapped in a DO $$ block
--              that checks information_schema.columns first, so applying
--              this forward migration twice is a no-op (matters for the
--              SQLite mirror parity test that calls apply_phase3_ddl on
--              an existing connection).
--
-- Authority:
--   - PHASE_3_PLAN.md §"Block 3 -- PDF/OCR family"
--   - PHASE_3_EXIT_CHECKLIST.md Block 2 (`reviewer_notes` JSONB column
--     requirement) + Block 3 (PDF/OCR family).
--
-- Wave: Phase 3 / Wave 2.5
-- Postgres target: 16+
-- Author: GL-DataIntegrationEngineer
-- Created: 2026-04-28
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. Idempotent ALTER TABLE -- only add the column if it does not exist.
-- -----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'factors_v0_1'
          AND table_name   = 'source_artifacts'
          AND column_name  = 'reviewer_notes'
    ) THEN
        ALTER TABLE factors_v0_1.source_artifacts
            ADD COLUMN reviewer_notes JSONB;

        COMMENT ON COLUMN factors_v0_1.source_artifacts.reviewer_notes IS
            'Phase 3 Wave 2.5: reviewer audit trail for PDF/OCR + manually '
            'corrected ingestions. JSONB shape carries extraction_method, '
            'ocr_confidence_min, manual_corrections[], reviewer_signoff. '
            'NULL when no human review was required.';
    END IF;
END
$$;

-- =============================================================================
-- End V509 FORWARD
-- =============================================================================
