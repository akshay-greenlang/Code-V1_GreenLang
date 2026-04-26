# -*- coding: utf-8 -*-
"""Domain errors for source-rights enforcement (Phase 1)."""
from __future__ import annotations


class SourceRightsError(Exception):
    """Base class for source-rights enforcement errors."""


class IngestionBlocked(SourceRightsError):
    """Raised when a source cannot be ingested.

    Reasons include: source is ``blocked``; source has
    ``legal_signoff.status != approved``; source's
    ``release_milestone`` is later than the running release profile;
    source is missing from the registry entirely.
    """


class RightsDenied(SourceRightsError):
    """Raised (or returned as a Decision) when a tenant cannot read a factor.

    The exception form is used in the ingestion path; the route layer
    typically uses the returned :class:`Decision` instead so it can
    filter quietly without 5xx-ing.
    """


class LicenceMismatch(SourceRightsError):
    """Raised when a record's ``licence`` field does not match the source registry.

    The provenance gate uses this to refuse publishing a record whose
    ``licence`` text doesn't match what the source registry says the
    source carries.
    """
