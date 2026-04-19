# -*- coding: utf-8 -*-
"""
ValueChain Intake Agent Exceptions

Custom exception hierarchy for intake agent error handling.

Version: 1.0.0
Phase: 3 (Weeks 7-10)
Date: 2025-10-30
"""


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class IntakeAgentError(Exception):
    """Base exception for ValueChain Intake Agent."""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# ============================================================================
# PARSER EXCEPTIONS
# ============================================================================

class ParserError(IntakeAgentError):
    """Base exception for parser errors."""
    pass


class UnsupportedFormatError(ParserError):
    """Raised when file format is not supported."""
    pass


class EncodingDetectionError(ParserError):
    """Raised when file encoding cannot be detected."""
    pass


class FileParseError(ParserError):
    """Raised when file parsing fails."""
    pass


class SchemaValidationError(ParserError):
    """Raised when data doesn't match expected schema."""
    pass


class ExcelSheetNotFoundError(ParserError):
    """Raised when specified Excel sheet is not found."""
    pass


class XMLPathError(ParserError):
    """Raised when XPath query fails."""
    pass


class PDFOCRError(ParserError):
    """Raised when PDF OCR processing fails."""
    pass


# ============================================================================
# CONNECTOR EXCEPTIONS
# ============================================================================

class ConnectorError(IntakeAgentError):
    """Base exception for ERP connector errors."""
    pass


class ConnectionError(ConnectorError):
    """Raised when connection to ERP system fails."""
    pass


class AuthenticationError(ConnectorError):
    """Raised when ERP authentication fails."""
    pass


class QueryError(ConnectorError):
    """Raised when ERP query execution fails."""
    pass


class DataExtractionError(ConnectorError):
    """Raised when data extraction from ERP fails."""
    pass


class RateLimitError(ConnectorError):
    """Raised when API rate limit is exceeded."""
    pass


# ============================================================================
# ENTITY RESOLUTION EXCEPTIONS
# ============================================================================

class EntityResolutionError(IntakeAgentError):
    """Base exception for entity resolution errors."""
    pass


class NoMatchFoundError(EntityResolutionError):
    """Raised when no entity match is found."""
    pass


class MultipleMatchesError(EntityResolutionError):
    """Raised when multiple ambiguous matches are found."""
    pass


class MDMLookupError(EntityResolutionError):
    """Raised when MDM lookup fails."""
    pass


class FuzzyMatchError(EntityResolutionError):
    """Raised when fuzzy matching fails."""
    pass


class ConfidenceThresholdError(EntityResolutionError):
    """Raised when match confidence is below threshold."""
    pass


# ============================================================================
# REVIEW QUEUE EXCEPTIONS
# ============================================================================

class ReviewQueueError(IntakeAgentError):
    """Base exception for review queue errors."""
    pass


class QueueItemNotFoundError(ReviewQueueError):
    """Raised when queue item is not found."""
    pass


class InvalidActionError(ReviewQueueError):
    """Raised when invalid review action is attempted."""
    pass


class ReviewAlreadyCompletedError(ReviewQueueError):
    """Raised when trying to review already completed item."""
    pass


class QueuePersistenceError(ReviewQueueError):
    """Raised when queue persistence fails."""
    pass


# ============================================================================
# DATA QUALITY EXCEPTIONS
# ============================================================================

class DataQualityError(IntakeAgentError):
    """Base exception for data quality errors."""
    pass


class DQICalculationError(DataQualityError):
    """Raised when DQI calculation fails."""
    pass


class CompletenessCheckError(DataQualityError):
    """Raised when completeness check fails."""
    pass


class ValidationError(DataQualityError):
    """Raised when data validation fails."""
    pass


class GapAnalysisError(DataQualityError):
    """Raised when gap analysis fails."""
    pass


# ============================================================================
# CONFIGURATION EXCEPTIONS
# ============================================================================

class ConfigurationError(IntakeAgentError):
    """Raised when configuration is invalid."""
    pass


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass


class InvalidConfigValueError(ConfigurationError):
    """Raised when configuration value is invalid."""
    pass


# ============================================================================
# INGESTION EXCEPTIONS
# ============================================================================

class IngestionError(IntakeAgentError):
    """Base exception for ingestion errors."""
    pass


class BatchProcessingError(IngestionError):
    """Raised when batch processing fails."""
    pass


class RecordValidationError(IngestionError):
    """Raised when record validation fails."""
    pass


class TenantIsolationError(IngestionError):
    """Raised when tenant isolation is violated."""
    pass


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Base
    "IntakeAgentError",

    # Parser Exceptions
    "ParserError",
    "UnsupportedFormatError",
    "EncodingDetectionError",
    "FileParseError",
    "SchemaValidationError",
    "ExcelSheetNotFoundError",
    "XMLPathError",
    "PDFOCRError",

    # Connector Exceptions
    "ConnectorError",
    "ConnectionError",
    "AuthenticationError",
    "QueryError",
    "DataExtractionError",
    "RateLimitError",

    # Entity Resolution Exceptions
    "EntityResolutionError",
    "NoMatchFoundError",
    "MultipleMatchesError",
    "MDMLookupError",
    "FuzzyMatchError",
    "ConfidenceThresholdError",

    # Review Queue Exceptions
    "ReviewQueueError",
    "QueueItemNotFoundError",
    "InvalidActionError",
    "ReviewAlreadyCompletedError",
    "QueuePersistenceError",

    # Data Quality Exceptions
    "DataQualityError",
    "DQICalculationError",
    "CompletenessCheckError",
    "ValidationError",
    "GapAnalysisError",

    # Configuration Exceptions
    "ConfigurationError",
    "MissingConfigError",
    "InvalidConfigValueError",

    # Ingestion Exceptions
    "IngestionError",
    "BatchProcessingError",
    "RecordValidationError",
    "TenantIsolationError",
]
