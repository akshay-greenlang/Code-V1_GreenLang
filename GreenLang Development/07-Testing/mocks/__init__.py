"""
GreenLang Mock Services Package

Mock services for testing external dependencies.

Provides:
- MockEmissionFactorDB: Mock emission factor database
- MockERPConnector: Mock ERP system connector
- MockRegulatoryAPI: Mock regulatory data API
- MockCacheLayer: Mock cache layer
- MockProvenanceStore: Mock provenance storage
"""

from tests.mocks.mock_services import (
    MockEmissionFactorDB,
    MockERPConnector,
    MockRegulatoryAPI,
    MockCacheLayer,
    MockProvenanceStore,
    create_mock_emission_factor_db,
    create_mock_erp_connector,
    create_mock_regulatory_api,
    create_mock_cache,
    create_mock_provenance_store,
    get_mock_fixtures,
)

__all__ = [
    "MockEmissionFactorDB",
    "MockERPConnector",
    "MockRegulatoryAPI",
    "MockCacheLayer",
    "MockProvenanceStore",
    "create_mock_emission_factor_db",
    "create_mock_erp_connector",
    "create_mock_regulatory_api",
    "create_mock_cache",
    "create_mock_provenance_store",
    "get_mock_fixtures",
]
