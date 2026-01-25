"""
External Integrations Module.

This package provides integrations with external systems:
- ERP Connectors (SAP, Oracle)
- File Parsers (CSV, Excel, XML, JSON)
- Industrial Protocol Connectors (OPC-UA, Modbus, MQTT)
- Process Historians (PI, Aveva, InfluxDB)
- DCS Systems (Honeywell, Emerson, ABB)

Example:
    >>> from integrations import SAPConnector, FileParserFactory
    >>>
    >>> # ERP Integration
    >>> connector = SAPConnector(config)
    >>> await connector.connect()
    >>> records = await connector.fetch_emissions_data(...)
    >>>
    >>> # File Parsing
    >>> parser = FileParserFactory.get_parser("data.xlsx")
    >>> result = await parser.parse("data.xlsx")
    >>>
    >>> # Industrial Protocol Integration
    >>> from integrations.industrial import OPCUAConnector, OPCUAConfig
    >>> opcua = OPCUAConnector(OPCUAConfig(endpoint_url="opc.tcp://localhost:4840"))
    >>> await opcua.connect()
    >>> values = await opcua.read_tags(["ns=2;s=Temperature"])
"""

from .erp import (
    BaseERPConnector,
    ConnectionConfig,
    ConnectionStatus,
    DataQuery,
    ERPRecord,
    ERPType,
    SAPConnector,
    SAPConfig,
    OracleConnector,
    OracleConfig,
    get_connector,
)

from .parsers import (
    BaseFileParser,
    CSVParser,
    ExcelParser,
    XMLParser,
    JSONParser,
    FileParserFactory,
    FileFormat,
    ParseResult,
    ParsedRecord,
    ParserConfig,
    ColumnMapping,
)

# Industrial Protocol Connectors
from .industrial import (
    # Data Models
    DataQuality,
    TagValue,
    TagMetadata,
    AlarmEvent,
    HistoricalQuery,
    HistoricalResult,
    BatchReadRequest,
    BatchReadResponse,
    BatchWriteRequest,
    BatchWriteResponse,
    SubscriptionConfig,
    ConnectionState,
    ConnectionMetrics,
    # Base
    BaseIndustrialConnector,
    BaseConnectorConfig,
    TLSConfig,
    # OPC-UA
    OPCUAConnector,
    OPCUAConfig,
    # Modbus
    ModbusConnector,
    ModbusTCPConfig,
    ModbusRTUConfig,
    # MQTT
    MQTTConnector,
    MQTTConfig,
    QoSLevel,
    # Historians
    PIConnector,
    PIConfig,
    AvevaConnector,
    AvevaConfig,
    InfluxDBConnector,
    InfluxDBConfig,
    get_historian_connector,
    # DCS
    ExperionConnector,
    ExperionConfig,
    DeltaVConnector,
    DeltaVConfig,
    ABB800xAConnector,
    ABB800xAConfig,
    get_dcs_connector,
)

__all__ = [
    # ERP
    "BaseERPConnector",
    "ConnectionConfig",
    "ConnectionStatus",
    "DataQuery",
    "ERPRecord",
    "ERPType",
    "SAPConnector",
    "SAPConfig",
    "OracleConnector",
    "OracleConfig",
    "get_connector",
    # Parsers
    "BaseFileParser",
    "CSVParser",
    "ExcelParser",
    "XMLParser",
    "JSONParser",
    "FileParserFactory",
    "FileFormat",
    "ParseResult",
    "ParsedRecord",
    "ParserConfig",
    "ColumnMapping",
    # Industrial Data Models
    "DataQuality",
    "TagValue",
    "TagMetadata",
    "AlarmEvent",
    "HistoricalQuery",
    "HistoricalResult",
    "BatchReadRequest",
    "BatchReadResponse",
    "BatchWriteRequest",
    "BatchWriteResponse",
    "SubscriptionConfig",
    "ConnectionState",
    "ConnectionMetrics",
    # Industrial Base
    "BaseIndustrialConnector",
    "BaseConnectorConfig",
    "TLSConfig",
    # OPC-UA
    "OPCUAConnector",
    "OPCUAConfig",
    # Modbus
    "ModbusConnector",
    "ModbusTCPConfig",
    "ModbusRTUConfig",
    # MQTT
    "MQTTConnector",
    "MQTTConfig",
    "QoSLevel",
    # Historians
    "PIConnector",
    "PIConfig",
    "AvevaConnector",
    "AvevaConfig",
    "InfluxDBConnector",
    "InfluxDBConfig",
    "get_historian_connector",
    # DCS
    "ExperionConnector",
    "ExperionConfig",
    "DeltaVConnector",
    "DeltaVConfig",
    "ABB800xAConnector",
    "ABB800xAConfig",
    "get_dcs_connector",
]
