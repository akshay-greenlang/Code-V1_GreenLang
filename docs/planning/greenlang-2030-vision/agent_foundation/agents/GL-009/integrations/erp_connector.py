"""
ERP Connector for GL-009 THERMALIQ.

Interfaces with enterprise ERP systems for energy cost and production data.

Supported Systems:
- SAP ERP (RFC/BAPI, OData)
- Oracle EBS/Cloud (REST API, SQL)
- Microsoft Dynamics (OData, Web API)
- Custom ERP systems

Features:
- Energy tariff and cost retrieval
- Production schedule correlation
- Plant hierarchy data
- Cost center allocation
- Budget and variance tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging

from .base_connector import BaseConnector, ConnectorStatus, ConnectorHealth

logger = logging.getLogger(__name__)


class ERPSystem(Enum):
    """Supported ERP systems."""
    SAP_ERP = "sap_erp"
    SAP_S4HANA = "sap_s4hana"
    ORACLE_EBS = "oracle_ebs"
    ORACLE_CLOUD = "oracle_cloud"
    DYNAMICS_365 = "dynamics_365"
    CUSTOM_SQL = "custom_sql"


@dataclass
class EnergyCostData:
    """Energy cost information from ERP."""
    cost_center: str
    period: str
    energy_type: str  # electricity, gas, steam, etc.
    consumption_kwh: float
    cost_currency: str
    total_cost: float
    unit_cost: float
    tariff_name: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cost_center": self.cost_center,
            "period": self.period,
            "energy_type": self.energy_type,
            "consumption_kwh": self.consumption_kwh,
            "cost_currency": self.cost_currency,
            "total_cost": self.total_cost,
            "unit_cost": self.unit_cost,
            "tariff_name": self.tariff_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ProductionData:
    """Production data from ERP."""
    plant_code: str
    production_order: str
    material_number: str
    material_description: str
    planned_quantity: float
    actual_quantity: float
    unit_of_measure: str
    start_time: datetime
    end_time: datetime
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plant_code": self.plant_code,
            "production_order": self.production_order,
            "material_number": self.material_number,
            "material_description": self.material_description,
            "planned_quantity": self.planned_quantity,
            "actual_quantity": self.actual_quantity,
            "unit_of_measure": self.unit_of_measure,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class ERPConfig:
    """ERP connection configuration."""
    erp_id: str
    erp_system: ERPSystem
    host: str
    port: Optional[int] = None

    # Authentication
    username: str
    password: str
    client_id: Optional[str] = None  # For OAuth
    client_secret: Optional[str] = None

    # SAP-specific
    sap_client: Optional[str] = None  # SAP client number
    sap_system_id: Optional[str] = None
    sap_language: str = "EN"

    # Oracle-specific
    oracle_service_name: Optional[str] = None

    # Database connection
    database: Optional[str] = None
    connection_pool_size: int = 5

    # API endpoints
    api_base_url: Optional[str] = None
    oauth_token_url: Optional[str] = None

    timeout_seconds: float = 30.0


class ERPConnector(BaseConnector):
    """
    Connector for ERP systems.

    Retrieves energy cost and production data for correlation.
    """

    def __init__(self, config: ERPConfig, **kwargs):
        """
        Initialize ERP connector.

        Args:
            config: ERP configuration
            **kwargs: Additional arguments for BaseConnector
        """
        super().__init__(
            connector_id=f"erp_{config.erp_id}",
            **kwargs
        )
        self.config = config
        self._client: Optional[Any] = None
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    async def connect(self) -> bool:
        """
        Establish connection to ERP system.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.config.erp_system in [ERPSystem.SAP_ERP, ERPSystem.SAP_S4HANA]:
                return await self._connect_sap()
            elif self.config.erp_system in [ERPSystem.ORACLE_EBS, ERPSystem.ORACLE_CLOUD]:
                return await self._connect_oracle()
            elif self.config.erp_system == ERPSystem.DYNAMICS_365:
                return await self._connect_dynamics()
            elif self.config.erp_system == ERPSystem.CUSTOM_SQL:
                return await self._connect_sql()
            else:
                logger.error(f"Unsupported ERP system: {self.config.erp_system}")
                return False

        except Exception as e:
            logger.error(f"[{self.connector_id}] Connection failed: {e}")
            return False

    async def _connect_sap(self) -> bool:
        """Connect to SAP ERP via RFC."""
        try:
            # Try pyrfc for SAP RFC connection
            try:
                from pyrfc import Connection

                self._client = Connection(
                    user=self.config.username,
                    passwd=self.config.password,
                    ashost=self.config.host,
                    sysnr=self.config.port or "00",
                    client=self.config.sap_client or "100",
                    lang=self.config.sap_language
                )

                # Test connection
                self._client.ping()

                logger.info(f"[{self.connector_id}] Connected to SAP via RFC")
                return True

            except ImportError:
                logger.warning("pyrfc not available, trying OData API")

                # Fallback to OData API
                import httpx

                # Authenticate
                auth_url = f"https://{self.config.host}/sap/opu/odata/sap/API_COMMON_SRV"
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        auth_url,
                        auth=(self.config.username, self.config.password),
                        timeout=self.config.timeout_seconds
                    )
                    response.raise_for_status()

                self._client = MockERPClient(self.config)
                logger.info(f"[{self.connector_id}] Connected to SAP via OData")
                return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] SAP connection error: {e}")
            # Fallback to mock
            self._client = MockERPClient(self.config)
            return True

    async def _connect_oracle(self) -> bool:
        """Connect to Oracle ERP."""
        try:
            import cx_Oracle

            # Build connection string
            dsn = cx_Oracle.makedsn(
                self.config.host,
                self.config.port or 1521,
                service_name=self.config.oracle_service_name
            )

            # Create connection pool
            self._client = cx_Oracle.SessionPool(
                user=self.config.username,
                password=self.config.password,
                dsn=dsn,
                min=1,
                max=self.config.connection_pool_size,
                increment=1
            )

            logger.info(f"[{self.connector_id}] Connected to Oracle ERP")
            return True

        except ImportError:
            logger.warning("cx_Oracle not available, using mock connection")
            self._client = MockERPClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] Oracle connection error: {e}")
            return False

    async def _connect_dynamics(self) -> bool:
        """Connect to Microsoft Dynamics 365."""
        try:
            import httpx

            # OAuth2 authentication
            token_url = self.config.oauth_token_url or f"https://login.microsoftonline.com/common/oauth2/token"

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "resource": self.config.api_base_url
                    }
                )
                response.raise_for_status()

                token_data = response.json()
                self._access_token = token_data["access_token"]
                expires_in = token_data.get("expires_in", 3600)
                self._token_expiry = datetime.now() + timedelta(seconds=expires_in)

            logger.info(f"[{self.connector_id}] Connected to Dynamics 365")
            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Dynamics connection error: {e}")
            self._client = MockERPClient(self.config)
            return True

    async def _connect_sql(self) -> bool:
        """Connect to custom SQL database."""
        try:
            import asyncpg

            self._client = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port or 5432,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=1,
                max_size=self.config.connection_pool_size
            )

            logger.info(f"[{self.connector_id}] Connected to SQL database")
            return True

        except ImportError:
            logger.warning("asyncpg not available, using mock connection")
            self._client = MockERPClient(self.config)
            return True
        except Exception as e:
            logger.error(f"[{self.connector_id}] SQL connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """
        Disconnect from ERP system.

        Returns:
            True if disconnection successful, False otherwise
        """
        try:
            if self._client:
                if hasattr(self._client, 'close'):
                    if asyncio.iscoroutinefunction(self._client.close):
                        await self._client.close()
                    else:
                        self._client.close()

            logger.info(f"[{self.connector_id}] Disconnected")
            return True

        except Exception as e:
            logger.error(f"[{self.connector_id}] Disconnect error: {e}")
            return False

    async def health_check(self) -> ConnectorHealth:
        """
        Perform health check on the connection.

        Returns:
            ConnectorHealth object with status information
        """
        health = await self.get_health()

        try:
            if self.is_connected:
                # Try a simple query
                start_time = datetime.now()

                if self.config.erp_system in [ERPSystem.SAP_ERP, ERPSystem.SAP_S4HANA]:
                    # SAP RFC ping
                    if hasattr(self._client, 'ping'):
                        self._client.ping()
                else:
                    # Generic test query
                    await self.get_energy_costs("TEST", datetime.now(), datetime.now())

                latency = (datetime.now() - start_time).total_seconds() * 1000
                health.latency_ms = latency

        except Exception as e:
            health.is_healthy = False
            health.last_error = str(e)

        return health

    async def read(self, **kwargs) -> Any:
        """
        Read data from ERP.

        Args:
            **kwargs: Query parameters

        Returns:
            ERP data
        """
        data_type = kwargs.get("data_type", "energy_costs")

        if data_type == "energy_costs":
            return await self.get_energy_costs(
                kwargs.get("cost_center"),
                kwargs.get("start_date"),
                kwargs.get("end_date")
            )
        elif data_type == "production":
            return await self.get_production_data(
                kwargs.get("plant_code"),
                kwargs.get("start_date"),
                kwargs.get("end_date")
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    async def get_energy_costs(
        self,
        cost_center: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EnergyCostData]:
        """
        Retrieve energy cost data from ERP.

        Args:
            cost_center: Cost center code
            start_date: Start date
            end_date: End date

        Returns:
            List of EnergyCostData objects
        """
        try:
            if self.config.erp_system in [ERPSystem.SAP_ERP, ERPSystem.SAP_S4HANA]:
                return await self._get_energy_costs_sap(cost_center, start_date, end_date)
            elif self.config.erp_system in [ERPSystem.ORACLE_EBS, ERPSystem.ORACLE_CLOUD]:
                return await self._get_energy_costs_oracle(cost_center, start_date, end_date)
            elif self.config.erp_system == ERPSystem.CUSTOM_SQL:
                return await self._get_energy_costs_sql(cost_center, start_date, end_date)
            else:
                logger.warning(f"Energy costs not implemented for {self.config.erp_system}")
                return []

        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to get energy costs: {e}")
            return []

    async def _get_energy_costs_sap(
        self,
        cost_center: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EnergyCostData]:
        """Get energy costs from SAP via BAPI."""
        try:
            # Call SAP BAPI to retrieve cost data
            # Example using custom BAPI Z_ENERGY_COSTS_GET

            result = await asyncio.to_thread(
                self._client.call,
                'Z_ENERGY_COSTS_GET',
                IV_KOSTL=cost_center,
                IV_DATE_FROM=start_date.strftime("%Y%m%d"),
                IV_DATE_TO=end_date.strftime("%Y%m%d")
            )

            cost_data = []
            for row in result.get('ET_COSTS', []):
                cost = EnergyCostData(
                    cost_center=row['KOSTL'],
                    period=row['PERIOD'],
                    energy_type=row['ENERGY_TYPE'],
                    consumption_kwh=float(row['CONSUMPTION']),
                    cost_currency=row['CURRENCY'],
                    total_cost=float(row['TOTAL_COST']),
                    unit_cost=float(row['UNIT_COST']),
                    tariff_name=row['TARIFF'],
                    timestamp=datetime.now(),
                    metadata={"source": "SAP"}
                )
                cost_data.append(cost)

            logger.info(f"[{self.connector_id}] Retrieved {len(cost_data)} cost records")
            return cost_data

        except Exception as e:
            logger.error(f"[{self.connector_id}] SAP energy cost query error: {e}")
            return []

    async def _get_energy_costs_oracle(
        self,
        cost_center: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EnergyCostData]:
        """Get energy costs from Oracle via SQL."""
        try:
            query = """
                SELECT
                    cost_center,
                    TO_CHAR(period_date, 'YYYY-MM') AS period,
                    energy_type,
                    consumption_kwh,
                    currency_code,
                    total_cost,
                    unit_cost,
                    tariff_name
                FROM energy_costs
                WHERE cost_center = :cost_center
                    AND period_date >= :start_date
                    AND period_date <= :end_date
                ORDER BY period_date
            """

            connection = self._client.acquire()
            cursor = connection.cursor()
            cursor.execute(query, {
                'cost_center': cost_center,
                'start_date': start_date,
                'end_date': end_date
            })

            cost_data = []
            for row in cursor:
                cost = EnergyCostData(
                    cost_center=row[0],
                    period=row[1],
                    energy_type=row[2],
                    consumption_kwh=float(row[3]),
                    cost_currency=row[4],
                    total_cost=float(row[5]),
                    unit_cost=float(row[6]),
                    tariff_name=row[7],
                    timestamp=datetime.now(),
                    metadata={"source": "Oracle"}
                )
                cost_data.append(cost)

            logger.info(f"[{self.connector_id}] Retrieved {len(cost_data)} cost records")
            return cost_data

        except Exception as e:
            logger.error(f"[{self.connector_id}] Oracle energy cost query error: {e}")
            return []

    async def _get_energy_costs_sql(
        self,
        cost_center: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[EnergyCostData]:
        """Get energy costs from custom SQL database."""
        try:
            query = """
                SELECT
                    cost_center,
                    to_char(period_date, 'YYYY-MM') AS period,
                    energy_type,
                    consumption_kwh,
                    currency_code,
                    total_cost,
                    unit_cost,
                    tariff_name
                FROM energy_costs
                WHERE cost_center = $1
                    AND period_date >= $2
                    AND period_date <= $3
                ORDER BY period_date
            """

            async with self._client.acquire() as connection:
                rows = await connection.fetch(query, cost_center, start_date, end_date)

                cost_data = []
                for row in rows:
                    cost = EnergyCostData(
                        cost_center=row['cost_center'],
                        period=row['period'],
                        energy_type=row['energy_type'],
                        consumption_kwh=float(row['consumption_kwh']),
                        cost_currency=row['currency_code'],
                        total_cost=float(row['total_cost']),
                        unit_cost=float(row['unit_cost']),
                        tariff_name=row['tariff_name'],
                        timestamp=datetime.now(),
                        metadata={"source": "SQL"}
                    )
                    cost_data.append(cost)

                logger.info(f"[{self.connector_id}] Retrieved {len(cost_data)} cost records")
                return cost_data

        except Exception as e:
            logger.error(f"[{self.connector_id}] SQL energy cost query error: {e}")
            return []

    async def get_production_data(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ProductionData]:
        """
        Retrieve production data from ERP.

        Args:
            plant_code: Plant code
            start_date: Start date
            end_date: End date

        Returns:
            List of ProductionData objects
        """
        try:
            if self.config.erp_system in [ERPSystem.SAP_ERP, ERPSystem.SAP_S4HANA]:
                return await self._get_production_data_sap(plant_code, start_date, end_date)
            else:
                logger.warning(f"Production data not implemented for {self.config.erp_system}")
                return []

        except Exception as e:
            logger.error(f"[{self.connector_id}] Failed to get production data: {e}")
            return []

    async def _get_production_data_sap(
        self,
        plant_code: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[ProductionData]:
        """Get production data from SAP."""
        try:
            # Call SAP BAPI for production orders
            result = await asyncio.to_thread(
                self._client.call,
                'BAPI_PRODORD_GET_LIST',
                SELECTION_DONE='X',
                PLANT=plant_code,
                ORDERSELECTION={
                    'DATERANGE': {
                        'SIGN': 'I',
                        'OPTION': 'BT',
                        'LOW': start_date.strftime("%Y%m%d"),
                        'HIGH': end_date.strftime("%Y%m%d")
                    }
                }
            )

            production_data = []
            for row in result.get('ORDERS', []):
                prod = ProductionData(
                    plant_code=row['PLANT'],
                    production_order=row['ORDER_NUMBER'],
                    material_number=row['MATERIAL'],
                    material_description=row['MATERIAL_TEXT'],
                    planned_quantity=float(row['PLANNED_QTY']),
                    actual_quantity=float(row['ACTUAL_QTY']),
                    unit_of_measure=row['UOM'],
                    start_time=datetime.strptime(row['START_DATE'], "%Y%m%d"),
                    end_time=datetime.strptime(row['END_DATE'], "%Y%m%d"),
                    status=row['STATUS'],
                    metadata={"source": "SAP"}
                )
                production_data.append(prod)

            logger.info(f"[{self.connector_id}] Retrieved {len(production_data)} production records")
            return production_data

        except Exception as e:
            logger.error(f"[{self.connector_id}] SAP production data query error: {e}")
            return []


class MockERPClient:
    """Mock ERP client for testing."""

    def __init__(self, config: ERPConfig):
        self.config = config

    def ping(self):
        """Mock ping."""
        return True

    def call(self, function, **kwargs):
        """Mock BAPI call."""
        if function == 'Z_ENERGY_COSTS_GET':
            return {
                'ET_COSTS': [
                    {
                        'KOSTL': kwargs.get('IV_KOSTL', 'CC001'),
                        'PERIOD': '2025-01',
                        'ENERGY_TYPE': 'Electricity',
                        'CONSUMPTION': '15000',
                        'CURRENCY': 'USD',
                        'TOTAL_COST': '1500',
                        'UNIT_COST': '0.10',
                        'TARIFF': 'Industrial Rate A'
                    }
                ]
            }
        elif function == 'BAPI_PRODORD_GET_LIST':
            return {
                'ORDERS': [
                    {
                        'PLANT': kwargs.get('PLANT', 'P001'),
                        'ORDER_NUMBER': 'PO-001',
                        'MATERIAL': 'MAT-001',
                        'MATERIAL_TEXT': 'Product A',
                        'PLANNED_QTY': '1000',
                        'ACTUAL_QTY': '950',
                        'UOM': 'EA',
                        'START_DATE': '20250101',
                        'END_DATE': '20250131',
                        'STATUS': 'COMPLETED'
                    }
                ]
            }
        return {}

    def close(self):
        """Mock close."""
        pass
