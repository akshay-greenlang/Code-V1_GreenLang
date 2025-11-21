"""
Example usage of GreenLang's critical data infrastructure components.

Demonstrates:
1. Transaction management with automatic rollback
2. Dead Letter Queue for failed records
3. Data validation with decorators
4. Deduplication utilities
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

# Import GreenLang data infrastructure components
from greenlang.database.transaction import TransactionManager, IsolationLevel, transactional
from greenlang.data.dead_letter_queue import (
    DeadLetterQueue,
    FailureReason,
    ReprocessingStrategy,
    get_dlq
)
from greenlang.data.validation import (
    validate_input,
    validate_output,
    validate_type_hints,
    DataValidator,
    ValidationLevel,
    ShipmentSchema,
    SupplierDataSchema,
    PipelineValidator
)
from greenlang.data.deduplication import (
    DataDeduplicator,
    DeduplicationStrategy,
    DuplicateAction,
    ConnectorDeduplicator,
    HashGenerator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Transaction Management
# ============================================================================

class EmissionsDatabase:
    """Example database class with transaction support."""

    def __init__(self, connection):
        """Initialize with database connection."""
        self.connection = connection
        self._transaction_manager = TransactionManager(
            db_connection=connection,
            isolation_level=IsolationLevel.READ_COMMITTED,
            max_retries=3
        )

    @transactional(isolation_level=IsolationLevel.SERIALIZABLE)
    def update_emissions_batch(self, emissions_data: List[Dict[str, Any]]):
        """
        Update emissions data in a transaction.
        Automatically rolls back on any failure.
        """
        cursor = self.connection.cursor()

        for record in emissions_data:
            # Update emissions table
            cursor.execute(
                "UPDATE emissions SET value = ?, updated_at = ? WHERE id = ?",
                (record['value'], datetime.now(), record['id'])
            )

            # Update audit log
            cursor.execute(
                "INSERT INTO audit_log (entity_id, action, timestamp, data) VALUES (?, ?, ?, ?)",
                (record['id'], 'UPDATE', datetime.now(), str(record))
            )

            # This could fail and trigger automatic rollback
            if record['value'] < 0:
                raise ValueError(f"Invalid negative emission value: {record['value']}")

        return {"status": "success", "updated": len(emissions_data)}


def example_transaction_usage():
    """Demonstrate transaction management."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Transaction Management")
    print("="*60)

    # Mock database connection
    class MockConnection:
        def cursor(self):
            return MockCursor()

    class MockCursor:
        def execute(self, query, params=None):
            logger.info(f"Executing: {query[:50]}...")

    connection = MockConnection()
    db = EmissionsDatabase(connection)

    # Example: Successful transaction
    try:
        with db._transaction_manager.transaction("emissions_update") as tx:
            tx.execute("UPDATE emissions SET value = 100 WHERE id = 1")
            tx.execute("INSERT INTO audit_log VALUES ('UPDATE', '2024-01-01')")
            print("✓ Transaction completed successfully")
    except Exception as e:
        print(f"✗ Transaction failed: {e}")

    # Example: Failed transaction with rollback
    try:
        with db._transaction_manager.transaction("failed_update") as tx:
            tx.execute("UPDATE emissions SET value = 200 WHERE id = 2")
            # Simulate error
            raise ValueError("Invalid data detected!")
    except Exception as e:
        print(f"✓ Transaction rolled back due to: {e}")

    # Get transaction metrics
    metrics = db._transaction_manager.get_metrics()
    print(f"\nTransaction Metrics: {metrics}")


# ============================================================================
# EXAMPLE 2: Dead Letter Queue
# ============================================================================

def example_dlq_usage():
    """Demonstrate Dead Letter Queue functionality."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Dead Letter Queue")
    print("="*60)

    # Initialize DLQ
    dlq = DeadLetterQueue(
        max_queue_size=1000,
        alert_threshold=50,
        auto_retry=True
    )

    # Register reprocessing handler
    def reprocess_transformation(data, metadata):
        """Custom reprocessing logic for transformation failures."""
        logger.info(f"Reprocessing transformation for: {data.get('id')}")
        # Apply more lenient transformation
        return {"status": "reprocessed", "data": data}

    dlq.register_reprocessing_handler("transformation", reprocess_transformation)

    # Example: Quarantine failed records
    failed_records = [
        {
            "id": "REC001",
            "value": -100,  # Invalid negative value
            "timestamp": datetime.now()
        },
        {
            "id": "REC002",
            "value": "not_a_number",  # Type error
            "timestamp": datetime.now()
        }
    ]

    for record in failed_records:
        try:
            # Simulate processing that fails
            if record['value'] < 0:
                raise ValueError("Negative values not allowed")
            elif not isinstance(record['value'], (int, float)):
                raise TypeError("Value must be numeric")
        except Exception as e:
            # Quarantine to DLQ
            record_id = dlq.quarantine_record(
                record=record,
                error=e,
                pipeline_stage="transformation",
                metadata={"source": "emissions_pipeline", "batch_id": "BATCH001"},
                reprocessing_strategy=ReprocessingStrategy.EXPONENTIAL_BACKOFF
            )
            print(f"✓ Quarantined record: {record_id}")

    # Get failed records
    failed = dlq.get_failed_records(
        failure_reason=FailureReason.VALIDATION_ERROR,
        limit=10
    )
    print(f"\nFailed records in DLQ: {len(failed)}")

    # Attempt reprocessing
    result = dlq.reprocess(
        pipeline_stage="transformation",
        max_records=5
    )
    print(f"Reprocessing result: {result}")

    # Get DLQ statistics
    stats = dlq.get_statistics()
    print(f"\nDLQ Statistics: {stats}")


# ============================================================================
# EXAMPLE 3: Data Validation
# ============================================================================

# Define validation schemas
from pydantic import BaseModel, Field

class EmissionsDataSchema(BaseModel):
    """Schema for emissions data validation."""
    emission_id: str = Field(..., min_length=1)
    source_type: str
    value: float = Field(..., ge=0)  # Must be non-negative
    unit: str
    timestamp: datetime
    confidence: float = Field(..., ge=0, le=1)


# Apply validation decorators
@validate_input(
    schema=EmissionsDataSchema,
    validation_level=ValidationLevel.STRICT
)
def process_emissions_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process emissions data with strict validation."""
    logger.info(f"Processing validated emissions data: {data['emission_id']}")

    # Calculate adjusted value
    adjusted_value = data['value'] * data['confidence']

    return {
        "emission_id": data['emission_id'],
        "adjusted_value": adjusted_value,
        "processing_timestamp": datetime.now()
    }


@validate_type_hints(validation_level=ValidationLevel.STRICT)
def calculate_carbon_footprint(
    distance: float,
    weight: float,
    transport_mode: str,
    efficiency_factor: float = 1.0
) -> float:
    """Calculate carbon footprint with type validation."""
    # Type hints are automatically validated
    base_emission = distance * weight * 0.001  # Simplified calculation
    return base_emission * efficiency_factor


def example_validation_usage():
    """Demonstrate data validation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Data Validation")
    print("="*60)

    # Example: Valid data passes validation
    valid_data = {
        "emission_id": "EM001",
        "source_type": "transportation",
        "value": 150.5,
        "unit": "kgCO2e",
        "timestamp": datetime.now(),
        "confidence": 0.95
    }

    try:
        result = process_emissions_data(valid_data)
        print(f"✓ Valid data processed: {result}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")

    # Example: Invalid data fails validation
    invalid_data = {
        "emission_id": "",  # Empty ID
        "source_type": "transportation",
        "value": -50,  # Negative value
        "unit": "kgCO2e",
        "timestamp": datetime.now(),
        "confidence": 1.5  # Out of range
    }

    try:
        result = process_emissions_data(invalid_data)
        print(f"✓ Invalid data processed: {result}")
    except Exception as e:
        print(f"✓ Validation correctly rejected invalid data: {e}")

    # Example: Type validation
    try:
        footprint = calculate_carbon_footprint(
            distance=100.5,
            weight=1000,
            transport_mode="truck"
        )
        print(f"✓ Carbon footprint calculated: {footprint} kgCO2e")
    except Exception as e:
        print(f"✗ Type validation failed: {e}")

    # Pipeline validator for multiple schemas
    pipeline_validator = PipelineValidator(
        schemas={
            "shipment": ShipmentSchema,
            "supplier": SupplierDataSchema,
            "emissions": EmissionsDataSchema
        }
    )

    # Validate shipment data
    shipment_data = {
        "shipment_id": "SHIP001",
        "origin": "New York",
        "destination": "Los Angeles",
        "weight": 1000.0,
        "transport_mode": "road",
        "departure_date": datetime.now()
    }

    validation_result = pipeline_validator.validate_entry(
        data=shipment_data,
        data_type="shipment"
    )
    print(f"\nPipeline validation result: Valid={validation_result.is_valid}")


# ============================================================================
# EXAMPLE 4: Data Deduplication
# ============================================================================

class SAPConnectorDeduplicator(ConnectorDeduplicator):
    """SAP-specific deduplication implementation."""

    def get_strategy(self) -> DeduplicationStrategy:
        """Use key-based strategy for SAP data."""
        return DeduplicationStrategy.KEY_BASED

    def get_cache_size(self) -> int:
        """Large cache for enterprise data."""
        return 50000

    def get_dedup_keys(self, record_type: str) -> List[str]:
        """Get deduplication keys for SAP record types."""
        if record_type == "purchase_order":
            return ["po_number", "vendor_id", "plant_code"]
        elif record_type == "material":
            return ["material_id", "plant_code"]
        elif record_type == "invoice":
            return ["invoice_number", "vendor_id"]
        else:
            return ["id"]


def example_deduplication_usage():
    """Demonstrate data deduplication."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Data Deduplication")
    print("="*60)

    # Initialize deduplicator
    dedup = DataDeduplicator(
        strategy=DeduplicationStrategy.KEY_BASED,
        cache_size=10000,
        similarity_threshold=0.85
    )

    # Example: Check for duplicates
    shipments = [
        {"shipment_id": "SH001", "origin": "NYC", "weight": 100},
        {"shipment_id": "SH002", "origin": "LAX", "weight": 200},
        {"shipment_id": "SH001", "origin": "NYC", "weight": 100},  # Duplicate
        {"shipment_id": "SH003", "origin": "CHI", "weight": 150},
        {"shipment_id": "SH001", "origin": "NYC", "weight": 110},  # Same ID, different weight
    ]

    for shipment in shipments:
        result = dedup.is_duplicate(
            data=shipment,
            keys=["shipment_id"]
        )

        if result.is_duplicate:
            print(f"✓ Duplicate detected: {shipment['shipment_id']} "
                  f"(Action: {result.suggested_action.value})")
        else:
            print(f"  Unique record: {shipment['shipment_id']}")

    # Batch deduplication
    unique, duplicates = dedup.batch_deduplicate(
        records=shipments,
        keys=["shipment_id"],
        action=DuplicateAction.UPDATE
    )

    print(f"\nBatch deduplication: {len(unique)} unique, {len(duplicates)} duplicates")

    # Hash generation examples
    hash_gen = HashGenerator()

    # Generate composite hash
    composite_hash = hash_gen.generate_composite_hash(
        data=shipments[0],
        hash_fields={
            "primary": ["shipment_id"],
            "secondary": ["origin", "weight"]
        }
    )
    print(f"\nComposite hash: {composite_hash}")

    # SAP connector deduplication
    sap_dedup = SAPConnectorDeduplicator()

    purchase_orders = [
        {"po_number": "PO001", "vendor_id": "V001", "plant_code": "P001", "amount": 1000},
        {"po_number": "PO001", "vendor_id": "V001", "plant_code": "P001", "amount": 1000},  # Duplicate
        {"po_number": "PO002", "vendor_id": "V002", "plant_code": "P001", "amount": 2000},
    ]

    unique_pos, dup_count = sap_dedup.deduplicate_batch(
        records=purchase_orders,
        record_type="purchase_order"
    )
    print(f"\nSAP PO deduplication: {len(unique_pos)} unique, {dup_count} duplicates")

    # Get deduplication statistics
    stats = dedup.get_statistics()
    print(f"\nDeduplication statistics: {stats}")


# ============================================================================
# EXAMPLE 5: Integrated Pipeline with All Components
# ============================================================================

class RobustDataPipeline:
    """
    Example of a robust data pipeline using all infrastructure components.
    """

    def __init__(self):
        """Initialize pipeline with all components."""
        # Transaction manager
        self.tx_manager = TransactionManager(
            db_connection=MockConnection(),
            isolation_level=IsolationLevel.READ_COMMITTED
        )

        # Dead Letter Queue
        self.dlq = get_dlq()

        # Validator
        self.validator = DataValidator(level=ValidationLevel.STRICT)

        # Deduplicator
        self.deduplicator = DataDeduplicator(
            strategy=DeduplicationStrategy.COMPOSITE,
            cache_size=10000
        )

    @validate_input(schema=ShipmentSchema)
    def process_shipment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process shipment with full data integrity protection.
        """
        try:
            # Step 1: Check for duplicates
            dedup_result = self.deduplicator.is_duplicate(
                data=data,
                keys=["shipment_id"]
            )

            if dedup_result.is_duplicate:
                logger.warning(f"Duplicate shipment detected: {data['shipment_id']}")
                if dedup_result.suggested_action == DuplicateAction.SKIP:
                    return {"status": "skipped", "reason": "duplicate"}

            # Step 2: Process in transaction
            with self.tx_manager.transaction("shipment_processing") as tx:
                # Insert shipment
                tx.execute(
                    "INSERT INTO shipments (id, origin, destination, weight) VALUES (?, ?, ?, ?)",
                    (data['shipment_id'], data['origin'], data['destination'], data['weight'])
                )

                # Calculate emissions
                emissions = self._calculate_emissions(data)

                # Insert emissions
                tx.execute(
                    "INSERT INTO emissions (shipment_id, value, unit) VALUES (?, ?, ?)",
                    (data['shipment_id'], emissions['value'], emissions['unit'])
                )

            return {
                "status": "success",
                "shipment_id": data['shipment_id'],
                "emissions": emissions
            }

        except Exception as e:
            # Step 3: Handle failure with DLQ
            record_id = self.dlq.quarantine_record(
                record=data,
                error=e,
                pipeline_stage="shipment_processing",
                metadata={"pipeline": "robust_data_pipeline"}
            )

            logger.error(f"Failed to process shipment, quarantined as: {record_id}")
            raise

    def _calculate_emissions(self, shipment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emissions for shipment."""
        # Simplified calculation
        base_emission = shipment['weight'] * 0.001  # kg to tons
        transport_factor = 2.5  # kgCO2e per ton-km

        return {
            "value": base_emission * transport_factor * 1000,  # Assume 1000km
            "unit": "kgCO2e"
        }


def example_integrated_pipeline():
    """Demonstrate integrated pipeline with all components."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Integrated Pipeline")
    print("="*60)

    pipeline = RobustDataPipeline()

    # Process multiple shipments
    shipments = [
        {
            "shipment_id": "INT001",
            "origin": "Boston",
            "destination": "Miami",
            "weight": 500.0,
            "transport_mode": "road",
            "departure_date": datetime.now()
        },
        {
            "shipment_id": "INT002",
            "origin": "Seattle",
            "destination": "Portland",
            "weight": 750.0,
            "transport_mode": "rail",
            "departure_date": datetime.now()
        },
        {
            "shipment_id": "INT001",  # Duplicate
            "origin": "Boston",
            "destination": "Miami",
            "weight": 500.0,
            "transport_mode": "road",
            "departure_date": datetime.now()
        }
    ]

    for shipment in shipments:
        try:
            result = pipeline.process_shipment(shipment)
            print(f"✓ Processed: {shipment['shipment_id']} - {result['status']}")
        except Exception as e:
            print(f"✗ Failed: {shipment['shipment_id']} - {e}")

    # Get pipeline statistics
    print("\nPipeline Statistics:")
    print(f"  Transaction Metrics: {pipeline.tx_manager.get_metrics()}")
    print(f"  DLQ Statistics: {pipeline.dlq.get_statistics()}")
    print(f"  Deduplication Stats: {pipeline.deduplicator.get_statistics()}")


# ============================================================================
# Mock Classes for Examples
# ============================================================================

class MockConnection:
    """Mock database connection for examples."""
    def cursor(self):
        return MockCursor()

    def acquire(self):
        return asyncio.create_task(self._acquire())

    async def _acquire(self):
        return self

    def release(self, conn):
        return asyncio.create_task(self._release())

    async def _release(self):
        pass

    def transaction(self):
        return MockTransaction()


class MockCursor:
    """Mock database cursor for examples."""
    def execute(self, query, params=None):
        logger.debug(f"Mock execute: {query[:50]}...")

    def close(self):
        pass


class MockTransaction:
    """Mock transaction for examples."""
    async def start(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GreenLang Data Infrastructure Examples")
    print("="*60)

    # Run all examples
    example_transaction_usage()
    example_dlq_usage()
    example_validation_usage()
    example_deduplication_usage()
    example_integrated_pipeline()

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)