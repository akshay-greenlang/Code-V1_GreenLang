"""
Example 9: Database Integration
================================

Demonstrates database operations with DatabaseManager.
"""

import asyncio
import pandas as pd
from greenlang.db import DatabaseManager, Model, Column, String, Float, Integer


async def main():
    """Run database example."""
    # Initialize database (SQLite for example)
    db = DatabaseManager(connection_string="sqlite:///example.db")

    # Create a table model
    class EmissionsRecord(Model):
        __tablename__ = "emissions"

        id = Column(Integer, primary_key=True)
        facility = Column(String(255), nullable=False)
        emissions = Column(Float, nullable=False)
        year = Column(Integer, nullable=False)

    # Create tables
    await db.create_tables([EmissionsRecord])
    print("Created database table: emissions")

    # Insert data from DataFrame
    data = pd.DataFrame({
        "facility": ["Plant A", "Plant B", "Plant C"],
        "emissions": [1500.5, 2300.8, 1800.2],
        "year": [2024, 2024, 2024]
    })

    await db.store_dataframe(
        data,
        table_name="emissions",
        if_exists="append"
    )
    print(f"\nInserted {len(data)} records")

    # Query data
    query = "SELECT facility, emissions FROM emissions WHERE year = 2024"
    results = await db.execute_query(query)

    print(f"\nQuery results:")
    for row in results:
        print(f"  {row['facility']}: {row['emissions']} kg CO2e")

    # Load as DataFrame
    df = await db.load_dataframe("emissions")
    print(f"\nLoaded DataFrame:")
    print(df)

    # Close connection
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
