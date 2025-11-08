# VCCI CLI Example Data Files

This directory contains sample data files for testing the VCCI CLI calculate command.

## File Descriptions

### Category 1 (Purchased Goods & Services)

- **sample_category1_single.json**: Single product calculation with Tier 1 supplier-specific PCF data
- **sample_category1_batch.csv**: Batch processing example with multiple products

### Category 4 (Upstream Transportation & Distribution)

- **sample_category4_transport.json**: Single transportation calculation (ISO 14083 compliant)

### Category 6 (Business Travel)

- **sample_category6_travel.json**: Business travel calculation including flights and hotels

## Usage Examples

### Single Record Calculation

```bash
# Category 1 - Purchased Goods
vcci calculate --category 1 --input examples/sample_category1_single.json

# Category 4 - Transportation
vcci calculate --category 4 --input examples/sample_category4_transport.json

# Category 6 - Business Travel
vcci calculate --category 6 --input examples/sample_category6_travel.json
```

### Batch Processing

```bash
# Process multiple records from CSV
vcci calculate --category 1 --input examples/sample_category1_batch.csv --batch

# With verbose output
vcci calculate --category 1 --input examples/sample_category1_batch.csv --batch --verbose

# Save results to file
vcci calculate --category 1 --input examples/sample_category1_batch.csv --batch --output results.json
```

### Advanced Options

```bash
# Disable Monte Carlo uncertainty (faster)
vcci calculate --category 4 --input examples/sample_category4_transport.json --no-mc

# Enable verbose mode for detailed output
vcci calculate --category 1 --input examples/sample_category1_single.json --verbose
```

## Input Data Format Reference

### Category 1 Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| product_name | string | Yes | Product or service name |
| quantity | number | Yes | Quantity purchased |
| quantity_unit | string | Yes | Unit (kg, units, etc.) |
| region | string | Yes | ISO 3166-1 alpha-2 country code |
| supplier_pcf | number | No | Supplier PCF (kgCO2e/unit) - Tier 1 |
| supplier_pcf_uncertainty | number | No | PCF uncertainty (0-1) |
| product_code | string | No | Product code - Tier 2 |
| product_category | string | No | Category - Tier 2 |
| spend_usd | number | No | Spend amount - Tier 3 |
| economic_sector | string | No | Economic sector - Tier 3 |

### Category 4 Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| transport_mode | string | Yes | Transport mode (see modes below) |
| distance_km | number | Yes | Distance in kilometers |
| weight_tonnes | number | Yes | Weight in tonnes |
| origin | string | No | Origin location |
| destination | string | No | Destination location |
| shipment_id | string | No | Shipment identifier |

**Transport Modes:**
- Road: `road_truck_light`, `road_truck_medium`, `road_truck_heavy`, `road_van`
- Rail: `rail_freight`, `rail_freight_electric`, `rail_freight_diesel`
- Sea: `sea_container`, `sea_bulk`, `sea_tanker`, `sea_ro_ro`
- Air: `air_cargo`, `air_freight`
- Inland: `inland_waterway`

### Category 6 Fields

Category 6 supports nested structures for flights, hotels, and ground transport.

**Flight Object:**
```json
{
  "distance_km": 3500,
  "cabin_class": "economy",
  "num_passengers": 1,
  "apply_radiative_forcing": true
}
```

**Cabin Classes:** `economy`, `premium_economy`, `business`, `first`

**Hotel Object:**
```json
{
  "nights": 3,
  "region": "GB"
}
```

## Creating Your Own Data Files

You can create your own JSON or CSV files following the formats above. The CLI will automatically:

- Detect the file format
- Validate the input data
- Select the appropriate calculation tier
- Calculate emissions with uncertainty
- Track data quality (DQI)
- Generate provenance information

## Support

For more information:
- Run `vcci calculate --help` for detailed CLI help
- Check `cli/CLI_QUICK_REFERENCE.md` for all available commands
- Review the main README.md for platform overview
