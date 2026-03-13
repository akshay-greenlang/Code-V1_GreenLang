# -*- coding: utf-8 -*-
"""
EUDR Form Template Definitions - AGENT-EUDR-015

Built-in EUDR form template definitions for the 6 form types required
by EU 2023/1115 Articles 4, 9, 10, 14, 16, and 22.  Each template
defines the complete structure (template_id, name, description, version,
form_type, commodity_types, fields with type/label/required/validation/
conditions, sections, and default_language) for:

    1. PRODUCER_REGISTRATION  - Art. 9(1)(f) operator identification
    2. PLOT_SURVEY            - Art. 9(1)(c-d) geolocation capture
    3. HARVEST_LOG            - Art. 9(1)(a-b,e) production data
    4. CUSTODY_TRANSFER       - Art. 9(1)(f-g) chain of custody
    5. QUALITY_INSPECTION     - Art. 10(1) risk assessment data
    6. SMALLHOLDER_DECLARATION - Art. 4(2) due diligence declaration

All templates are module-level frozen dictionaries, importable without
side effects.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015 Mobile Data Collector (GL-EUDR-MDC-015)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helper: field definition builder
# ---------------------------------------------------------------------------


def _field(
    field_id: str,
    field_type: str,
    label: str,
    required: bool = False,
    description: str = "",
    validation: Optional[Dict[str, Any]] = None,
    conditions: Optional[List[Dict[str, Any]]] = None,
    options: Optional[List[Dict[str, str]]] = None,
    default_value: Any = None,
    section: str = "",
    placeholder: str = "",
    help_text: str = "",
) -> Dict[str, Any]:
    """Build a field definition dictionary.

    Args:
        field_id: Unique field identifier within the template.
        field_type: Field type (text, number, date, select, etc.).
        label: Human-readable label for the field.
        required: Whether the field is required for submission.
        description: Long-form description for the field.
        validation: Validation rules dictionary.
        conditions: Conditional display/skip logic rules.
        options: Options list for select/multiselect fields.
        default_value: Default value for the field.
        section: Section grouping identifier.
        placeholder: Placeholder text hint.
        help_text: Help text displayed below the field.

    Returns:
        Complete field definition dictionary.
    """
    defn: Dict[str, Any] = {
        "field_id": field_id,
        "field_type": field_type,
        "label": label,
        "required": required,
        "description": description,
        "validation": validation or {},
        "conditions": conditions or [],
        "section": section,
        "placeholder": placeholder,
        "help_text": help_text,
    }
    if options is not None:
        defn["options"] = options
    if default_value is not None:
        defn["default_value"] = default_value
    return defn


# ===========================================================================
# Template 1: PRODUCER_REGISTRATION
# ===========================================================================

PRODUCER_REGISTRATION_TEMPLATE: Dict[str, Any] = {
    "template_id": "tmpl-eudr-producer-registration-v1",
    "name": "Producer Registration",
    "description": (
        "EUDR Art. 9(1)(f) producer/operator identification form "
        "capturing operator details, EORI number, type, country, "
        "commodities handled, registration date, and certification "
        "references for due diligence statement preparation."
    ),
    "version": "1.0.0",
    "form_type": "producer_registration",
    "commodity_types": [
        "cattle", "cocoa", "coffee", "oil_palm",
        "rubber", "soya", "wood",
    ],
    "default_language": "en",
    "sections": [
        {
            "section_id": "operator_info",
            "title": "Operator Information",
            "order": 1,
        },
        {
            "section_id": "registration_details",
            "title": "Registration Details",
            "order": 2,
        },
        {
            "section_id": "certification",
            "title": "Certification References",
            "order": 3,
        },
    ],
    "fields": [
        _field(
            "operator_name", "text", "Operator Name",
            required=True,
            description="Full legal name of the operator or trader.",
            validation={"min_length": 2, "max_length": 200},
            section="operator_info",
            placeholder="Enter operator legal name",
        ),
        _field(
            "eori_number", "text", "EORI Number",
            required=True,
            description=(
                "Economic Operators Registration and Identification "
                "number (2-letter country code + up to 15 alphanumeric)."
            ),
            validation={
                "pattern": r"^[A-Z]{2}[A-Z0-9]{1,15}$",
                "pattern_description": "2 uppercase letters + up to 15 alphanumeric",
            },
            section="operator_info",
            placeholder="e.g. DE123456789012345",
        ),
        _field(
            "operator_type", "select", "Operator Type",
            required=True,
            description="Classification of the operator per EUDR Art. 2.",
            options=[
                {"value": "operator", "label": "Operator"},
                {"value": "trader", "label": "Trader"},
                {"value": "sme_operator", "label": "SME Operator"},
                {"value": "sme_trader", "label": "SME Trader"},
                {"value": "authorized_representative", "label": "Authorized Representative"},
            ],
            section="operator_info",
        ),
        _field(
            "country_code", "text", "Country",
            required=True,
            description="ISO 3166-1 alpha-2 country code of the operator.",
            validation={
                "pattern": r"^[A-Z]{2}$",
                "pattern_description": "2 uppercase letters (ISO 3166-1 alpha-2)",
            },
            section="operator_info",
            placeholder="e.g. DE",
        ),
        _field(
            "address_line_1", "text", "Address Line 1",
            required=True,
            validation={"max_length": 200},
            section="operator_info",
        ),
        _field(
            "address_line_2", "text", "Address Line 2",
            required=False,
            validation={"max_length": 200},
            section="operator_info",
        ),
        _field(
            "postal_code", "text", "Postal Code",
            required=True,
            validation={"max_length": 20},
            section="operator_info",
        ),
        _field(
            "city", "text", "City",
            required=True,
            validation={"max_length": 100},
            section="operator_info",
        ),
        _field(
            "contact_name", "text", "Contact Person",
            required=True,
            validation={"max_length": 150},
            section="operator_info",
        ),
        _field(
            "contact_email", "text", "Contact Email",
            required=True,
            validation={
                "pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                "pattern_description": "Valid email address",
            },
            section="operator_info",
        ),
        _field(
            "contact_phone", "text", "Contact Phone",
            required=False,
            validation={"max_length": 30},
            section="operator_info",
        ),
        _field(
            "commodities_handled", "multiselect", "Commodities Handled",
            required=True,
            description="EUDR-regulated commodities this operator handles.",
            options=[
                {"value": "cattle", "label": "Cattle"},
                {"value": "cocoa", "label": "Cocoa"},
                {"value": "coffee", "label": "Coffee"},
                {"value": "oil_palm", "label": "Oil Palm"},
                {"value": "rubber", "label": "Rubber"},
                {"value": "soya", "label": "Soya"},
                {"value": "wood", "label": "Wood"},
            ],
            section="registration_details",
        ),
        _field(
            "registration_date", "date", "Registration Date",
            required=True,
            description="Date of operator registration.",
            section="registration_details",
        ),
        _field(
            "vat_number", "text", "VAT Number",
            required=False,
            validation={"max_length": 30},
            section="registration_details",
        ),
        _field(
            "annual_volume_tonnes", "number", "Annual Volume (tonnes)",
            required=False,
            description="Estimated annual commodity volume in metric tonnes.",
            validation={"min_value": 0, "max_value": 10000000},
            section="registration_details",
        ),
        _field(
            "is_sme", "checkbox", "SME Status",
            required=False,
            description="Check if operator qualifies as an SME per EU definition.",
            default_value=False,
            section="registration_details",
        ),
        _field(
            "certification_scheme", "select", "Certification Scheme",
            required=False,
            description="Voluntary certification scheme if applicable.",
            options=[
                {"value": "none", "label": "None"},
                {"value": "fairtrade", "label": "Fairtrade"},
                {"value": "rainforest_alliance", "label": "Rainforest Alliance"},
                {"value": "fsc", "label": "FSC"},
                {"value": "pefc", "label": "PEFC"},
                {"value": "rspo", "label": "RSPO"},
                {"value": "iscc", "label": "ISCC"},
                {"value": "utz", "label": "UTZ"},
                {"value": "organic_eu", "label": "EU Organic"},
                {"value": "other", "label": "Other"},
            ],
            section="certification",
        ),
        _field(
            "certification_number", "text", "Certification Number",
            required=False,
            description="Certification reference number if scheme selected.",
            validation={"max_length": 100},
            conditions=[{
                "condition_type": "show",
                "depends_on": "certification_scheme",
                "operator": "not_equal",
                "value": "none",
            }],
            section="certification",
        ),
        _field(
            "certification_expiry", "date", "Certification Expiry Date",
            required=False,
            conditions=[{
                "condition_type": "show",
                "depends_on": "certification_scheme",
                "operator": "not_equal",
                "value": "none",
            }],
            section="certification",
        ),
        _field(
            "notes", "textarea", "Additional Notes",
            required=False,
            validation={"max_length": 2000},
            section="certification",
        ),
    ],
    "validation_rules": [
        {
            "rule_id": "vr_eori_country_match",
            "description": "EORI prefix must match country_code.",
            "fields": ["eori_number", "country_code"],
            "rule_type": "cross_field",
            "expression": "eori_number[:2] == country_code",
        },
    ],
}


# ===========================================================================
# Template 2: PLOT_SURVEY
# ===========================================================================

PLOT_SURVEY_TEMPLATE: Dict[str, Any] = {
    "template_id": "tmpl-eudr-plot-survey-v1",
    "name": "Plot Survey",
    "description": (
        "EUDR Art. 9(1)(c-d) geolocation capture form for plot "
        "boundary mapping with GPS polygon boundary, area calculation, "
        "land use history, forest cover status, soil type, elevation, "
        "and accessibility assessment."
    ),
    "version": "1.0.0",
    "form_type": "plot_survey",
    "commodity_types": [
        "cattle", "cocoa", "coffee", "oil_palm",
        "rubber", "soya", "wood",
    ],
    "default_language": "en",
    "sections": [
        {"section_id": "plot_identity", "title": "Plot Identity", "order": 1},
        {"section_id": "geolocation", "title": "Geolocation", "order": 2},
        {"section_id": "land_characteristics", "title": "Land Characteristics", "order": 3},
        {"section_id": "forest_status", "title": "Forest Cover Status", "order": 4},
    ],
    "fields": [
        _field(
            "plot_id", "text", "Plot ID",
            required=True,
            description="Unique identifier for this production plot.",
            validation={"min_length": 1, "max_length": 50},
            section="plot_identity",
        ),
        _field(
            "plot_name", "text", "Plot Name",
            required=False,
            description="Human-readable name for the plot.",
            validation={"max_length": 200},
            section="plot_identity",
        ),
        _field(
            "associated_producer_id", "text", "Producer ID",
            required=True,
            description="Reference to the registered producer.",
            section="plot_identity",
        ),
        _field(
            "commodity_type", "select", "Primary Commodity",
            required=True,
            options=[
                {"value": "cattle", "label": "Cattle"},
                {"value": "cocoa", "label": "Cocoa"},
                {"value": "coffee", "label": "Coffee"},
                {"value": "oil_palm", "label": "Oil Palm"},
                {"value": "rubber", "label": "Rubber"},
                {"value": "soya", "label": "Soya"},
                {"value": "wood", "label": "Wood"},
            ],
            section="plot_identity",
        ),
        _field(
            "gps_polygon", "gps", "GPS Polygon Boundary",
            required=True,
            description=(
                "Walk-around GPS trace of plot perimeter. Minimum 3 "
                "vertices. Polygon above 4 ha requires full boundary per "
                "EUDR Art. 9(1)(d)."
            ),
            validation={
                "capture_type": "polygon",
                "min_vertices": 3,
                "max_vertices": 10000,
                "max_accuracy_meters": 5.0,
            },
            section="geolocation",
        ),
        _field(
            "gps_centroid", "gps", "GPS Centroid Point",
            required=False,
            description="Central GPS point of the plot (auto-calculated or manual).",
            validation={"capture_type": "point", "max_accuracy_meters": 3.0},
            section="geolocation",
        ),
        _field(
            "area_ha", "number", "Area (hectares)",
            required=True,
            description="Plot area in hectares, calculated from GPS polygon.",
            validation={"min_value": 0.001, "max_value": 100000},
            section="geolocation",
            help_text="Auto-calculated from polygon boundary.",
        ),
        _field(
            "elevation_m", "number", "Elevation (meters)",
            required=False,
            description="Average plot elevation above sea level in meters.",
            validation={"min_value": -500, "max_value": 9000},
            section="geolocation",
        ),
        _field(
            "land_use_current", "select", "Current Land Use",
            required=True,
            options=[
                {"value": "cropland", "label": "Cropland"},
                {"value": "pasture", "label": "Pasture"},
                {"value": "agroforestry", "label": "Agroforestry"},
                {"value": "plantation", "label": "Plantation"},
                {"value": "fallow", "label": "Fallow"},
                {"value": "mixed", "label": "Mixed"},
                {"value": "other", "label": "Other"},
            ],
            section="land_characteristics",
        ),
        _field(
            "land_use_history", "textarea", "Land Use History",
            required=False,
            description=(
                "Description of land use changes since 31 December 2020 "
                "(EUDR deforestation cutoff date)."
            ),
            validation={"max_length": 2000},
            section="land_characteristics",
        ),
        _field(
            "soil_type", "select", "Soil Type",
            required=False,
            options=[
                {"value": "clay", "label": "Clay"},
                {"value": "loam", "label": "Loam"},
                {"value": "sand", "label": "Sand"},
                {"value": "silt", "label": "Silt"},
                {"value": "peat", "label": "Peat"},
                {"value": "laterite", "label": "Laterite"},
                {"value": "volcanic", "label": "Volcanic"},
                {"value": "unknown", "label": "Unknown"},
            ],
            section="land_characteristics",
        ),
        _field(
            "slope_percent", "number", "Slope (%)",
            required=False,
            validation={"min_value": 0, "max_value": 100},
            section="land_characteristics",
        ),
        _field(
            "accessibility", "select", "Accessibility",
            required=False,
            options=[
                {"value": "road_accessible", "label": "Road Accessible"},
                {"value": "track_only", "label": "Track/Trail Only"},
                {"value": "boat_only", "label": "Boat Access Only"},
                {"value": "foot_only", "label": "Foot Access Only"},
                {"value": "seasonal_access", "label": "Seasonal Access"},
            ],
            section="land_characteristics",
        ),
        _field(
            "water_source", "select", "Water Source",
            required=False,
            options=[
                {"value": "rain_fed", "label": "Rain-fed"},
                {"value": "river_stream", "label": "River/Stream"},
                {"value": "well_borehole", "label": "Well/Borehole"},
                {"value": "irrigation", "label": "Irrigation System"},
                {"value": "none", "label": "None"},
            ],
            section="land_characteristics",
        ),
        _field(
            "forest_cover_status", "select", "Forest Cover Status",
            required=True,
            description=(
                "Forest cover status as of 31 December 2020 per EUDR "
                "Art. 2(1) deforestation cutoff."
            ),
            options=[
                {"value": "no_forest", "label": "No Forest Cover (as of 2020-12-31)"},
                {"value": "forest_before_cutoff", "label": "Forest Removed Before Cutoff"},
                {"value": "forest_intact", "label": "Forest Intact"},
                {"value": "partially_forested", "label": "Partially Forested"},
                {"value": "reforested", "label": "Reforested After Cutoff"},
                {"value": "unknown", "label": "Unknown"},
            ],
            section="forest_status",
        ),
        _field(
            "deforestation_free_declaration", "checkbox",
            "Deforestation-Free Declaration",
            required=True,
            description=(
                "Declaration that this plot has not been subject to "
                "deforestation after 31 December 2020."
            ),
            default_value=False,
            section="forest_status",
        ),
        _field(
            "satellite_imagery_reference", "text",
            "Satellite Imagery Reference",
            required=False,
            description="Reference to supporting satellite imagery for verification.",
            validation={"max_length": 200},
            section="forest_status",
        ),
        _field(
            "plot_photos", "photo", "Plot Photos",
            required=True,
            description="Photographic evidence of the plot landscape.",
            validation={"min_photos": 1, "max_photos": 10},
            section="forest_status",
        ),
        _field(
            "surveyor_notes", "textarea", "Surveyor Notes",
            required=False,
            validation={"max_length": 2000},
            section="forest_status",
        ),
    ],
    "validation_rules": [
        {
            "rule_id": "vr_area_positive",
            "description": "Area must be greater than zero.",
            "fields": ["area_ha"],
            "rule_type": "field",
            "expression": "area_ha > 0",
        },
        {
            "rule_id": "vr_deforestation_free_required",
            "description": "Deforestation-free declaration required for submission.",
            "fields": ["deforestation_free_declaration"],
            "rule_type": "field",
            "expression": "deforestation_free_declaration == True",
        },
    ],
}


# ===========================================================================
# Template 3: HARVEST_LOG
# ===========================================================================

HARVEST_LOG_TEMPLATE: Dict[str, Any] = {
    "template_id": "tmpl-eudr-harvest-log-v1",
    "name": "Harvest Log",
    "description": (
        "EUDR Art. 9(1)(a-b,e) production data capture form recording "
        "commodity type, quantity, harvest date, plot reference, quality "
        "grade, moisture content, and processing method."
    ),
    "version": "1.0.0",
    "form_type": "harvest_log",
    "commodity_types": [
        "cattle", "cocoa", "coffee", "oil_palm",
        "rubber", "soya", "wood",
    ],
    "default_language": "en",
    "sections": [
        {"section_id": "harvest_identity", "title": "Harvest Identity", "order": 1},
        {"section_id": "quantity", "title": "Quantity & Weight", "order": 2},
        {"section_id": "quality", "title": "Quality Assessment", "order": 3},
        {"section_id": "processing", "title": "Processing", "order": 4},
    ],
    "fields": [
        _field(
            "harvest_id", "text", "Harvest ID",
            required=True,
            validation={"min_length": 1, "max_length": 50},
            section="harvest_identity",
        ),
        _field(
            "plot_reference", "text", "Plot Reference",
            required=True,
            description="Reference to the plot survey form for this harvest.",
            section="harvest_identity",
        ),
        _field(
            "commodity_type", "select", "Commodity Type",
            required=True,
            options=[
                {"value": "cattle", "label": "Cattle (live/beef/leather/tallow)"},
                {"value": "cocoa", "label": "Cocoa (beans)"},
                {"value": "coffee", "label": "Coffee (cherry/parchment/green)"},
                {"value": "oil_palm", "label": "Oil Palm (FFB)"},
                {"value": "rubber", "label": "Rubber (cup lump/latex)"},
                {"value": "soya", "label": "Soya (beans)"},
                {"value": "wood", "label": "Wood (logs/timber)"},
            ],
            section="harvest_identity",
        ),
        _field(
            "harvest_date", "date", "Harvest Date",
            required=True,
            description="Date of harvest or collection.",
            section="harvest_identity",
        ),
        _field(
            "harvest_end_date", "date", "Harvest End Date",
            required=False,
            description="End date if harvest spans multiple days.",
            section="harvest_identity",
        ),
        _field(
            "harvester_name", "text", "Harvester Name",
            required=False,
            validation={"max_length": 150},
            section="harvest_identity",
        ),
        _field(
            "quantity_kg", "number", "Quantity (kg)",
            required=True,
            description="Harvested quantity in kilograms.",
            validation={"min_value": 0.01, "max_value": 10000000},
            section="quantity",
        ),
        _field(
            "unit_of_measure", "select", "Unit of Measure",
            required=True,
            default_value="kg",
            options=[
                {"value": "kg", "label": "Kilograms (kg)"},
                {"value": "tonnes", "label": "Metric Tonnes (t)"},
                {"value": "pieces", "label": "Pieces/Head"},
                {"value": "m3", "label": "Cubic Meters (m3)"},
                {"value": "bags", "label": "Bags (60kg)"},
            ],
            section="quantity",
        ),
        _field(
            "number_of_bags", "number", "Number of Bags",
            required=False,
            validation={"min_value": 0, "max_value": 100000},
            conditions=[{
                "condition_type": "show",
                "depends_on": "unit_of_measure",
                "operator": "equal",
                "value": "bags",
            }],
            section="quantity",
        ),
        _field(
            "tare_weight_kg", "number", "Tare Weight (kg)",
            required=False,
            validation={"min_value": 0, "max_value": 1000000},
            section="quantity",
        ),
        _field(
            "quality_grade", "select", "Quality Grade",
            required=True,
            options=[
                {"value": "grade_1", "label": "Grade 1 (Premium)"},
                {"value": "grade_2", "label": "Grade 2 (Standard)"},
                {"value": "grade_3", "label": "Grade 3 (Below Standard)"},
                {"value": "reject", "label": "Reject"},
                {"value": "ungraded", "label": "Ungraded"},
            ],
            section="quality",
        ),
        _field(
            "moisture_content_pct", "number", "Moisture Content (%)",
            required=False,
            description="Moisture content as percentage of total weight.",
            validation={"min_value": 0, "max_value": 100},
            section="quality",
        ),
        _field(
            "defect_rate_pct", "number", "Defect Rate (%)",
            required=False,
            validation={"min_value": 0, "max_value": 100},
            section="quality",
        ),
        _field(
            "foreign_matter_pct", "number", "Foreign Matter (%)",
            required=False,
            validation={"min_value": 0, "max_value": 100},
            section="quality",
        ),
        _field(
            "processing_method", "select", "Processing Method",
            required=False,
            options=[
                {"value": "none", "label": "None (Raw)"},
                {"value": "washed", "label": "Washed"},
                {"value": "natural", "label": "Natural/Sun-dried"},
                {"value": "honey", "label": "Honey/Pulped Natural"},
                {"value": "fermented", "label": "Fermented"},
                {"value": "dried", "label": "Dried"},
                {"value": "smoked", "label": "Smoked"},
                {"value": "pressed", "label": "Pressed"},
                {"value": "milled", "label": "Milled"},
                {"value": "other", "label": "Other"},
            ],
            section="processing",
        ),
        _field(
            "processing_date", "date", "Processing Date",
            required=False,
            conditions=[{
                "condition_type": "show",
                "depends_on": "processing_method",
                "operator": "not_equal",
                "value": "none",
            }],
            section="processing",
        ),
        _field(
            "storage_location", "text", "Storage Location",
            required=False,
            validation={"max_length": 200},
            section="processing",
        ),
        _field(
            "batch_reference", "text", "Batch Reference",
            required=False,
            validation={"max_length": 50},
            section="processing",
        ),
        _field(
            "harvest_photos", "photo", "Harvest Photos",
            required=False,
            description="Photos of harvested commodity.",
            validation={"min_photos": 0, "max_photos": 10},
            section="processing",
        ),
        _field(
            "harvest_gps", "gps", "Harvest Location GPS",
            required=False,
            description="GPS point at the harvest collection point.",
            validation={"capture_type": "point", "max_accuracy_meters": 10.0},
            section="harvest_identity",
        ),
        _field(
            "notes", "textarea", "Notes",
            required=False,
            validation={"max_length": 2000},
            section="processing",
        ),
    ],
    "validation_rules": [
        {
            "rule_id": "vr_quantity_positive",
            "description": "Harvest quantity must be positive.",
            "fields": ["quantity_kg"],
            "rule_type": "field",
            "expression": "quantity_kg > 0",
        },
        {
            "rule_id": "vr_moisture_range",
            "description": "Moisture must be between 0 and 100.",
            "fields": ["moisture_content_pct"],
            "rule_type": "field",
            "expression": "moisture_content_pct >= 0 and moisture_content_pct <= 100",
        },
    ],
}


# ===========================================================================
# Template 4: CUSTODY_TRANSFER
# ===========================================================================

CUSTODY_TRANSFER_TEMPLATE: Dict[str, Any] = {
    "template_id": "tmpl-eudr-custody-transfer-v1",
    "name": "Custody Transfer",
    "description": (
        "EUDR Art. 9(1)(f-g) chain of custody form recording transfer "
        "of commodity between parties including from/to party, commodity, "
        "quantity, transport mode, vehicle ID, departure/arrival "
        "timestamps, and witness information."
    ),
    "version": "1.0.0",
    "form_type": "custody_transfer",
    "commodity_types": [
        "cattle", "cocoa", "coffee", "oil_palm",
        "rubber", "soya", "wood",
    ],
    "default_language": "en",
    "sections": [
        {"section_id": "parties", "title": "Transfer Parties", "order": 1},
        {"section_id": "commodity_details", "title": "Commodity Details", "order": 2},
        {"section_id": "transport", "title": "Transport Information", "order": 3},
        {"section_id": "verification", "title": "Verification & Witness", "order": 4},
    ],
    "fields": [
        _field(
            "transfer_id", "text", "Transfer ID",
            required=True,
            validation={"min_length": 1, "max_length": 50},
            section="parties",
        ),
        _field(
            "from_party_name", "text", "From Party Name",
            required=True,
            description="Name of the party transferring custody.",
            validation={"max_length": 200},
            section="parties",
        ),
        _field(
            "from_party_id", "text", "From Party ID",
            required=True,
            description="Unique identifier of the transferring party.",
            validation={"max_length": 50},
            section="parties",
        ),
        _field(
            "from_party_role", "select", "From Party Role",
            required=True,
            options=[
                {"value": "producer", "label": "Producer"},
                {"value": "collector", "label": "Collector/Aggregator"},
                {"value": "processor", "label": "Processor"},
                {"value": "trader", "label": "Trader"},
                {"value": "transporter", "label": "Transporter"},
                {"value": "warehouse", "label": "Warehouse"},
                {"value": "exporter", "label": "Exporter"},
            ],
            section="parties",
        ),
        _field(
            "to_party_name", "text", "To Party Name",
            required=True,
            validation={"max_length": 200},
            section="parties",
        ),
        _field(
            "to_party_id", "text", "To Party ID",
            required=True,
            validation={"max_length": 50},
            section="parties",
        ),
        _field(
            "to_party_role", "select", "To Party Role",
            required=True,
            options=[
                {"value": "collector", "label": "Collector/Aggregator"},
                {"value": "processor", "label": "Processor"},
                {"value": "trader", "label": "Trader"},
                {"value": "transporter", "label": "Transporter"},
                {"value": "warehouse", "label": "Warehouse"},
                {"value": "exporter", "label": "Exporter"},
                {"value": "importer", "label": "Importer"},
            ],
            section="parties",
        ),
        _field(
            "commodity_type", "select", "Commodity Type",
            required=True,
            options=[
                {"value": "cattle", "label": "Cattle"},
                {"value": "cocoa", "label": "Cocoa"},
                {"value": "coffee", "label": "Coffee"},
                {"value": "oil_palm", "label": "Oil Palm"},
                {"value": "rubber", "label": "Rubber"},
                {"value": "soya", "label": "Soya"},
                {"value": "wood", "label": "Wood"},
            ],
            section="commodity_details",
        ),
        _field(
            "quantity_kg", "number", "Quantity (kg)",
            required=True,
            validation={"min_value": 0.01, "max_value": 10000000},
            section="commodity_details",
        ),
        _field(
            "harvest_reference", "text", "Harvest Log Reference",
            required=False,
            description="Reference to origin harvest log form(s).",
            section="commodity_details",
        ),
        _field(
            "batch_reference", "text", "Batch Reference",
            required=False,
            validation={"max_length": 50},
            section="commodity_details",
        ),
        _field(
            "quality_grade", "select", "Quality Grade",
            required=False,
            options=[
                {"value": "grade_1", "label": "Grade 1"},
                {"value": "grade_2", "label": "Grade 2"},
                {"value": "grade_3", "label": "Grade 3"},
                {"value": "ungraded", "label": "Ungraded"},
            ],
            section="commodity_details",
        ),
        _field(
            "transport_mode", "select", "Transport Mode",
            required=True,
            options=[
                {"value": "truck", "label": "Truck"},
                {"value": "motorcycle", "label": "Motorcycle"},
                {"value": "bicycle", "label": "Bicycle"},
                {"value": "boat", "label": "Boat"},
                {"value": "donkey_horse", "label": "Donkey/Horse"},
                {"value": "foot", "label": "On Foot"},
                {"value": "rail", "label": "Rail"},
                {"value": "container", "label": "Container Ship"},
                {"value": "air", "label": "Air Freight"},
                {"value": "other", "label": "Other"},
            ],
            section="transport",
        ),
        _field(
            "vehicle_id", "text", "Vehicle/Vessel ID",
            required=False,
            description="License plate, vessel name, or container number.",
            validation={"max_length": 50},
            section="transport",
        ),
        _field(
            "departure_timestamp", "date", "Departure Date/Time",
            required=True,
            section="transport",
        ),
        _field(
            "departure_gps", "gps", "Departure GPS",
            required=True,
            description="GPS coordinates at departure point.",
            validation={"capture_type": "point", "max_accuracy_meters": 10.0},
            section="transport",
        ),
        _field(
            "arrival_timestamp", "date", "Arrival Date/Time",
            required=False,
            section="transport",
        ),
        _field(
            "arrival_gps", "gps", "Arrival GPS",
            required=False,
            validation={"capture_type": "point", "max_accuracy_meters": 10.0},
            section="transport",
        ),
        _field(
            "distance_km", "number", "Distance (km)",
            required=False,
            validation={"min_value": 0, "max_value": 50000},
            section="transport",
        ),
        _field(
            "witness_name", "text", "Witness Name",
            required=False,
            validation={"max_length": 150},
            section="verification",
        ),
        _field(
            "witness_role", "text", "Witness Role",
            required=False,
            validation={"max_length": 100},
            section="verification",
        ),
        _field(
            "from_party_signature", "signature", "From Party Signature",
            required=True,
            description="Digital signature of the transferring party.",
            section="verification",
        ),
        _field(
            "to_party_signature", "signature", "To Party Signature",
            required=True,
            description="Digital signature of the receiving party.",
            section="verification",
        ),
        _field(
            "witness_signature", "signature", "Witness Signature",
            required=False,
            section="verification",
        ),
        _field(
            "transfer_photos", "photo", "Transfer Photos",
            required=False,
            description="Photos of commodity at transfer.",
            validation={"max_photos": 10},
            section="verification",
        ),
        _field(
            "notes", "textarea", "Notes",
            required=False,
            validation={"max_length": 2000},
            section="verification",
        ),
    ],
    "validation_rules": [
        {
            "rule_id": "vr_parties_different",
            "description": "From and To parties must be different.",
            "fields": ["from_party_id", "to_party_id"],
            "rule_type": "cross_field",
            "expression": "from_party_id != to_party_id",
        },
        {
            "rule_id": "vr_departure_before_arrival",
            "description": "Departure must be before arrival.",
            "fields": ["departure_timestamp", "arrival_timestamp"],
            "rule_type": "cross_field",
            "expression": "departure_timestamp <= arrival_timestamp",
        },
    ],
}


# ===========================================================================
# Template 5: QUALITY_INSPECTION
# ===========================================================================

QUALITY_INSPECTION_TEMPLATE: Dict[str, Any] = {
    "template_id": "tmpl-eudr-quality-inspection-v1",
    "name": "Quality Inspection",
    "description": (
        "EUDR Art. 10(1) risk assessment data form capturing quality "
        "inspection results including sample ID, visual grade, moisture, "
        "defects, foreign matter, cup score (coffee), and fermentation "
        "status (cocoa)."
    ),
    "version": "1.0.0",
    "form_type": "quality_inspection",
    "commodity_types": [
        "cattle", "cocoa", "coffee", "oil_palm",
        "rubber", "soya", "wood",
    ],
    "default_language": "en",
    "sections": [
        {"section_id": "sample_identity", "title": "Sample Identity", "order": 1},
        {"section_id": "physical_inspection", "title": "Physical Inspection", "order": 2},
        {"section_id": "commodity_specific", "title": "Commodity-Specific Tests", "order": 3},
        {"section_id": "results", "title": "Inspection Results", "order": 4},
    ],
    "fields": [
        _field(
            "sample_id", "text", "Sample ID",
            required=True,
            validation={"min_length": 1, "max_length": 50},
            section="sample_identity",
        ),
        _field(
            "inspection_date", "date", "Inspection Date",
            required=True,
            section="sample_identity",
        ),
        _field(
            "inspector_name", "text", "Inspector Name",
            required=True,
            validation={"max_length": 150},
            section="sample_identity",
        ),
        _field(
            "batch_reference", "text", "Batch Reference",
            required=True,
            description="Reference to the harvest or custody batch.",
            section="sample_identity",
        ),
        _field(
            "commodity_type", "select", "Commodity Type",
            required=True,
            options=[
                {"value": "cocoa", "label": "Cocoa"},
                {"value": "coffee", "label": "Coffee"},
                {"value": "oil_palm", "label": "Oil Palm"},
                {"value": "rubber", "label": "Rubber"},
                {"value": "soya", "label": "Soya"},
                {"value": "wood", "label": "Wood"},
                {"value": "cattle", "label": "Cattle"},
            ],
            section="sample_identity",
        ),
        _field(
            "sample_weight_g", "number", "Sample Weight (g)",
            required=True,
            validation={"min_value": 1, "max_value": 100000},
            section="sample_identity",
        ),
        _field(
            "visual_grade", "select", "Visual Grade",
            required=True,
            options=[
                {"value": "excellent", "label": "Excellent"},
                {"value": "good", "label": "Good"},
                {"value": "acceptable", "label": "Acceptable"},
                {"value": "below_standard", "label": "Below Standard"},
                {"value": "reject", "label": "Reject"},
            ],
            section="physical_inspection",
        ),
        _field(
            "moisture_pct", "number", "Moisture (%)",
            required=True,
            description="Moisture content as percentage of sample weight.",
            validation={"min_value": 0, "max_value": 100, "decimal_places": 2},
            section="physical_inspection",
        ),
        _field(
            "defect_pct", "number", "Defect (%)",
            required=True,
            description="Percentage of defective commodity in sample.",
            validation={"min_value": 0, "max_value": 100, "decimal_places": 2},
            section="physical_inspection",
        ),
        _field(
            "foreign_matter_pct", "number", "Foreign Matter (%)",
            required=True,
            description="Percentage of foreign matter in sample.",
            validation={"min_value": 0, "max_value": 100, "decimal_places": 2},
            section="physical_inspection",
        ),
        _field(
            "color_assessment", "select", "Color Assessment",
            required=False,
            options=[
                {"value": "uniform", "label": "Uniform"},
                {"value": "slightly_varied", "label": "Slightly Varied"},
                {"value": "varied", "label": "Varied"},
                {"value": "discolored", "label": "Discolored"},
            ],
            section="physical_inspection",
        ),
        _field(
            "odor_assessment", "select", "Odor Assessment",
            required=False,
            options=[
                {"value": "normal", "label": "Normal"},
                {"value": "slightly_off", "label": "Slightly Off"},
                {"value": "off", "label": "Off/Abnormal"},
                {"value": "moldy", "label": "Moldy"},
            ],
            section="physical_inspection",
        ),
        # Coffee-specific fields
        _field(
            "cup_score", "number", "Cup Score",
            required=False,
            description="SCA cupping score for coffee (0-100).",
            validation={"min_value": 0, "max_value": 100, "decimal_places": 1},
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "coffee",
            }],
            section="commodity_specific",
        ),
        _field(
            "screen_size", "number", "Screen Size",
            required=False,
            description="Coffee bean screen size (10-20).",
            validation={"min_value": 8, "max_value": 22},
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "coffee",
            }],
            section="commodity_specific",
        ),
        _field(
            "bean_count_per_300g", "number", "Bean Count per 300g",
            required=False,
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "coffee",
            }],
            section="commodity_specific",
        ),
        # Cocoa-specific fields
        _field(
            "fermentation_status", "select", "Fermentation Status",
            required=False,
            description="Cocoa bean fermentation assessment.",
            options=[
                {"value": "well_fermented", "label": "Well Fermented (>75%)"},
                {"value": "moderately_fermented", "label": "Moderately Fermented (50-75%)"},
                {"value": "under_fermented", "label": "Under Fermented (<50%)"},
                {"value": "unfermented", "label": "Unfermented"},
            ],
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "cocoa",
            }],
            section="commodity_specific",
        ),
        _field(
            "bean_count_per_100g", "number", "Bean Count per 100g",
            required=False,
            description="Cocoa bean count per 100 grams.",
            validation={"min_value": 50, "max_value": 200},
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "cocoa",
            }],
            section="commodity_specific",
        ),
        # Oil palm-specific
        _field(
            "ffa_pct", "number", "Free Fatty Acid (%)",
            required=False,
            description="Free Fatty Acid content for oil palm.",
            validation={"min_value": 0, "max_value": 100, "decimal_places": 2},
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "oil_palm",
            }],
            section="commodity_specific",
        ),
        # Rubber-specific
        _field(
            "drc_pct", "number", "Dry Rubber Content (%)",
            required=False,
            description="Dry Rubber Content percentage.",
            validation={"min_value": 0, "max_value": 100, "decimal_places": 2},
            conditions=[{
                "condition_type": "show",
                "depends_on": "commodity_type",
                "operator": "equal",
                "value": "rubber",
            }],
            section="commodity_specific",
        ),
        _field(
            "overall_pass_fail", "select", "Overall Result",
            required=True,
            options=[
                {"value": "pass", "label": "Pass"},
                {"value": "conditional_pass", "label": "Conditional Pass"},
                {"value": "fail", "label": "Fail"},
            ],
            section="results",
        ),
        _field(
            "rejection_reason", "textarea", "Rejection Reason",
            required=False,
            validation={"max_length": 1000},
            conditions=[{
                "condition_type": "show",
                "depends_on": "overall_pass_fail",
                "operator": "equal",
                "value": "fail",
            }],
            section="results",
        ),
        _field(
            "inspector_signature", "signature", "Inspector Signature",
            required=True,
            section="results",
        ),
        _field(
            "inspection_photos", "photo", "Inspection Photos",
            required=False,
            validation={"max_photos": 10},
            section="results",
        ),
        _field(
            "notes", "textarea", "Notes",
            required=False,
            validation={"max_length": 2000},
            section="results",
        ),
    ],
    "validation_rules": [
        {
            "rule_id": "vr_total_pct_max",
            "description": "Defect + foreign matter cannot exceed 100%.",
            "fields": ["defect_pct", "foreign_matter_pct"],
            "rule_type": "cross_field",
            "expression": "defect_pct + foreign_matter_pct <= 100",
        },
    ],
}


# ===========================================================================
# Template 6: SMALLHOLDER_DECLARATION
# ===========================================================================

SMALLHOLDER_DECLARATION_TEMPLATE: Dict[str, Any] = {
    "template_id": "tmpl-eudr-smallholder-declaration-v1",
    "name": "Smallholder Declaration",
    "description": (
        "EUDR Art. 4(2) due diligence declaration form for smallholder "
        "farmers capturing farmer name, national ID, farm size, number "
        "of plots, commodities grown, deforestation-free declaration, "
        "and GPS of homestead."
    ),
    "version": "1.0.0",
    "form_type": "smallholder_declaration",
    "commodity_types": [
        "cattle", "cocoa", "coffee", "oil_palm",
        "rubber", "soya", "wood",
    ],
    "default_language": "en",
    "sections": [
        {"section_id": "farmer_identity", "title": "Farmer Identity", "order": 1},
        {"section_id": "farm_details", "title": "Farm Details", "order": 2},
        {"section_id": "declaration", "title": "Declaration", "order": 3},
    ],
    "fields": [
        _field(
            "farmer_name", "text", "Farmer Full Name",
            required=True,
            validation={"min_length": 2, "max_length": 200},
            section="farmer_identity",
        ),
        _field(
            "national_id", "text", "National ID Number",
            required=True,
            description="Government-issued national identification number.",
            validation={"min_length": 3, "max_length": 30},
            section="farmer_identity",
        ),
        _field(
            "national_id_type", "select", "ID Type",
            required=True,
            options=[
                {"value": "national_id_card", "label": "National ID Card"},
                {"value": "passport", "label": "Passport"},
                {"value": "voter_id", "label": "Voter ID"},
                {"value": "drivers_license", "label": "Driver's License"},
                {"value": "birth_certificate", "label": "Birth Certificate"},
                {"value": "other", "label": "Other"},
            ],
            section="farmer_identity",
        ),
        _field(
            "date_of_birth", "date", "Date of Birth",
            required=False,
            section="farmer_identity",
        ),
        _field(
            "gender", "select", "Gender",
            required=False,
            options=[
                {"value": "male", "label": "Male"},
                {"value": "female", "label": "Female"},
                {"value": "other", "label": "Other"},
                {"value": "prefer_not_to_say", "label": "Prefer Not to Say"},
            ],
            section="farmer_identity",
        ),
        _field(
            "phone_number", "text", "Phone Number",
            required=False,
            validation={"max_length": 20},
            section="farmer_identity",
        ),
        _field(
            "homestead_gps", "gps", "Homestead GPS",
            required=True,
            description="GPS coordinates of the farmer's homestead.",
            validation={"capture_type": "point", "max_accuracy_meters": 10.0},
            section="farmer_identity",
        ),
        _field(
            "village_name", "text", "Village/Community",
            required=False,
            validation={"max_length": 100},
            section="farmer_identity",
        ),
        _field(
            "district", "text", "District/Region",
            required=False,
            validation={"max_length": 100},
            section="farmer_identity",
        ),
        _field(
            "country_code", "text", "Country",
            required=True,
            validation={
                "pattern": r"^[A-Z]{2}$",
                "pattern_description": "ISO 3166-1 alpha-2",
            },
            section="farmer_identity",
        ),
        _field(
            "total_farm_size_ha", "number", "Total Farm Size (ha)",
            required=True,
            description="Total farm area in hectares across all plots.",
            validation={"min_value": 0.01, "max_value": 10000},
            section="farm_details",
        ),
        _field(
            "number_of_plots", "number", "Number of Plots",
            required=True,
            description="Total number of production plots.",
            validation={"min_value": 1, "max_value": 1000},
            section="farm_details",
        ),
        _field(
            "commodities_grown", "multiselect", "Commodities Grown",
            required=True,
            description="EUDR-regulated commodities cultivated by this farmer.",
            options=[
                {"value": "cattle", "label": "Cattle"},
                {"value": "cocoa", "label": "Cocoa"},
                {"value": "coffee", "label": "Coffee"},
                {"value": "oil_palm", "label": "Oil Palm"},
                {"value": "rubber", "label": "Rubber"},
                {"value": "soya", "label": "Soya"},
                {"value": "wood", "label": "Wood"},
            ],
            section="farm_details",
        ),
        _field(
            "years_farming", "number", "Years of Farming Experience",
            required=False,
            validation={"min_value": 0, "max_value": 100},
            section="farm_details",
        ),
        _field(
            "cooperative_member", "checkbox", "Cooperative Member",
            required=False,
            default_value=False,
            section="farm_details",
        ),
        _field(
            "cooperative_name", "text", "Cooperative Name",
            required=False,
            validation={"max_length": 200},
            conditions=[{
                "condition_type": "show",
                "depends_on": "cooperative_member",
                "operator": "equal",
                "value": True,
            }],
            section="farm_details",
        ),
        _field(
            "deforestation_free_declaration", "checkbox",
            "Deforestation-Free Declaration",
            required=True,
            description=(
                "I declare that my farm plots have not been subject to "
                "deforestation after 31 December 2020 as defined by "
                "EU Regulation 2023/1115."
            ),
            default_value=False,
            section="declaration",
        ),
        _field(
            "legal_compliance_declaration", "checkbox",
            "Legal Compliance Declaration",
            required=True,
            description=(
                "I declare that commodities produced on my plots comply "
                "with the relevant legislation of the country of production."
            ),
            default_value=False,
            section="declaration",
        ),
        _field(
            "data_consent", "checkbox", "Data Processing Consent",
            required=True,
            description=(
                "I consent to the processing of my personal data for "
                "the purpose of EUDR due diligence compliance."
            ),
            default_value=False,
            section="declaration",
        ),
        _field(
            "declaration_date", "date", "Declaration Date",
            required=True,
            section="declaration",
        ),
        _field(
            "farmer_signature", "signature", "Farmer Signature",
            required=True,
            description="Digital signature of the declaring farmer.",
            section="declaration",
        ),
        _field(
            "witness_name", "text", "Witness Name",
            required=False,
            validation={"max_length": 150},
            section="declaration",
        ),
        _field(
            "witness_signature", "signature", "Witness Signature",
            required=False,
            section="declaration",
        ),
        _field(
            "farmer_photo", "photo", "Farmer Photo",
            required=False,
            description="Photo of farmer for identity verification.",
            validation={"max_photos": 2},
            section="farmer_identity",
        ),
        _field(
            "notes", "textarea", "Notes",
            required=False,
            validation={"max_length": 2000},
            section="declaration",
        ),
    ],
    "validation_rules": [
        {
            "rule_id": "vr_deforestation_required",
            "description": "Deforestation-free declaration must be checked.",
            "fields": ["deforestation_free_declaration"],
            "rule_type": "field",
            "expression": "deforestation_free_declaration == True",
        },
        {
            "rule_id": "vr_legal_compliance_required",
            "description": "Legal compliance declaration must be checked.",
            "fields": ["legal_compliance_declaration"],
            "rule_type": "field",
            "expression": "legal_compliance_declaration == True",
        },
        {
            "rule_id": "vr_data_consent_required",
            "description": "Data processing consent must be given.",
            "fields": ["data_consent"],
            "rule_type": "field",
            "expression": "data_consent == True",
        },
    ],
}


# ===========================================================================
# Template registry
# ===========================================================================

ALL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "producer_registration": PRODUCER_REGISTRATION_TEMPLATE,
    "plot_survey": PLOT_SURVEY_TEMPLATE,
    "harvest_log": HARVEST_LOG_TEMPLATE,
    "custody_transfer": CUSTODY_TRANSFER_TEMPLATE,
    "quality_inspection": QUALITY_INSPECTION_TEMPLATE,
    "smallholder_declaration": SMALLHOLDER_DECLARATION_TEMPLATE,
}

TEMPLATE_REGISTRY: Dict[str, str] = {
    k: v["template_id"] for k, v in ALL_TEMPLATES.items()
}


# ===========================================================================
# Accessor functions
# ===========================================================================


def get_template(form_type: str) -> Optional[Dict[str, Any]]:
    """Return the built-in template for the given form type.

    Args:
        form_type: EUDR form type key.

    Returns:
        Template dictionary or None if not found.
    """
    return ALL_TEMPLATES.get(form_type)


def list_template_names() -> List[str]:
    """Return list of all available template form type names.

    Returns:
        Sorted list of template form type keys.
    """
    return sorted(ALL_TEMPLATES.keys())


def get_template_fields(form_type: str) -> List[Dict[str, Any]]:
    """Return the fields list for a given form type template.

    Args:
        form_type: EUDR form type key.

    Returns:
        List of field definitions, or empty list if template not found.
    """
    tmpl = ALL_TEMPLATES.get(form_type)
    if tmpl is None:
        return []
    return tmpl.get("fields", [])


def get_required_fields(form_type: str) -> List[str]:
    """Return the field IDs of all required fields for a given form type.

    Args:
        form_type: EUDR form type key.

    Returns:
        List of field_id strings for required fields.
    """
    fields = get_template_fields(form_type)
    return [f["field_id"] for f in fields if f.get("required", False)]


def validate_template_data(
    form_type: str,
    data: Dict[str, Any],
) -> List[str]:
    """Validate form data against the template's required fields.

    Args:
        form_type: EUDR form type key.
        data: Form field values to validate.

    Returns:
        List of validation error messages. Empty list means valid.
    """
    errors: List[str] = []
    tmpl = ALL_TEMPLATES.get(form_type)
    if tmpl is None:
        errors.append(f"Unknown form type: {form_type}")
        return errors

    for field_def in tmpl.get("fields", []):
        fid = field_def["field_id"]
        if field_def.get("required", False):
            val = data.get(fid)
            if val is None or val == "" or val == []:
                errors.append(f"Required field '{fid}' is missing or empty")

    return errors


# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "PRODUCER_REGISTRATION_TEMPLATE",
    "PLOT_SURVEY_TEMPLATE",
    "HARVEST_LOG_TEMPLATE",
    "CUSTODY_TRANSFER_TEMPLATE",
    "QUALITY_INSPECTION_TEMPLATE",
    "SMALLHOLDER_DECLARATION_TEMPLATE",
    "ALL_TEMPLATES",
    "TEMPLATE_REGISTRY",
    "get_template",
    "list_template_names",
    "get_template_fields",
    "get_required_fields",
    "validate_template_data",
]
