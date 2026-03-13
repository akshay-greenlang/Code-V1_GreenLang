# -*- coding: utf-8 -*-
"""
Label Template Definitions - AGENT-EUDR-014

Pre-defined label template specifications for five EUDR compliance
label types used in QR code generation.  Each template defines
physical dimensions (mm), DPI, and a list of element positions
with properties for rendering QR codes, text, barcodes, logos,
and compliance status indicators.

Templates (5):
    1. PRODUCT_LABEL_TEMPLATE    - 50x30mm individual product label
    2. SHIPPING_LABEL_TEMPLATE   - 100x150mm shipping carton label
    3. PALLET_LABEL_TEMPLATE     - 148x210mm A5 pallet label
    4. CONTAINER_LABEL_TEMPLATE  - 297x210mm A4 container label
    5. CONSUMER_LABEL_TEMPLATE   - 30x20mm consumer-facing label

Element Types:
    - qr_code: QR code image placeholder
    - text: Text element with font, size, alignment
    - barcode: Linear barcode (Code 128, EAN-13)
    - image: Logo or icon placeholder
    - rectangle: Coloured rectangle for compliance status
    - line: Divider line
    - compliance_badge: EUDR compliance status indicator

Coordinate System:
    Origin (0, 0) is the top-left corner of the label.
    All positions and dimensions are in millimetres.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
Status: Production Ready
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template 1: Product Label (50x30mm)
# ---------------------------------------------------------------------------

PRODUCT_LABEL_TEMPLATE: Dict[str, Any] = {
    "name": "product_label",
    "display_name": "Product Label",
    "description": (
        "Individual product label for EUDR compliance with QR code, "
        "product name, batch number, origin country, and compliance "
        "status badge. Designed for application to retail products."
    ),
    "width_mm": 50.0,
    "height_mm": 30.0,
    "dpi": 300,
    "orientation": "landscape",
    "bleed_mm": 3.0,
    "margin_mm": 2.0,
    "background_color": "#FFFFFF",
    "version": "1.0",
    "elements": [
        {
            "type": "qr_code",
            "id": "main_qr",
            "x_mm": 2.0,
            "y_mm": 2.0,
            "width_mm": 20.0,
            "height_mm": 20.0,
            "properties": {
                "error_correction": "M",
                "quiet_zone_modules": 2,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
                "module_shape": "square",
            },
        },
        {
            "type": "text",
            "id": "product_name",
            "x_mm": 24.0,
            "y_mm": 2.0,
            "width_mm": 24.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "left",
                "max_lines": 1,
                "overflow": "ellipsis",
                "placeholder": "{product_name}",
            },
        },
        {
            "type": "text",
            "id": "batch_number",
            "x_mm": 24.0,
            "y_mm": 7.5,
            "width_mm": 24.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 5,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "max_lines": 1,
                "placeholder": "Batch: {batch_code}",
            },
        },
        {
            "type": "text",
            "id": "origin_country",
            "x_mm": 24.0,
            "y_mm": 12.0,
            "width_mm": 24.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 5,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "max_lines": 1,
                "placeholder": "Origin: {origin_country}",
            },
        },
        {
            "type": "compliance_badge",
            "id": "compliance_badge",
            "x_mm": 24.0,
            "y_mm": 17.0,
            "width_mm": 24.0,
            "height_mm": 5.0,
            "properties": {
                "compliant_color": "#2E7D32",
                "pending_color": "#F57F17",
                "non_compliant_color": "#C62828",
                "font_family": "DejaVuSans",
                "font_size_pt": 5,
                "font_weight": "bold",
                "text_color": "#FFFFFF",
                "border_radius_mm": 1.0,
                "label_compliant": "EUDR COMPLIANT",
                "label_pending": "EUDR PENDING",
                "label_non_compliant": "NON-COMPLIANT",
                "placeholder": "{compliance_status}",
            },
        },
        {
            "type": "text",
            "id": "scan_prompt",
            "x_mm": 2.0,
            "y_mm": 23.0,
            "width_mm": 20.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 4,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "center",
                "max_lines": 2,
                "placeholder": "Scan for\ntraceability info",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Template 2: Shipping Label (100x150mm)
# ---------------------------------------------------------------------------

SHIPPING_LABEL_TEMPLATE: Dict[str, Any] = {
    "name": "shipping_label",
    "display_name": "Shipping Label",
    "description": (
        "Shipping carton label with QR code, sender and receiver "
        "areas, batch code, weight, carrier barcode space, and "
        "EUDR compliance status. Designed for logistics use."
    ),
    "width_mm": 100.0,
    "height_mm": 150.0,
    "dpi": 300,
    "orientation": "portrait",
    "bleed_mm": 3.0,
    "margin_mm": 4.0,
    "background_color": "#FFFFFF",
    "version": "1.0",
    "elements": [
        # -- Header --
        {
            "type": "text",
            "id": "header_title",
            "x_mm": 4.0,
            "y_mm": 4.0,
            "width_mm": 92.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "center",
                "placeholder": "SHIPPING LABEL",
            },
        },
        {
            "type": "line",
            "id": "header_line",
            "x_mm": 4.0,
            "y_mm": 11.0,
            "width_mm": 92.0,
            "height_mm": 0.3,
            "properties": {
                "line_color": "#000000",
                "line_width_mm": 0.3,
            },
        },
        # -- Sender area --
        {
            "type": "text",
            "id": "sender_label",
            "x_mm": 4.0,
            "y_mm": 13.0,
            "width_mm": 44.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "bold",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "FROM:",
            },
        },
        {
            "type": "text",
            "id": "sender_details",
            "x_mm": 4.0,
            "y_mm": 17.0,
            "width_mm": 44.0,
            "height_mm": 18.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#000000",
                "alignment": "left",
                "max_lines": 4,
                "placeholder": "{sender_name}\n{sender_address}",
            },
        },
        # -- Receiver area --
        {
            "type": "text",
            "id": "receiver_label",
            "x_mm": 52.0,
            "y_mm": 13.0,
            "width_mm": 44.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "bold",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "TO:",
            },
        },
        {
            "type": "text",
            "id": "receiver_details",
            "x_mm": 52.0,
            "y_mm": 17.0,
            "width_mm": 44.0,
            "height_mm": 18.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#000000",
                "alignment": "left",
                "max_lines": 4,
                "placeholder": "{receiver_name}\n{receiver_address}",
            },
        },
        {
            "type": "line",
            "id": "sender_receiver_line",
            "x_mm": 4.0,
            "y_mm": 37.0,
            "width_mm": 92.0,
            "height_mm": 0.3,
            "properties": {
                "line_color": "#000000",
                "line_width_mm": 0.3,
            },
        },
        # -- QR code and details --
        {
            "type": "qr_code",
            "id": "main_qr",
            "x_mm": 4.0,
            "y_mm": 40.0,
            "width_mm": 35.0,
            "height_mm": 35.0,
            "properties": {
                "error_correction": "M",
                "quiet_zone_modules": 4,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
            },
        },
        {
            "type": "text",
            "id": "batch_code",
            "x_mm": 42.0,
            "y_mm": 40.0,
            "width_mm": 54.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 8,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "left",
                "placeholder": "Batch: {batch_code}",
            },
        },
        {
            "type": "text",
            "id": "commodity",
            "x_mm": 42.0,
            "y_mm": 46.0,
            "width_mm": 54.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Commodity: {commodity}",
            },
        },
        {
            "type": "text",
            "id": "weight",
            "x_mm": 42.0,
            "y_mm": 52.0,
            "width_mm": 54.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Net Weight: {weight_kg} kg",
            },
        },
        {
            "type": "text",
            "id": "origin",
            "x_mm": 42.0,
            "y_mm": 58.0,
            "width_mm": 54.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Origin: {origin_country}",
            },
        },
        {
            "type": "text",
            "id": "dds_reference",
            "x_mm": 42.0,
            "y_mm": 64.0,
            "width_mm": 54.0,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "left",
                "placeholder": "DDS: {dds_reference}",
            },
        },
        # -- Compliance badge --
        {
            "type": "compliance_badge",
            "id": "compliance_badge",
            "x_mm": 42.0,
            "y_mm": 70.0,
            "width_mm": 54.0,
            "height_mm": 6.0,
            "properties": {
                "compliant_color": "#2E7D32",
                "pending_color": "#F57F17",
                "non_compliant_color": "#C62828",
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "bold",
                "text_color": "#FFFFFF",
                "border_radius_mm": 1.5,
                "label_compliant": "EUDR COMPLIANT",
                "label_pending": "EUDR PENDING",
                "label_non_compliant": "NON-COMPLIANT",
            },
        },
        {
            "type": "line",
            "id": "mid_line",
            "x_mm": 4.0,
            "y_mm": 80.0,
            "width_mm": 92.0,
            "height_mm": 0.3,
            "properties": {"line_color": "#000000", "line_width_mm": 0.3},
        },
        # -- Carrier barcode area --
        {
            "type": "text",
            "id": "carrier_label",
            "x_mm": 4.0,
            "y_mm": 82.0,
            "width_mm": 92.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "bold",
                "text_color": "#333333",
                "alignment": "center",
                "placeholder": "CARRIER BARCODE",
            },
        },
        {
            "type": "barcode",
            "id": "carrier_barcode",
            "x_mm": 10.0,
            "y_mm": 88.0,
            "width_mm": 80.0,
            "height_mm": 25.0,
            "properties": {
                "symbology": "code128",
                "bar_width_mm": 0.33,
                "bar_color": "#000000",
                "background_color": "#FFFFFF",
                "show_text": True,
                "text_font_size_pt": 8,
                "placeholder": "{tracking_number}",
            },
        },
        # -- Footer --
        {
            "type": "line",
            "id": "footer_line",
            "x_mm": 4.0,
            "y_mm": 116.0,
            "width_mm": 92.0,
            "height_mm": 0.3,
            "properties": {"line_color": "#000000", "line_width_mm": 0.3},
        },
        {
            "type": "text",
            "id": "handling_instructions",
            "x_mm": 4.0,
            "y_mm": 118.0,
            "width_mm": 92.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "max_lines": 2,
                "placeholder": "{handling_instructions}",
            },
        },
        {
            "type": "text",
            "id": "verification_url",
            "x_mm": 4.0,
            "y_mm": 128.0,
            "width_mm": 92.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 5,
                "font_weight": "normal",
                "text_color": "#0066CC",
                "alignment": "center",
                "placeholder": "{verification_url}",
            },
        },
        {
            "type": "text",
            "id": "footer_date",
            "x_mm": 4.0,
            "y_mm": 134.0,
            "width_mm": 92.0,
            "height_mm": 4.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 5,
                "font_weight": "normal",
                "text_color": "#999999",
                "alignment": "center",
                "placeholder": "Generated: {generation_date}",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Template 3: Pallet Label (148x210mm A5)
# ---------------------------------------------------------------------------

PALLET_LABEL_TEMPLATE: Dict[str, Any] = {
    "name": "pallet_label",
    "display_name": "Pallet Label",
    "description": (
        "Pallet-level A5 label with large QR code readable from 2 metres, "
        "SSCC, batch codes, commodity information, handling instructions, "
        "and EUDR compliance status."
    ),
    "width_mm": 148.0,
    "height_mm": 210.0,
    "dpi": 300,
    "orientation": "portrait",
    "bleed_mm": 3.0,
    "margin_mm": 5.0,
    "background_color": "#FFFFFF",
    "version": "1.0",
    "elements": [
        {
            "type": "text",
            "id": "header_title",
            "x_mm": 5.0,
            "y_mm": 5.0,
            "width_mm": 138.0,
            "height_mm": 10.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 18,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "center",
                "placeholder": "PALLET LABEL",
            },
        },
        {
            "type": "line",
            "id": "header_line",
            "x_mm": 5.0,
            "y_mm": 17.0,
            "width_mm": 138.0,
            "height_mm": 0.5,
            "properties": {"line_color": "#000000", "line_width_mm": 0.5},
        },
        {
            "type": "qr_code",
            "id": "main_qr",
            "x_mm": 24.0,
            "y_mm": 22.0,
            "width_mm": 100.0,
            "height_mm": 100.0,
            "properties": {
                "error_correction": "H",
                "quiet_zone_modules": 4,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
                "min_module_size_mm": 1.0,
            },
        },
        {
            "type": "text",
            "id": "sscc_label",
            "x_mm": 5.0,
            "y_mm": 126.0,
            "width_mm": 138.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 14,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "center",
                "placeholder": "SSCC: {sscc}",
            },
        },
        {
            "type": "text",
            "id": "batch_codes",
            "x_mm": 5.0,
            "y_mm": 136.0,
            "width_mm": 66.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Batch: {batch_code}",
            },
        },
        {
            "type": "text",
            "id": "commodity_info",
            "x_mm": 75.0,
            "y_mm": 136.0,
            "width_mm": 68.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Commodity: {commodity}",
            },
        },
        {
            "type": "text",
            "id": "weight_info",
            "x_mm": 5.0,
            "y_mm": 146.0,
            "width_mm": 66.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Gross Weight: {gross_weight_kg} kg",
            },
        },
        {
            "type": "text",
            "id": "origin_info",
            "x_mm": 75.0,
            "y_mm": 146.0,
            "width_mm": 68.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Origin: {origin_country}",
            },
        },
        {
            "type": "compliance_badge",
            "id": "compliance_badge",
            "x_mm": 5.0,
            "y_mm": 158.0,
            "width_mm": 138.0,
            "height_mm": 10.0,
            "properties": {
                "compliant_color": "#2E7D32",
                "pending_color": "#F57F17",
                "non_compliant_color": "#C62828",
                "font_family": "DejaVuSans",
                "font_size_pt": 12,
                "font_weight": "bold",
                "text_color": "#FFFFFF",
                "border_radius_mm": 2.0,
                "label_compliant": "EUDR COMPLIANT - DEFORESTATION FREE",
                "label_pending": "EUDR COMPLIANCE PENDING",
                "label_non_compliant": "EUDR NON-COMPLIANT",
            },
        },
        {
            "type": "text",
            "id": "handling_instructions",
            "x_mm": 5.0,
            "y_mm": 172.0,
            "width_mm": 138.0,
            "height_mm": 12.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 9,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "max_lines": 3,
                "placeholder": "{handling_instructions}",
            },
        },
        {
            "type": "text",
            "id": "operator_info",
            "x_mm": 5.0,
            "y_mm": 188.0,
            "width_mm": 138.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "center",
                "placeholder": "Operator: {operator_id} | DDS: {dds_reference}",
            },
        },
        {
            "type": "text",
            "id": "verification_url",
            "x_mm": 5.0,
            "y_mm": 196.0,
            "width_mm": 138.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#0066CC",
                "alignment": "center",
                "placeholder": "{verification_url}",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Template 4: Container Label (297x210mm A4)
# ---------------------------------------------------------------------------

CONTAINER_LABEL_TEMPLATE: Dict[str, Any] = {
    "name": "container_label",
    "display_name": "Container Label",
    "description": (
        "Shipping container A4 label with multiple QR codes, container "
        "number, seal number, full traceability info, customs reference, "
        "and EUDR compliance status."
    ),
    "width_mm": 297.0,
    "height_mm": 210.0,
    "dpi": 300,
    "orientation": "landscape",
    "bleed_mm": 3.0,
    "margin_mm": 8.0,
    "background_color": "#FFFFFF",
    "version": "1.0",
    "elements": [
        {
            "type": "text",
            "id": "header_title",
            "x_mm": 8.0,
            "y_mm": 8.0,
            "width_mm": 281.0,
            "height_mm": 12.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 20,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "center",
                "placeholder": "CONTAINER TRACEABILITY LABEL",
            },
        },
        {
            "type": "line",
            "id": "header_line",
            "x_mm": 8.0,
            "y_mm": 22.0,
            "width_mm": 281.0,
            "height_mm": 0.5,
            "properties": {"line_color": "#000000", "line_width_mm": 0.5},
        },
        # -- Main QR code --
        {
            "type": "qr_code",
            "id": "main_qr",
            "x_mm": 8.0,
            "y_mm": 28.0,
            "width_mm": 70.0,
            "height_mm": 70.0,
            "properties": {
                "error_correction": "H",
                "quiet_zone_modules": 4,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
                "label": "Full Traceability",
            },
        },
        {
            "type": "text",
            "id": "qr_label_main",
            "x_mm": 8.0,
            "y_mm": 100.0,
            "width_mm": 70.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 7,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "center",
                "placeholder": "Full Traceability QR",
            },
        },
        # -- Secondary QR code (compact verification) --
        {
            "type": "qr_code",
            "id": "compact_qr",
            "x_mm": 8.0,
            "y_mm": 110.0,
            "width_mm": 40.0,
            "height_mm": 40.0,
            "properties": {
                "error_correction": "M",
                "quiet_zone_modules": 4,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
                "label": "Compact Verification",
            },
        },
        {
            "type": "text",
            "id": "qr_label_compact",
            "x_mm": 8.0,
            "y_mm": 152.0,
            "width_mm": 40.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "center",
                "placeholder": "Compact QR",
            },
        },
        # -- Tertiary QR code (consumer summary) --
        {
            "type": "qr_code",
            "id": "consumer_qr",
            "x_mm": 52.0,
            "y_mm": 110.0,
            "width_mm": 40.0,
            "height_mm": 40.0,
            "properties": {
                "error_correction": "M",
                "quiet_zone_modules": 4,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
                "label": "Consumer Summary",
            },
        },
        {
            "type": "text",
            "id": "qr_label_consumer",
            "x_mm": 52.0,
            "y_mm": 152.0,
            "width_mm": 40.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "center",
                "placeholder": "Consumer QR",
            },
        },
        # -- Container details --
        {
            "type": "text",
            "id": "container_number",
            "x_mm": 86.0,
            "y_mm": 28.0,
            "width_mm": 100.0,
            "height_mm": 10.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 14,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "left",
                "placeholder": "Container: {container_number}",
            },
        },
        {
            "type": "text",
            "id": "seal_number",
            "x_mm": 86.0,
            "y_mm": 40.0,
            "width_mm": 100.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Seal: {seal_number}",
            },
        },
        {
            "type": "text",
            "id": "customs_reference",
            "x_mm": 86.0,
            "y_mm": 50.0,
            "width_mm": 100.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Customs Ref: {customs_reference}",
            },
        },
        {
            "type": "text",
            "id": "commodity_details",
            "x_mm": 86.0,
            "y_mm": 60.0,
            "width_mm": 100.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Commodity: {commodity} | HS: {hs_code}",
            },
        },
        {
            "type": "text",
            "id": "origin_details",
            "x_mm": 86.0,
            "y_mm": 70.0,
            "width_mm": 100.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Origin: {origin_country} | Weight: {weight_kg} kg",
            },
        },
        {
            "type": "text",
            "id": "operator_dds",
            "x_mm": 86.0,
            "y_mm": 80.0,
            "width_mm": 100.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 9,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "placeholder": "Operator: {operator_id} | DDS: {dds_reference}",
            },
        },
        # -- Compliance badge (large) --
        {
            "type": "compliance_badge",
            "id": "compliance_badge",
            "x_mm": 86.0,
            "y_mm": 92.0,
            "width_mm": 100.0,
            "height_mm": 12.0,
            "properties": {
                "compliant_color": "#2E7D32",
                "pending_color": "#F57F17",
                "non_compliant_color": "#C62828",
                "font_family": "DejaVuSans",
                "font_size_pt": 12,
                "font_weight": "bold",
                "text_color": "#FFFFFF",
                "border_radius_mm": 2.0,
                "label_compliant": "EUDR COMPLIANT - DEFORESTATION FREE",
                "label_pending": "EUDR COMPLIANCE PENDING",
                "label_non_compliant": "EUDR NON-COMPLIANT",
            },
        },
        # -- Right side: batch list and traceability --
        {
            "type": "text",
            "id": "batch_list_header",
            "x_mm": 196.0,
            "y_mm": 28.0,
            "width_mm": 93.0,
            "height_mm": 8.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 10,
                "font_weight": "bold",
                "text_color": "#000000",
                "alignment": "left",
                "placeholder": "BATCH CODES:",
            },
        },
        {
            "type": "text",
            "id": "batch_list",
            "x_mm": 196.0,
            "y_mm": 38.0,
            "width_mm": 93.0,
            "height_mm": 60.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 8,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "left",
                "max_lines": 12,
                "placeholder": "{batch_list}",
            },
        },
        # -- Footer --
        {
            "type": "line",
            "id": "footer_line",
            "x_mm": 8.0,
            "y_mm": 168.0,
            "width_mm": 281.0,
            "height_mm": 0.5,
            "properties": {"line_color": "#000000", "line_width_mm": 0.5},
        },
        {
            "type": "text",
            "id": "verification_url",
            "x_mm": 8.0,
            "y_mm": 172.0,
            "width_mm": 281.0,
            "height_mm": 6.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 8,
                "font_weight": "normal",
                "text_color": "#0066CC",
                "alignment": "center",
                "placeholder": "Verify: {verification_url}",
            },
        },
        {
            "type": "text",
            "id": "footer_legal",
            "x_mm": 8.0,
            "y_mm": 180.0,
            "width_mm": 281.0,
            "height_mm": 10.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 6,
                "font_weight": "normal",
                "text_color": "#999999",
                "alignment": "center",
                "max_lines": 2,
                "placeholder": (
                    "EU Regulation 2023/1115 (EUDR). Due diligence statement "
                    "reference: {dds_reference}. Generated: {generation_date}."
                ),
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Template 5: Consumer Label (30x20mm)
# ---------------------------------------------------------------------------

CONSUMER_LABEL_TEMPLATE: Dict[str, Any] = {
    "name": "consumer_label",
    "display_name": "Consumer Label",
    "description": (
        "Compact consumer-facing label with QR code, minimal text, "
        "and scan prompt. Designed for retail product packaging "
        "where space is limited."
    ),
    "width_mm": 30.0,
    "height_mm": 20.0,
    "dpi": 600,
    "orientation": "landscape",
    "bleed_mm": 1.0,
    "margin_mm": 1.5,
    "background_color": "#FFFFFF",
    "version": "1.0",
    "elements": [
        {
            "type": "qr_code",
            "id": "main_qr",
            "x_mm": 1.5,
            "y_mm": 1.5,
            "width_mm": 14.0,
            "height_mm": 14.0,
            "properties": {
                "error_correction": "Q",
                "quiet_zone_modules": 2,
                "foreground_color": "#000000",
                "background_color": "#FFFFFF",
            },
        },
        {
            "type": "compliance_badge",
            "id": "compliance_dot",
            "x_mm": 17.0,
            "y_mm": 1.5,
            "width_mm": 11.5,
            "height_mm": 4.0,
            "properties": {
                "compliant_color": "#2E7D32",
                "pending_color": "#F57F17",
                "non_compliant_color": "#C62828",
                "font_family": "DejaVuSans",
                "font_size_pt": 3,
                "font_weight": "bold",
                "text_color": "#FFFFFF",
                "border_radius_mm": 1.0,
                "label_compliant": "DF",
                "label_pending": "P",
                "label_non_compliant": "NC",
            },
        },
        {
            "type": "text",
            "id": "scan_prompt",
            "x_mm": 17.0,
            "y_mm": 6.5,
            "width_mm": 11.5,
            "height_mm": 5.0,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 3,
                "font_weight": "normal",
                "text_color": "#333333",
                "alignment": "center",
                "max_lines": 2,
                "placeholder": "Scan for\norigin info",
            },
        },
        {
            "type": "text",
            "id": "origin_country",
            "x_mm": 17.0,
            "y_mm": 12.0,
            "width_mm": 11.5,
            "height_mm": 3.5,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 3,
                "font_weight": "normal",
                "text_color": "#666666",
                "alignment": "center",
                "max_lines": 1,
                "placeholder": "{origin_country}",
            },
        },
        {
            "type": "text",
            "id": "eudr_ref",
            "x_mm": 1.5,
            "y_mm": 16.5,
            "width_mm": 27.0,
            "height_mm": 2.5,
            "properties": {
                "font_family": "DejaVuSans",
                "font_size_pt": 2,
                "font_weight": "normal",
                "text_color": "#999999",
                "alignment": "center",
                "max_lines": 1,
                "placeholder": "EU 2023/1115",
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Template collections
# ---------------------------------------------------------------------------

ALL_TEMPLATES: List[Dict[str, Any]] = [
    PRODUCT_LABEL_TEMPLATE,
    SHIPPING_LABEL_TEMPLATE,
    PALLET_LABEL_TEMPLATE,
    CONTAINER_LABEL_TEMPLATE,
    CONSUMER_LABEL_TEMPLATE,
]

TEMPLATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    template["name"]: template for template in ALL_TEMPLATES
}


# ---------------------------------------------------------------------------
# Template validation
# ---------------------------------------------------------------------------

_VALID_ELEMENT_TYPES = frozenset({
    "qr_code",
    "text",
    "barcode",
    "image",
    "rectangle",
    "line",
    "compliance_badge",
})

_REQUIRED_ELEMENT_FIELDS = frozenset({
    "type", "id", "x_mm", "y_mm", "width_mm", "height_mm",
})


def validate_template(template: Dict[str, Any]) -> List[str]:
    """Validate a label template definition.

    Checks for required top-level fields, valid dimensions, and
    well-formed element definitions.

    Args:
        template: Template dictionary to validate.

    Returns:
        List of validation error strings. Empty list indicates a
        valid template.

    Example:
        >>> errors = validate_template(PRODUCT_LABEL_TEMPLATE)
        >>> assert len(errors) == 0
    """
    errors: List[str] = []

    # Required top-level fields
    required_top = {"name", "width_mm", "height_mm", "dpi", "elements"}
    for field in required_top:
        if field not in template:
            errors.append(f"Missing required field: {field}")

    if "name" in template and not template["name"]:
        errors.append("Template name must not be empty")

    if "width_mm" in template:
        if not isinstance(template["width_mm"], (int, float)):
            errors.append("width_mm must be a number")
        elif template["width_mm"] <= 0:
            errors.append("width_mm must be > 0")

    if "height_mm" in template:
        if not isinstance(template["height_mm"], (int, float)):
            errors.append("height_mm must be a number")
        elif template["height_mm"] <= 0:
            errors.append("height_mm must be > 0")

    if "dpi" in template:
        if not isinstance(template["dpi"], int):
            errors.append("dpi must be an integer")
        elif template["dpi"] < 72:
            errors.append("dpi must be >= 72")

    # Validate elements
    if "elements" in template:
        if not isinstance(template["elements"], list):
            errors.append("elements must be a list")
        else:
            element_ids = set()
            for i, element in enumerate(template["elements"]):
                if not isinstance(element, dict):
                    errors.append(f"Element [{i}] must be a dictionary")
                    continue

                for field in _REQUIRED_ELEMENT_FIELDS:
                    if field not in element:
                        errors.append(
                            f"Element [{i}] missing required field: {field}"
                        )

                if "type" in element:
                    if element["type"] not in _VALID_ELEMENT_TYPES:
                        errors.append(
                            f"Element [{i}] invalid type: "
                            f"{element['type']}"
                        )

                if "id" in element:
                    if element["id"] in element_ids:
                        errors.append(
                            f"Element [{i}] duplicate id: {element['id']}"
                        )
                    element_ids.add(element["id"])

                for coord in ("x_mm", "y_mm"):
                    if coord in element:
                        val = element[coord]
                        if not isinstance(val, (int, float)):
                            errors.append(
                                f"Element [{i}] {coord} must be a number"
                            )
                        elif val < 0:
                            errors.append(
                                f"Element [{i}] {coord} must be >= 0"
                            )

                for dim in ("width_mm", "height_mm"):
                    if dim in element:
                        val = element[dim]
                        if not isinstance(val, (int, float)):
                            errors.append(
                                f"Element [{i}] {dim} must be a number"
                            )
                        elif val <= 0:
                            errors.append(
                                f"Element [{i}] {dim} must be > 0"
                            )

    return errors


def get_template(name: str) -> Optional[Dict[str, Any]]:
    """Get a label template by name.

    Returns a deep copy to prevent mutation of the reference data.

    Args:
        name: Template name (product_label, shipping_label,
            pallet_label, container_label, consumer_label).

    Returns:
        Deep copy of the template dictionary, or None if not found.

    Example:
        >>> t = get_template("product_label")
        >>> t["name"]
        'product_label'
    """
    template = TEMPLATE_REGISTRY.get(name)
    if template is None:
        return None
    return copy.deepcopy(template)


def list_template_names() -> List[str]:
    """Return a sorted list of all available template names.

    Returns:
        List of template name strings.
    """
    return sorted(TEMPLATE_REGISTRY.keys())


def get_template_dimensions(name: str) -> Optional[Dict[str, float]]:
    """Get the physical dimensions for a template.

    Args:
        name: Template name.

    Returns:
        Dictionary with width_mm, height_mm, and dpi, or None
        if the template is not found.
    """
    template = TEMPLATE_REGISTRY.get(name)
    if template is None:
        return None
    return {
        "width_mm": template["width_mm"],
        "height_mm": template["height_mm"],
        "dpi": template["dpi"],
    }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "PRODUCT_LABEL_TEMPLATE",
    "SHIPPING_LABEL_TEMPLATE",
    "PALLET_LABEL_TEMPLATE",
    "CONTAINER_LABEL_TEMPLATE",
    "CONSUMER_LABEL_TEMPLATE",
    "ALL_TEMPLATES",
    "TEMPLATE_REGISTRY",
    "validate_template",
    "get_template",
    "list_template_names",
    "get_template_dimensions",
]
