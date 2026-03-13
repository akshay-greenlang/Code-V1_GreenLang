# -*- coding: utf-8 -*-
"""
EUDR Commodity Codes and Identifiers - AGENT-EUDR-014

Reference data for EUDR-regulated commodity codes per EU 2023/1115
Article 1 and Annex I.  Provides HS (Harmonized System), CN (Combined
Nomenclature), and TARIC code mappings for the seven EUDR commodity
groups plus their derived products.

Includes:
    - EUDR_COMMODITIES: 7 commodity groups with HS, CN, and TARIC codes
    - COMMODITY_CODE_PREFIX: Short codes for batch code prefixes
    - HS_CODE_RANGES: Nested dict by commodity -> product type -> HS range
    - is_eudr_commodity: Check if an HS code falls under EUDR
    - get_commodity_from_hs: Determine commodity type from an HS code
    - DERIVED_PRODUCTS: Mapping of derived products to source commodities
    - COUNTRY_RISK_CLASSIFICATION: Standard/high/low risk per Article 29

Data Sources:
    - EU 2023/1115 Annex I (commodity codes and CN codes)
    - World Customs Organization Harmonized System 2022 edition
    - EU Combined Nomenclature 2024 (Commission Implementing Regulation)
    - EU TARIC database (trade policy measures)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014 QR Code Generator (GL-EUDR-QRG-014)
Status: Production Ready
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EUDR Commodities with HS/CN/TARIC codes
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: Dict[str, Dict[str, Any]] = {
    "cattle": {
        "name": "Cattle",
        "description": (
            "Cattle and derived products including live bovine animals, "
            "beef, veal, and leather per EUDR Annex I."
        ),
        "hs_codes": [
            "0102",  # Live bovine animals
            "0201",  # Meat of bovine animals, fresh or chilled
            "0202",  # Meat of bovine animals, frozen
            "0206",  # Edible offal of bovine animals
            "0210",  # Meat and edible offal, salted, dried, smoked
            "1602",  # Prepared or preserved meat
            "4101",  # Raw hides and skins of bovine animals
            "4104",  # Tanned or crust hides of bovine animals
            "4107",  # Leather further prepared after tanning
            "4112",  # Leather further prepared (chamois, patent)
            "4113",  # Leather further prepared (goat or kid)
            "4114",  # Chamois leather; patent leather
            "4115",  # Composition leather
        ],
        "cn_codes": [
            "0102 21",  # Pure-bred breeding cattle
            "0102 29",  # Other live bovine animals
            "0102 31",  # Pure-bred breeding buffalo
            "0102 39",  # Other live buffalo
            "0102 90",  # Other bovine animals
            "0201 10",  # Carcasses and half-carcasses, fresh/chilled
            "0201 20",  # Other cuts with bone in, fresh/chilled
            "0201 30",  # Boneless, fresh or chilled
            "0202 10",  # Carcasses and half-carcasses, frozen
            "0202 20",  # Other cuts with bone in, frozen
            "0202 30",  # Boneless, frozen
        ],
        "taric_codes": [
            "0102 21 10",  # Heifers
            "0102 21 30",  # Cows
            "0102 21 90",  # Other pure-bred breeding cattle
        ],
        "eudr_annex_section": "A",
    },
    "cocoa": {
        "name": "Cocoa",
        "description": (
            "Cocoa beans and derived products including cocoa paste, "
            "cocoa butter, cocoa powder, and chocolate per EUDR Annex I."
        ),
        "hs_codes": [
            "1801",  # Cocoa beans, whole or broken, raw or roasted
            "1802",  # Cocoa shells, husks, skins, and other waste
            "1803",  # Cocoa paste, whether or not defatted
            "1804",  # Cocoa butter, fat, and oil
            "1805",  # Cocoa powder, not containing added sugar
            "1806",  # Chocolate and other food preparations with cocoa
        ],
        "cn_codes": [
            "1801 00",  # Cocoa beans, whole or broken
            "1802 00",  # Cocoa shells, husks, skins
            "1803 10",  # Cocoa paste, not defatted
            "1803 20",  # Cocoa paste, wholly or partly defatted
            "1804 00",  # Cocoa butter, fat, and oil
            "1805 00",  # Cocoa powder without sugar
            "1806 10",  # Cocoa powder with sugar
            "1806 20",  # Other preparations, blocks > 2kg
            "1806 31",  # Filled chocolate blocks/bars
            "1806 32",  # Unfilled chocolate blocks/bars
            "1806 90",  # Other chocolate preparations
        ],
        "taric_codes": [
            "1801 00 00",  # Cocoa beans
            "1803 10 00",  # Cocoa paste, not defatted
            "1803 20 00",  # Cocoa paste, defatted
        ],
        "eudr_annex_section": "B",
    },
    "coffee": {
        "name": "Coffee",
        "description": (
            "Coffee beans and derived products including roasted coffee, "
            "decaffeinated coffee, and instant coffee per EUDR Annex I."
        ),
        "hs_codes": [
            "0901",  # Coffee, whether or not roasted/decaffeinated
        ],
        "cn_codes": [
            "0901 11",  # Coffee, not roasted, not decaffeinated
            "0901 12",  # Coffee, not roasted, decaffeinated
            "0901 21",  # Coffee, roasted, not decaffeinated
            "0901 22",  # Coffee, roasted, decaffeinated
            "0901 90",  # Other coffee (husks, skins, substitutes)
        ],
        "taric_codes": [
            "0901 11 00",  # Arabica and Robusta, not roasted
            "0901 12 00",  # Decaffeinated, not roasted
            "0901 21 00",  # Roasted, not decaffeinated
            "0901 22 00",  # Roasted, decaffeinated
        ],
        "eudr_annex_section": "C",
    },
    "oil_palm": {
        "name": "Oil palm",
        "description": (
            "Oil palm and derived products including palm oil, palm "
            "kernel oil, and palm kernel cake per EUDR Annex I."
        ),
        "hs_codes": [
            "1207",  # Oil palm fruit and kernels
            "1511",  # Palm oil and its fractions
            "1513",  # Coconut/palm kernel/babassu oil and fractions
            "2306",  # Oil-cake and other solid residues (palm kernel)
            "3823",  # Industrial monocarboxylic fatty acids (palm)
            "3826",  # Biodiesel from palm oil
        ],
        "cn_codes": [
            "1511 10",  # Crude palm oil
            "1511 90",  # Other palm oil and fractions
            "1513 21",  # Crude palm kernel or babassu oil
            "1513 29",  # Other palm kernel or babassu oil
            "2306 60",  # Oil-cake from palm nuts or kernels
        ],
        "taric_codes": [
            "1511 10 10",  # Crude palm oil for technical use
            "1511 10 90",  # Crude palm oil, other
            "1511 90 11",  # Palm oil solid fractions
            "1513 21 11",  # Crude palm kernel oil for technical use
        ],
        "eudr_annex_section": "D",
    },
    "rubber": {
        "name": "Rubber",
        "description": (
            "Natural rubber and derived products including latex, "
            "tyres, tubes, and rubber articles per EUDR Annex I."
        ),
        "hs_codes": [
            "4001",  # Natural rubber in primary forms
            "4005",  # Compounded rubber, unvulcanised
            "4006",  # Unvulcanised rubber in other forms
            "4007",  # Vulcanised rubber thread and cord
            "4008",  # Plates, sheets, strip, rods of vulcanised rubber
            "4009",  # Tubes, pipes, and hoses of vulcanised rubber
            "4010",  # Conveyor or transmission belts
            "4011",  # New pneumatic tyres, of rubber
            "4012",  # Retreaded or used pneumatic tyres
            "4013",  # Inner tubes, of rubber
            "4014",  # Hygienic or pharmaceutical rubber articles
            "4015",  # Articles of apparel of vulcanised rubber
            "4016",  # Other articles of vulcanised rubber
            "4017",  # Hard rubber (ebonite) in all forms
        ],
        "cn_codes": [
            "4001 10",  # Natural rubber latex
            "4001 21",  # Smoked sheets
            "4001 22",  # Technically specified natural rubber (TSNR)
            "4001 29",  # Other natural rubber
            "4001 30",  # Balata, gutta-percha
            "4011 10",  # New pneumatic tyres for motor cars
            "4011 20",  # New pneumatic tyres for buses/lorries
        ],
        "taric_codes": [
            "4001 10 00",  # Natural rubber latex
            "4001 21 00",  # Smoked sheets
            "4001 22 00",  # TSNR
        ],
        "eudr_annex_section": "E",
    },
    "soya": {
        "name": "Soya",
        "description": (
            "Soybeans and derived products including soya meal, soya "
            "oil, soya flour, and soya protein per EUDR Annex I."
        ),
        "hs_codes": [
            "1201",  # Soybeans, whether or not broken
            "1208",  # Flours and meals of oil seeds (soya)
            "1507",  # Soya-bean oil and its fractions
            "2304",  # Oil-cake from soya-bean oil extraction
        ],
        "cn_codes": [
            "1201 10",  # Seed for sowing
            "1201 90",  # Other soybeans
            "1208 10",  # Soya bean flour and meal
            "1507 10",  # Crude soya-bean oil
            "1507 90",  # Other soya-bean oil and fractions
            "2304 00",  # Oil-cake and other solid residues of soya
        ],
        "taric_codes": [
            "1201 10 00",  # Soybeans for sowing
            "1201 90 00",  # Other soybeans
            "1507 10 10",  # Crude soya-bean oil for technical use
            "1507 10 90",  # Crude soya-bean oil, other
        ],
        "eudr_annex_section": "F",
    },
    "wood": {
        "name": "Wood",
        "description": (
            "Wood and derived products including timber, sawn wood, "
            "wood-based panels, pulp, paper, printed matter, and "
            "wooden furniture per EUDR Annex I."
        ),
        "hs_codes": [
            "4401",  # Fuel wood, wood chips, sawdust
            "4402",  # Charcoal
            "4403",  # Wood in the rough
            "4404",  # Hoopwood, poles, stakes
            "4405",  # Wood wool and wood flour
            "4406",  # Railway or tramway sleepers
            "4407",  # Wood sawn or chipped lengthwise, > 6mm thick
            "4408",  # Veneer sheets and sheets for plywood
            "4409",  # Wood continuously shaped (tongued, grooved)
            "4410",  # Particle board and OSB
            "4411",  # Fibreboard of wood (MDF, HDF)
            "4412",  # Plywood, veneered panels
            "4413",  # Densified wood
            "4414",  # Wooden frames for paintings
            "4415",  # Packing cases, drums, pallets of wood
            "4416",  # Casks, barrels, vats of wood
            "4417",  # Tools and tool bodies of wood
            "4418",  # Builders' joinery and carpentry of wood
            "4419",  # Tableware and kitchenware of wood
            "4420",  # Wood marquetry, caskets, ornaments
            "4421",  # Other articles of wood
            "4701",  # Mechanical wood pulp
            "4702",  # Chemical wood pulp, dissolving grades
            "4703",  # Chemical wood pulp, soda or sulphate
            "4704",  # Chemical wood pulp, sulphite
            "4705",  # Wood pulp from mechanical and chemical processes
            "4706",  # Pulps of fibres from recovered paper
            "4707",  # Recovered paper or paperboard
            "4801",  # Newsprint
            "4802",  # Uncoated paper and paperboard
            "4803",  # Tissue, towel, napkin stock
            "4804",  # Uncoated kraft paper and paperboard
            "4805",  # Other uncoated paper and paperboard
            "4806",  # Vegetable parchment, greaseproof papers
            "4807",  # Composite paper and paperboard
            "4808",  # Corrugated paper and paperboard
            "4809",  # Carbon or self-copy paper
            "4810",  # Coated paper and paperboard
            "4811",  # Paper and paperboard, coated, impregnated
            "4812",  # Filter blocks, slabs of paper pulp
            "4813",  # Cigarette paper
            "4814",  # Wallpaper
            "4816",  # Carbon paper, self-copy paper, duplicator stencils
            "4817",  # Envelopes, letter cards, postcards
            "4818",  # Toilet paper, tissues, towels
            "4819",  # Cartons, boxes, cases of paper/paperboard
            "4820",  # Registers, notebooks, diaries
            "4821",  # Paper or paperboard labels
            "4822",  # Bobbins, spools, cops of paper
            "4823",  # Other paper, paperboard, cellulose wadding
            "9401",  # Seats (with wooden frames)
            "9403",  # Other furniture (wooden)
        ],
        "cn_codes": [
            "4401 11",  # Fuel wood, coniferous, logs
            "4401 12",  # Fuel wood, non-coniferous, logs
            "4403 11",  # Wood in the rough, coniferous, treated
            "4403 12",  # Wood in the rough, non-coniferous, treated
            "4403 21",  # Pine (Pinus spp.) in the rough
            "4403 22",  # Fir/spruce (Abies/Picea) in the rough
            "4403 23",  # Other coniferous wood in the rough
            "4403 41",  # Dark Red Meranti, Light Red Meranti
            "4403 49",  # Other tropical wood in the rough
            "4407 11",  # Pine sawn or chipped, > 6mm
            "4407 12",  # Fir/spruce sawn or chipped, > 6mm
            "4407 19",  # Other coniferous sawn wood
            "4407 21",  # Mahogany sawn
            "4407 29",  # Other tropical wood sawn
            "9401 61",  # Upholstered seats with wooden frames
            "9401 69",  # Other seats with wooden frames
            "9403 30",  # Wooden furniture for offices
            "9403 40",  # Wooden furniture for kitchens
            "9403 50",  # Wooden furniture for bedrooms
            "9403 60",  # Other wooden furniture
        ],
        "taric_codes": [
            "4403 21 10",  # Pine, diameter > 15cm
            "4403 21 90",  # Pine, other
            "4407 11 10",  # Pine sawn, thickness > 6mm
        ],
        "eudr_annex_section": "G",
    },
}


# ---------------------------------------------------------------------------
# Short commodity code prefixes for batch codes
# ---------------------------------------------------------------------------

COMMODITY_CODE_PREFIX: Dict[str, str] = {
    "cattle": "CAT",
    "cocoa": "COC",
    "coffee": "COF",
    "oil_palm": "OPM",
    "rubber": "RUB",
    "soya": "SOY",
    "wood": "WOD",
}


# ---------------------------------------------------------------------------
# HS code ranges by commodity and product type
# ---------------------------------------------------------------------------

HS_CODE_RANGES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "cattle": {
        "live_animals": ("0102", "0102"),
        "fresh_meat": ("0201", "0201"),
        "frozen_meat": ("0202", "0202"),
        "offal": ("0206", "0206"),
        "preserved_meat": ("0210", "0210"),
        "prepared_meat": ("1602", "1602"),
        "hides_raw": ("4101", "4101"),
        "hides_tanned": ("4104", "4104"),
        "leather": ("4107", "4115"),
    },
    "cocoa": {
        "beans": ("1801", "1801"),
        "shells_waste": ("1802", "1802"),
        "paste": ("1803", "1803"),
        "butter": ("1804", "1804"),
        "powder": ("1805", "1805"),
        "chocolate": ("1806", "1806"),
    },
    "coffee": {
        "beans_and_products": ("0901", "0901"),
    },
    "oil_palm": {
        "seeds_kernels": ("1207", "1207"),
        "palm_oil": ("1511", "1511"),
        "kernel_oil": ("1513", "1513"),
        "oil_cake": ("2306", "2306"),
        "fatty_acids": ("3823", "3823"),
        "biodiesel": ("3826", "3826"),
    },
    "rubber": {
        "natural_primary": ("4001", "4001"),
        "compounded": ("4005", "4006"),
        "vulcanised_products": ("4007", "4017"),
    },
    "soya": {
        "beans": ("1201", "1201"),
        "flour_meal": ("1208", "1208"),
        "oil": ("1507", "1507"),
        "oil_cake": ("2304", "2304"),
    },
    "wood": {
        "fuel_wood": ("4401", "4402"),
        "wood_rough": ("4403", "4406"),
        "sawn_wood": ("4407", "4413"),
        "wood_articles": ("4414", "4421"),
        "pulp": ("4701", "4707"),
        "paper": ("4801", "4823"),
        "furniture": ("9401", "9403"),
    },
}


# ---------------------------------------------------------------------------
# Derived products mapped to source commodities
# ---------------------------------------------------------------------------

DERIVED_PRODUCTS: Dict[str, Dict[str, Any]] = {
    # Cattle-derived
    "beef": {"commodity": "cattle", "hs_prefix": "0201"},
    "veal": {"commodity": "cattle", "hs_prefix": "0202"},
    "leather_goods": {"commodity": "cattle", "hs_prefix": "4107"},
    "leather_shoes": {"commodity": "cattle", "hs_prefix": "6403"},
    "leather_bags": {"commodity": "cattle", "hs_prefix": "4202"},
    "gelatin": {"commodity": "cattle", "hs_prefix": "3503"},
    "tallow": {"commodity": "cattle", "hs_prefix": "1502"},
    # Cocoa-derived
    "chocolate_bars": {"commodity": "cocoa", "hs_prefix": "1806"},
    "cocoa_butter": {"commodity": "cocoa", "hs_prefix": "1804"},
    "cocoa_powder": {"commodity": "cocoa", "hs_prefix": "1805"},
    "cocoa_paste": {"commodity": "cocoa", "hs_prefix": "1803"},
    "chocolate_spread": {"commodity": "cocoa", "hs_prefix": "1806"},
    # Coffee-derived
    "roasted_coffee": {"commodity": "coffee", "hs_prefix": "0901"},
    "instant_coffee": {"commodity": "coffee", "hs_prefix": "2101"},
    "decaf_coffee": {"commodity": "coffee", "hs_prefix": "0901"},
    "coffee_extract": {"commodity": "coffee", "hs_prefix": "2101"},
    # Oil palm-derived
    "palm_oil": {"commodity": "oil_palm", "hs_prefix": "1511"},
    "palm_kernel_oil": {"commodity": "oil_palm", "hs_prefix": "1513"},
    "palm_biodiesel": {"commodity": "oil_palm", "hs_prefix": "3826"},
    "palm_fatty_acids": {"commodity": "oil_palm", "hs_prefix": "3823"},
    "palm_kernel_cake": {"commodity": "oil_palm", "hs_prefix": "2306"},
    "margarine": {"commodity": "oil_palm", "hs_prefix": "1517"},
    # Rubber-derived
    "natural_latex": {"commodity": "rubber", "hs_prefix": "4001"},
    "rubber_tyres": {"commodity": "rubber", "hs_prefix": "4011"},
    "rubber_tubes": {"commodity": "rubber", "hs_prefix": "4009"},
    "rubber_gloves": {"commodity": "rubber", "hs_prefix": "4015"},
    "conveyor_belts": {"commodity": "rubber", "hs_prefix": "4010"},
    # Soya-derived
    "soya_oil": {"commodity": "soya", "hs_prefix": "1507"},
    "soya_meal": {"commodity": "soya", "hs_prefix": "2304"},
    "soya_flour": {"commodity": "soya", "hs_prefix": "1208"},
    "soya_protein": {"commodity": "soya", "hs_prefix": "2106"},
    "tofu": {"commodity": "soya", "hs_prefix": "2106"},
    # Wood-derived
    "timber": {"commodity": "wood", "hs_prefix": "4407"},
    "plywood": {"commodity": "wood", "hs_prefix": "4412"},
    "particle_board": {"commodity": "wood", "hs_prefix": "4410"},
    "mdf": {"commodity": "wood", "hs_prefix": "4411"},
    "wood_pulp": {"commodity": "wood", "hs_prefix": "4703"},
    "paper": {"commodity": "wood", "hs_prefix": "4802"},
    "cardboard": {"commodity": "wood", "hs_prefix": "4808"},
    "newsprint": {"commodity": "wood", "hs_prefix": "4801"},
    "tissue_paper": {"commodity": "wood", "hs_prefix": "4818"},
    "wooden_furniture": {"commodity": "wood", "hs_prefix": "9403"},
    "charcoal": {"commodity": "wood", "hs_prefix": "4402"},
    "wooden_pallets": {"commodity": "wood", "hs_prefix": "4415"},
}


# ---------------------------------------------------------------------------
# Country risk classification per EUDR Article 29
# ---------------------------------------------------------------------------

COUNTRY_RISK_CLASSIFICATION: Dict[str, Dict[str, Any]] = {
    # -- High Risk Countries --
    # (Countries with significant deforestation or forest degradation)
    "BR": {"risk": "high", "name": "Brazil", "region": "South America"},
    "ID": {"risk": "high", "name": "Indonesia", "region": "Southeast Asia"},
    "CD": {"risk": "high", "name": "Democratic Republic of Congo", "region": "Central Africa"},
    "CG": {"risk": "high", "name": "Republic of Congo", "region": "Central Africa"},
    "CM": {"risk": "high", "name": "Cameroon", "region": "Central Africa"},
    "GA": {"risk": "high", "name": "Gabon", "region": "Central Africa"},
    "BO": {"risk": "high", "name": "Bolivia", "region": "South America"},
    "PY": {"risk": "high", "name": "Paraguay", "region": "South America"},
    "PE": {"risk": "high", "name": "Peru", "region": "South America"},
    "CO": {"risk": "high", "name": "Colombia", "region": "South America"},
    "MY": {"risk": "high", "name": "Malaysia", "region": "Southeast Asia"},
    "MM": {"risk": "high", "name": "Myanmar", "region": "Southeast Asia"},
    "LA": {"risk": "high", "name": "Laos", "region": "Southeast Asia"},
    "KH": {"risk": "high", "name": "Cambodia", "region": "Southeast Asia"},
    "PG": {"risk": "high", "name": "Papua New Guinea", "region": "Oceania"},
    "GH": {"risk": "high", "name": "Ghana", "region": "West Africa"},
    "CI": {"risk": "high", "name": "Cote d'Ivoire", "region": "West Africa"},
    "NG": {"risk": "high", "name": "Nigeria", "region": "West Africa"},
    "MZ": {"risk": "high", "name": "Mozambique", "region": "East Africa"},
    "TZ": {"risk": "high", "name": "Tanzania", "region": "East Africa"},
    "MG": {"risk": "high", "name": "Madagascar", "region": "East Africa"},
    "EC": {"risk": "high", "name": "Ecuador", "region": "South America"},
    "VE": {"risk": "high", "name": "Venezuela", "region": "South America"},
    "HN": {"risk": "high", "name": "Honduras", "region": "Central America"},
    "GT": {"risk": "high", "name": "Guatemala", "region": "Central America"},
    "NI": {"risk": "high", "name": "Nicaragua", "region": "Central America"},
    # -- Low Risk Countries --
    # (Countries with strong forest governance and monitoring)
    "FI": {"risk": "low", "name": "Finland", "region": "Northern Europe"},
    "SE": {"risk": "low", "name": "Sweden", "region": "Northern Europe"},
    "NO": {"risk": "low", "name": "Norway", "region": "Northern Europe"},
    "AT": {"risk": "low", "name": "Austria", "region": "Central Europe"},
    "CH": {"risk": "low", "name": "Switzerland", "region": "Central Europe"},
    "DE": {"risk": "low", "name": "Germany", "region": "Central Europe"},
    "FR": {"risk": "low", "name": "France", "region": "Western Europe"},
    "NL": {"risk": "low", "name": "Netherlands", "region": "Western Europe"},
    "BE": {"risk": "low", "name": "Belgium", "region": "Western Europe"},
    "DK": {"risk": "low", "name": "Denmark", "region": "Northern Europe"},
    "IE": {"risk": "low", "name": "Ireland", "region": "Western Europe"},
    "LU": {"risk": "low", "name": "Luxembourg", "region": "Western Europe"},
    "CA": {"risk": "low", "name": "Canada", "region": "North America"},
    "US": {"risk": "low", "name": "United States", "region": "North America"},
    "JP": {"risk": "low", "name": "Japan", "region": "East Asia"},
    "KR": {"risk": "low", "name": "South Korea", "region": "East Asia"},
    "AU": {"risk": "low", "name": "Australia", "region": "Oceania"},
    "NZ": {"risk": "low", "name": "New Zealand", "region": "Oceania"},
    "GB": {"risk": "low", "name": "United Kingdom", "region": "Western Europe"},
    "CZ": {"risk": "low", "name": "Czech Republic", "region": "Central Europe"},
    "PL": {"risk": "low", "name": "Poland", "region": "Central Europe"},
    "IT": {"risk": "low", "name": "Italy", "region": "Southern Europe"},
    "ES": {"risk": "low", "name": "Spain", "region": "Southern Europe"},
    "PT": {"risk": "low", "name": "Portugal", "region": "Southern Europe"},
    # -- Standard Risk Countries --
    # (All countries not classified as high or low are standard risk)
    "IN": {"risk": "standard", "name": "India", "region": "South Asia"},
    "CN": {"risk": "standard", "name": "China", "region": "East Asia"},
    "TH": {"risk": "standard", "name": "Thailand", "region": "Southeast Asia"},
    "VN": {"risk": "standard", "name": "Vietnam", "region": "Southeast Asia"},
    "PH": {"risk": "standard", "name": "Philippines", "region": "Southeast Asia"},
    "ET": {"risk": "standard", "name": "Ethiopia", "region": "East Africa"},
    "KE": {"risk": "standard", "name": "Kenya", "region": "East Africa"},
    "UG": {"risk": "standard", "name": "Uganda", "region": "East Africa"},
    "RW": {"risk": "standard", "name": "Rwanda", "region": "East Africa"},
    "MX": {"risk": "standard", "name": "Mexico", "region": "Central America"},
    "CR": {"risk": "standard", "name": "Costa Rica", "region": "Central America"},
    "PA": {"risk": "standard", "name": "Panama", "region": "Central America"},
    "AR": {"risk": "standard", "name": "Argentina", "region": "South America"},
    "CL": {"risk": "standard", "name": "Chile", "region": "South America"},
    "UY": {"risk": "standard", "name": "Uruguay", "region": "South America"},
    "LK": {"risk": "standard", "name": "Sri Lanka", "region": "South Asia"},
    "BD": {"risk": "standard", "name": "Bangladesh", "region": "South Asia"},
    "TR": {"risk": "standard", "name": "Turkey", "region": "Western Asia"},
    "RU": {"risk": "standard", "name": "Russia", "region": "Northern Asia"},
    "ZA": {"risk": "standard", "name": "South Africa", "region": "Southern Africa"},
    "SN": {"risk": "standard", "name": "Senegal", "region": "West Africa"},
}


# ---------------------------------------------------------------------------
# Valid EUDR commodity names
# ---------------------------------------------------------------------------

VALID_COMMODITY_NAMES: List[str] = sorted(EUDR_COMMODITIES.keys())


# ---------------------------------------------------------------------------
# Lookup: all HS code prefixes across all commodities
# ---------------------------------------------------------------------------

_ALL_HS_PREFIXES: Dict[str, str] = {}
for _commodity_name, _commodity_data in EUDR_COMMODITIES.items():
    for _hs_code in _commodity_data["hs_codes"]:
        _ALL_HS_PREFIXES[_hs_code] = _commodity_name


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def is_eudr_commodity(hs_code: str) -> bool:
    """Check if an HS code falls under an EUDR-regulated commodity.

    Strips whitespace and tests the first 4 characters (HS heading)
    against all registered EUDR commodity HS codes.

    Args:
        hs_code: HS code string (4, 6, or 8 digits, with or without
            spaces/dots).

    Returns:
        True if the HS code corresponds to an EUDR-regulated commodity.

    Example:
        >>> is_eudr_commodity("0901 21 00")
        True
        >>> is_eudr_commodity("7208")
        False
    """
    cleaned = re.sub(r"[\s.\-]", "", hs_code.strip())
    if len(cleaned) < 4:
        return False
    heading = cleaned[:4]
    return heading in _ALL_HS_PREFIXES


def get_commodity_from_hs(hs_code: str) -> Optional[str]:
    """Determine the EUDR commodity type from an HS code.

    Args:
        hs_code: HS code string (4, 6, or 8 digits, with or without
            spaces/dots).

    Returns:
        Commodity name (cattle, cocoa, coffee, oil_palm, rubber, soya,
        wood) or None if the HS code does not match any EUDR commodity.

    Example:
        >>> get_commodity_from_hs("1801 00 00")
        'cocoa'
        >>> get_commodity_from_hs("7208 10 00")
        >>> # Returns None
    """
    cleaned = re.sub(r"[\s.\-]", "", hs_code.strip())
    if len(cleaned) < 4:
        return None
    heading = cleaned[:4]
    return _ALL_HS_PREFIXES.get(heading)


def get_commodity_prefix(commodity: str) -> Optional[str]:
    """Get the short batch-code prefix for a commodity.

    Args:
        commodity: EUDR commodity name.

    Returns:
        Three-letter prefix (CAT, COC, COF, OPM, RUB, SOY, WOD) or
        None if the commodity is not recognized.

    Example:
        >>> get_commodity_prefix("coffee")
        'COF'
    """
    return COMMODITY_CODE_PREFIX.get(commodity)


def get_country_risk(country_code: str) -> str:
    """Get the EUDR risk classification for a country.

    Per EUDR Article 29, countries are classified as standard, high,
    or low risk based on deforestation rates and governance.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Risk level string: "high", "low", or "standard".
        Defaults to "standard" if the country is not found.

    Example:
        >>> get_country_risk("BR")
        'high'
        >>> get_country_risk("FI")
        'low'
        >>> get_country_risk("XX")
        'standard'
    """
    upper_code = country_code.upper().strip()
    entry = COUNTRY_RISK_CLASSIFICATION.get(upper_code)
    if entry is None:
        return "standard"
    return entry["risk"]


def get_hs_codes_for_commodity(commodity: str) -> List[str]:
    """Get all HS codes for a given EUDR commodity.

    Args:
        commodity: EUDR commodity name.

    Returns:
        List of HS code headings (4-digit strings), or empty list
        if the commodity is not recognized.

    Example:
        >>> get_hs_codes_for_commodity("coffee")
        ['0901']
    """
    entry = EUDR_COMMODITIES.get(commodity)
    if entry is None:
        return []
    return list(entry["hs_codes"])


def validate_commodity(commodity: str) -> bool:
    """Check if a commodity name is a valid EUDR commodity.

    Args:
        commodity: Commodity name to validate.

    Returns:
        True if the commodity is one of the 7 EUDR-regulated
        commodities.

    Example:
        >>> validate_commodity("cocoa")
        True
        >>> validate_commodity("cotton")
        False
    """
    return commodity in EUDR_COMMODITIES


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "EUDR_COMMODITIES",
    "COMMODITY_CODE_PREFIX",
    "HS_CODE_RANGES",
    "DERIVED_PRODUCTS",
    "COUNTRY_RISK_CLASSIFICATION",
    "VALID_COMMODITY_NAMES",
    "is_eudr_commodity",
    "get_commodity_from_hs",
    "get_commodity_prefix",
    "get_country_risk",
    "get_hs_codes_for_commodity",
    "validate_commodity",
]
