# -*- coding: utf-8 -*-
"""
Commodity Conversion & Yield Factors - AGENT-EUDR-009 Chain of Custody Agent

Provides commodity-specific conversion factors, yield ratios, loss tolerances,
and by-product mappings for the Chain of Custody Agent. These deterministic
reference values drive the mass balance engine and transformation recording,
ensuring zero-hallucination verification of yield claims throughout the
supply chain.

Each conversion factor entry is keyed by (commodity, process_type) and
contains:
    - yield_ratio: dict with min, typical, max ratios (0.0-1.0)
    - source: Authoritative data source citation
    - loss_tolerance_pct: Acceptable loss beyond typical yield (percentage points)
    - by_products: list of (product_name, typical_ratio) tuples

Data Sources:
    - International Cocoa Organization (ICCO) Technical Papers 2024
    - Malaysian Palm Oil Board (MPOB) Annual Report 2024
    - International Coffee Organization (ICO) Statistics 2024
    - USDA Oil Crops Yearbook 2024
    - FAO Forestry Statistics 2024
    - International Rubber Study Group (IRSG) 2024
    - OECD-FAO Agricultural Outlook 2024-2033

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009 (Chain of Custody)
Agent ID: GL-EUDR-COC-009
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data version for provenance tracking
# ---------------------------------------------------------------------------

DATA_VERSION = "1.0.0"
DATA_SOURCE = "GreenLang Reference Data v1.0.0 (2026-03)"

# ---------------------------------------------------------------------------
# Conversion Factors Data
# ---------------------------------------------------------------------------
# Keys are (commodity, process_type) tuples serialized as "commodity::process"
# for dictionary compatibility.
#
# yield_ratio: min/typical/max as fraction of input weight that becomes the
#              named output product (0.0 = total loss, 1.0 = no loss).
# loss_tolerance_pct: acceptable deviation above expected loss. If actual
#                     loss exceeds (1 - typical_yield) + tolerance, an alert
#                     fires.
# by_products: list of (name, typical_ratio) for secondary outputs. The
#              sum of primary yield + by_product ratios + expected waste
#              should approximate 1.0.

CONVERSION_FACTORS: Dict[str, Dict[str, Any]] = {
    # ===================================================================
    # COCOA (Theobroma cacao)
    # ===================================================================
    "cocoa::beans_to_nibs": {
        "commodity": "cocoa",
        "process_type": "beans_to_nibs",
        "description": "Cocoa bean winnowing: removal of shell to extract nibs",
        "input_product": "cocoa_beans",
        "output_product": "cocoa_nibs",
        "yield_ratio": {"min": 0.84, "typical": 0.87, "max": 0.89},
        "source": "ICCO Technical Paper: Cocoa Bean Processing (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("cocoa_shell", 0.13),
        ],
        "notes": (
            "Yield depends on bean size, fermentation quality, and "
            "winnowing equipment efficiency. Forastero beans typically "
            "yield slightly lower than Criollo or Trinitario."
        ),
    },
    "cocoa::nibs_to_liquor": {
        "commodity": "cocoa",
        "process_type": "nibs_to_liquor",
        "description": "Grinding cocoa nibs into cocoa liquor (mass/paste)",
        "input_product": "cocoa_nibs",
        "output_product": "cocoa_liquor",
        "yield_ratio": {"min": 0.78, "typical": 0.80, "max": 0.82},
        "source": "ICCO Technical Paper: Cocoa Bean Processing (2024)",
        "loss_tolerance_pct": 2.5,
        "by_products": [],
        "notes": (
            "Minimal by-products; losses due to equipment retention "
            "and moisture evaporation during grinding."
        ),
    },
    "cocoa::liquor_to_butter": {
        "commodity": "cocoa",
        "process_type": "liquor_to_butter",
        "description": "Hydraulic pressing cocoa liquor to extract cocoa butter",
        "input_product": "cocoa_liquor",
        "output_product": "cocoa_butter",
        "yield_ratio": {"min": 0.42, "typical": 0.45, "max": 0.48},
        "source": "ICCO Technical Paper: Cocoa Butter Extraction (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("cocoa_press_cake", 0.55),
        ],
        "notes": (
            "Press cake is the primary by-product, subsequently ground "
            "into cocoa powder. Fat content of input liquor determines "
            "butter yield; typically 52-56% fat in well-fermented beans."
        ),
    },
    "cocoa::liquor_to_powder": {
        "commodity": "cocoa",
        "process_type": "liquor_to_powder",
        "description": "Pressing and grinding cocoa liquor to produce cocoa powder",
        "input_product": "cocoa_liquor",
        "output_product": "cocoa_powder",
        "yield_ratio": {"min": 0.52, "typical": 0.55, "max": 0.58},
        "source": "ICCO Technical Paper: Cocoa Powder Production (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("cocoa_butter", 0.45),
        ],
        "notes": (
            "Complementary to butter extraction. Total recovery of "
            "butter + powder should approximate 100% of liquor input."
        ),
    },
    "cocoa::beans_to_shell": {
        "commodity": "cocoa",
        "process_type": "beans_to_shell",
        "description": "Cocoa bean shell separation (by-product tracking)",
        "input_product": "cocoa_beans",
        "output_product": "cocoa_shell",
        "yield_ratio": {"min": 0.11, "typical": 0.13, "max": 0.16},
        "source": "ICCO Technical Paper: Cocoa Bean Processing (2024)",
        "loss_tolerance_pct": 2.0,
        "by_products": [
            ("cocoa_nibs", 0.87),
        ],
        "notes": (
            "Shell is the by-product of winnowing. Used in animal feed, "
            "mulch, or bioenergy. Tracked for mass balance completeness."
        ),
    },

    # ===================================================================
    # PALM OIL (Elaeis guineensis)
    # ===================================================================
    "palm_oil::ffb_to_cpo": {
        "commodity": "palm_oil",
        "process_type": "ffb_to_cpo",
        "description": "Milling fresh fruit bunches (FFB) to crude palm oil (CPO)",
        "input_product": "fresh_fruit_bunches",
        "output_product": "crude_palm_oil",
        "yield_ratio": {"min": 0.20, "typical": 0.22, "max": 0.23},
        "source": "MPOB Annual Report: Oil Extraction Rates (2024)",
        "loss_tolerance_pct": 2.0,
        "by_products": [
            ("palm_kernel", 0.05),
            ("empty_fruit_bunches", 0.22),
            ("palm_fibre", 0.13),
            ("palm_shell", 0.06),
            ("palm_oil_mill_effluent", 0.32),
        ],
        "notes": (
            "Oil extraction rate (OER) is a key mill performance metric. "
            "Malaysian national average OER is ~20-22%. High-performing "
            "mills with fresh FFB achieve 22-23%."
        ),
    },
    "palm_oil::ffb_to_pko": {
        "commodity": "palm_oil",
        "process_type": "ffb_to_pko",
        "description": "Extracting palm kernel oil from FFB via kernel crushing",
        "input_product": "fresh_fruit_bunches",
        "output_product": "palm_kernel_oil",
        "yield_ratio": {"min": 0.03, "typical": 0.035, "max": 0.04},
        "source": "MPOB Annual Report: Kernel Extraction Rates (2024)",
        "loss_tolerance_pct": 1.0,
        "by_products": [
            ("palm_kernel_cake", 0.025),
        ],
        "notes": (
            "Kernel extraction rate (KER) averages 3.5-4.0%. PKO is "
            "extracted from the palm kernel after CPO milling separates "
            "the mesocarp from the endocarp."
        ),
    },
    "palm_oil::cpo_to_rbd": {
        "commodity": "palm_oil",
        "process_type": "cpo_to_rbd",
        "description": "Refining CPO to RBD (refined, bleached, deodorized) palm oil",
        "input_product": "crude_palm_oil",
        "output_product": "rbd_palm_oil",
        "yield_ratio": {"min": 0.90, "typical": 0.92, "max": 0.94},
        "source": "MPOB Technical Advisory: Refining Parameters (2024)",
        "loss_tolerance_pct": 2.0,
        "by_products": [
            ("palm_fatty_acid_distillate", 0.04),
            ("bleaching_earth_waste", 0.02),
        ],
        "notes": (
            "Refining loss depends on FFA content of CPO. Higher FFA "
            "(>5%) leads to higher refining losses. Physical refining "
            "is standard in Malaysia and Indonesia."
        ),
    },
    "palm_oil::rbd_to_olein": {
        "commodity": "palm_oil",
        "process_type": "rbd_to_olein",
        "description": "Fractionating RBD palm oil to palm olein (liquid fraction)",
        "input_product": "rbd_palm_oil",
        "output_product": "palm_olein",
        "yield_ratio": {"min": 0.62, "typical": 0.65, "max": 0.68},
        "source": "MPOB Technical Advisory: Fractionation (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("palm_stearin", 0.35),
        ],
        "notes": (
            "Dry fractionation separates RBD into olein (liquid, IV 56+) "
            "and stearin (solid). Ratio depends on cooling profile and "
            "target iodine value. Total recovery ~100%."
        ),
    },
    "palm_oil::rbd_to_stearin": {
        "commodity": "palm_oil",
        "process_type": "rbd_to_stearin",
        "description": "Fractionating RBD palm oil to palm stearin (solid fraction)",
        "input_product": "rbd_palm_oil",
        "output_product": "palm_stearin",
        "yield_ratio": {"min": 0.32, "typical": 0.35, "max": 0.38},
        "source": "MPOB Technical Advisory: Fractionation (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("palm_olein", 0.65),
        ],
        "notes": (
            "Stearin is the complement of olein fractionation. "
            "Used in margarine, shortening, and oleochemicals."
        ),
    },

    # ===================================================================
    # COFFEE (Coffea arabica / Coffea canephora)
    # ===================================================================
    "coffee::cherry_to_green": {
        "commodity": "coffee",
        "process_type": "cherry_to_green",
        "description": "Processing coffee cherries to green (unroasted) beans",
        "input_product": "coffee_cherry",
        "output_product": "green_coffee_beans",
        "yield_ratio": {"min": 0.17, "typical": 0.19, "max": 0.20},
        "source": "ICO Statistics: Coffee Processing Yields (2024)",
        "loss_tolerance_pct": 2.0,
        "by_products": [
            ("coffee_pulp", 0.40),
            ("coffee_mucilage", 0.17),
            ("coffee_parchment", 0.08),
            ("coffee_silverskin", 0.02),
        ],
        "notes": (
            "Washed processing yields slightly less green bean than "
            "natural (dry) processing due to mucilage removal. "
            "Approximately 5-6 kg of cherry per 1 kg of green bean."
        ),
    },
    "coffee::green_to_roasted": {
        "commodity": "coffee",
        "process_type": "green_to_roasted",
        "description": "Roasting green coffee beans",
        "input_product": "green_coffee_beans",
        "output_product": "roasted_coffee_beans",
        "yield_ratio": {"min": 0.80, "typical": 0.83, "max": 0.85},
        "source": "ICO Statistics: Roasting Parameters (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("chaff", 0.01),
        ],
        "notes": (
            "Weight loss is primarily moisture evaporation (12-20%) "
            "plus some organic compound volatilization. Darker roasts "
            "lose more weight. Light roast ~15%, medium ~17%, dark ~20%."
        ),
    },
    "coffee::roasted_to_ground": {
        "commodity": "coffee",
        "process_type": "roasted_to_ground",
        "description": "Grinding roasted coffee beans to ground coffee",
        "input_product": "roasted_coffee_beans",
        "output_product": "ground_coffee",
        "yield_ratio": {"min": 0.97, "typical": 0.98, "max": 0.99},
        "source": "ICO Statistics: Coffee Grinding Yields (2024)",
        "loss_tolerance_pct": 1.5,
        "by_products": [],
        "notes": (
            "Grinding is nearly lossless. Minor losses from equipment "
            "retention and dust. Industrial grinders achieve 99%+ yield."
        ),
    },
    "coffee::cherry_to_parchment": {
        "commodity": "coffee",
        "process_type": "cherry_to_parchment",
        "description": "Wet processing coffee cherry to parchment coffee",
        "input_product": "coffee_cherry",
        "output_product": "parchment_coffee",
        "yield_ratio": {"min": 0.22, "typical": 0.25, "max": 0.28},
        "source": "ICO Statistics: Wet Processing Yields (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("coffee_pulp", 0.40),
            ("coffee_mucilage", 0.17),
        ],
        "notes": (
            "Parchment coffee is the intermediate product in washed "
            "processing. Further hulling removes parchment to yield "
            "green beans."
        ),
    },
    "coffee::parchment_to_green": {
        "commodity": "coffee",
        "process_type": "parchment_to_green",
        "description": "Hulling parchment coffee to green beans",
        "input_product": "parchment_coffee",
        "output_product": "green_coffee_beans",
        "yield_ratio": {"min": 0.72, "typical": 0.75, "max": 0.78},
        "source": "ICO Statistics: Hulling Yields (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("coffee_parchment_husk", 0.08),
            ("coffee_silverskin", 0.02),
        ],
        "notes": (
            "Hulling removes the parchment and silverskin layers. "
            "Quality grading and defect removal further reduce yield."
        ),
    },

    # ===================================================================
    # SOYA (Glycine max)
    # ===================================================================
    "soya::beans_to_oil": {
        "commodity": "soya",
        "process_type": "beans_to_oil",
        "description": "Solvent extraction of crude soybean oil from soybeans",
        "input_product": "soybeans",
        "output_product": "crude_soybean_oil",
        "yield_ratio": {"min": 0.18, "typical": 0.19, "max": 0.20},
        "source": "USDA Oil Crops Yearbook: Soybean Crushing (2024)",
        "loss_tolerance_pct": 1.5,
        "by_products": [
            ("soybean_meal", 0.79),
            ("soybean_hulls", 0.03),
        ],
        "notes": (
            "Oil content of soybeans ranges 18-22% depending on variety "
            "and growing conditions. Hexane solvent extraction recovers "
            "~97% of available oil. Meal is the dominant output by mass."
        ),
    },
    "soya::beans_to_meal": {
        "commodity": "soya",
        "process_type": "beans_to_meal",
        "description": "Soybean crushing and desolventizing to produce soybean meal",
        "input_product": "soybeans",
        "output_product": "soybean_meal",
        "yield_ratio": {"min": 0.78, "typical": 0.79, "max": 0.80},
        "source": "USDA Oil Crops Yearbook: Soybean Crushing (2024)",
        "loss_tolerance_pct": 1.5,
        "by_products": [
            ("crude_soybean_oil", 0.19),
            ("soybean_hulls", 0.03),
        ],
        "notes": (
            "High-protein meal (47-48% protein) is the primary economic "
            "driver of soybean crushing. Dehulled meal commands a premium."
        ),
    },
    "soya::beans_to_hulls": {
        "commodity": "soya",
        "process_type": "beans_to_hulls",
        "description": "Soybean dehulling to separate hulls",
        "input_product": "soybeans",
        "output_product": "soybean_hulls",
        "yield_ratio": {"min": 0.02, "typical": 0.03, "max": 0.04},
        "source": "USDA Oil Crops Yearbook: Soybean Processing (2024)",
        "loss_tolerance_pct": 1.0,
        "by_products": [
            ("dehulled_soybeans", 0.97),
        ],
        "notes": (
            "Hull fraction is small (~3%). Hulls used in animal feed "
            "and fibre supplements. Some mills skip dehulling."
        ),
    },
    "soya::oil_to_refined": {
        "commodity": "soya",
        "process_type": "oil_to_refined",
        "description": "Refining crude soybean oil to RBD soybean oil",
        "input_product": "crude_soybean_oil",
        "output_product": "rbd_soybean_oil",
        "yield_ratio": {"min": 0.93, "typical": 0.95, "max": 0.96},
        "source": "USDA Oil Crops Yearbook: Refining Losses (2024)",
        "loss_tolerance_pct": 2.0,
        "by_products": [
            ("soybean_soapstock", 0.03),
            ("soybean_acid_oil", 0.01),
        ],
        "notes": (
            "Chemical (alkali) refining typical in Americas. Physical "
            "refining less common for soy due to phospholipid content."
        ),
    },

    # ===================================================================
    # RUBBER (Hevea brasiliensis)
    # ===================================================================
    "rubber::latex_to_rss": {
        "commodity": "rubber",
        "process_type": "latex_to_rss",
        "description": "Coagulating field latex to ribbed smoked sheet (RSS)",
        "input_product": "field_latex",
        "output_product": "ribbed_smoked_sheet",
        "yield_ratio": {"min": 0.30, "typical": 0.33, "max": 0.35},
        "source": "IRSG Statistical Bulletin: Natural Rubber (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("rubber_serum", 0.55),
        ],
        "notes": (
            "Field latex is ~30-35% dry rubber content (DRC). RSS "
            "processing involves coagulation with formic acid, milling "
            "into sheets, and smoking for preservation."
        ),
    },
    "rubber::latex_to_tsr": {
        "commodity": "rubber",
        "process_type": "latex_to_tsr",
        "description": "Processing field latex to technically specified rubber (TSR)",
        "input_product": "field_latex",
        "output_product": "technically_specified_rubber",
        "yield_ratio": {"min": 0.28, "typical": 0.30, "max": 0.32},
        "source": "IRSG Statistical Bulletin: Natural Rubber (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("rubber_serum", 0.58),
        ],
        "notes": (
            "TSR (e.g., STR 20, SMR 20) is crumb rubber processed "
            "in large factories. Slightly lower yield than RSS due to "
            "additional cleaning and drying steps."
        ),
    },
    "rubber::coagulum_to_tsr": {
        "commodity": "rubber",
        "process_type": "coagulum_to_tsr",
        "description": "Processing cup lump / field coagulum to TSR",
        "input_product": "field_coagulum",
        "output_product": "technically_specified_rubber",
        "yield_ratio": {"min": 0.45, "typical": 0.50, "max": 0.55},
        "source": "IRSG Statistical Bulletin: Cup Lump Processing (2024)",
        "loss_tolerance_pct": 5.0,
        "by_products": [],
        "notes": (
            "Cup lump has higher DRC (~45-55%) than field latex. "
            "Variable quality due to field conditions. TSR from cup "
            "lump is typically lower grade (TSR 20 vs TSR 5)."
        ),
    },
    "rubber::latex_to_concentrate": {
        "commodity": "rubber",
        "process_type": "latex_to_concentrate",
        "description": "Centrifuging field latex to latex concentrate (60% DRC)",
        "input_product": "field_latex",
        "output_product": "latex_concentrate",
        "yield_ratio": {"min": 0.50, "typical": 0.55, "max": 0.60},
        "source": "IRSG Statistical Bulletin: Latex Concentrate (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("skim_latex", 0.35),
        ],
        "notes": (
            "Centrifuging concentrates latex from ~30% DRC to ~60% DRC. "
            "Skim latex by-product processed into skim block rubber."
        ),
    },

    # ===================================================================
    # WOOD / TIMBER
    # ===================================================================
    "wood::log_to_sawn": {
        "commodity": "wood",
        "process_type": "log_to_sawn",
        "description": "Sawmilling round logs into sawn timber / lumber",
        "input_product": "round_logs",
        "output_product": "sawn_timber",
        "yield_ratio": {"min": 0.45, "typical": 0.50, "max": 0.55},
        "source": "FAO Forestry Statistics: Sawnwood Recovery Rates (2024)",
        "loss_tolerance_pct": 5.0,
        "by_products": [
            ("wood_chips", 0.20),
            ("sawdust", 0.15),
            ("bark", 0.08),
            ("wood_slabs", 0.07),
        ],
        "notes": (
            "Recovery rate depends heavily on log diameter, taper, "
            "species, and sawmill technology. Band saws achieve higher "
            "recovery than circular saws. Tropical hardwoods typically "
            "45-50%; temperate softwoods 50-55%."
        ),
    },
    "wood::log_to_veneer": {
        "commodity": "wood",
        "process_type": "log_to_veneer",
        "description": "Peeling or slicing logs into veneer sheets",
        "input_product": "round_logs",
        "output_product": "veneer_sheets",
        "yield_ratio": {"min": 0.50, "typical": 0.55, "max": 0.60},
        "source": "FAO Forestry Statistics: Veneer Recovery Rates (2024)",
        "loss_tolerance_pct": 5.0,
        "by_products": [
            ("peeler_core", 0.10),
            ("veneer_clippings", 0.08),
            ("bark", 0.08),
            ("sawdust", 0.05),
        ],
        "notes": (
            "Rotary peeling yields higher volume recovery than slicing. "
            "Log roundness and centering critical for rotary peeling. "
            "Tropical species commonly peeled for plywood."
        ),
    },
    "wood::log_to_chips": {
        "commodity": "wood",
        "process_type": "log_to_chips",
        "description": "Chipping round logs into wood chips for pulp or energy",
        "input_product": "round_logs",
        "output_product": "wood_chips",
        "yield_ratio": {"min": 0.82, "typical": 0.85, "max": 0.90},
        "source": "FAO Forestry Statistics: Chip Recovery Rates (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("bark", 0.08),
            ("fines", 0.04),
        ],
        "notes": (
            "High conversion efficiency. Debarking losses are the "
            "primary yield reduction. Chips used in pulp & paper, "
            "particleboard, and bioenergy."
        ),
    },
    "wood::log_to_plywood": {
        "commodity": "wood",
        "process_type": "log_to_plywood",
        "description": "Manufacturing plywood from round logs via veneering and gluing",
        "input_product": "round_logs",
        "output_product": "plywood",
        "yield_ratio": {"min": 0.40, "typical": 0.45, "max": 0.50},
        "source": "FAO Forestry Statistics: Plywood Recovery Rates (2024)",
        "loss_tolerance_pct": 5.0,
        "by_products": [
            ("veneer_clippings", 0.10),
            ("peeler_core", 0.10),
            ("bark", 0.08),
            ("sawdust", 0.05),
        ],
        "notes": (
            "Plywood manufacturing combines peeling + drying + gluing + "
            "trimming. Each step introduces losses. Indonesian/Malaysian "
            "tropical plywood mills typically achieve 42-48%."
        ),
    },
    "wood::sawn_to_dried": {
        "commodity": "wood",
        "process_type": "sawn_to_dried",
        "description": "Kiln drying sawn timber to target moisture content",
        "input_product": "sawn_timber",
        "output_product": "kiln_dried_timber",
        "yield_ratio": {"min": 0.88, "typical": 0.92, "max": 0.95},
        "source": "FAO Forestry Statistics: Drying Yields (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [],
        "notes": (
            "Weight loss is moisture removal, not material loss. "
            "Volume shrinkage 3-8% depending on species. Defect-based "
            "losses (checking, warping) reduce usable volume further."
        ),
    },

    # ===================================================================
    # CATTLE (Bos taurus / Bos indicus)
    # ===================================================================
    "cattle::live_to_carcass": {
        "commodity": "cattle",
        "process_type": "live_to_carcass",
        "description": "Slaughter dressing: live animal to hot carcass weight",
        "input_product": "live_cattle",
        "output_product": "beef_carcass",
        "yield_ratio": {"min": 0.52, "typical": 0.55, "max": 0.58},
        "source": "OECD-FAO Agricultural Outlook: Meat Dressing (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("cattle_hide", 0.07),
            ("cattle_offal", 0.15),
            ("blood", 0.04),
            ("head_and_hooves", 0.06),
            ("gut_contents", 0.10),
        ],
        "notes": (
            "Dressing percentage varies by breed, age, condition score, "
            "and gut fill. Dairy breeds typically 48-52%; beef breeds "
            "54-58%. By-products tracked for mass balance completeness."
        ),
    },
    "cattle::live_to_hide": {
        "commodity": "cattle",
        "process_type": "live_to_hide",
        "description": "Hide removal from slaughtered cattle",
        "input_product": "live_cattle",
        "output_product": "cattle_hide",
        "yield_ratio": {"min": 0.06, "typical": 0.07, "max": 0.08},
        "source": "OECD-FAO Agricultural Outlook: By-Products (2024)",
        "loss_tolerance_pct": 1.0,
        "by_products": [
            ("beef_carcass", 0.55),
            ("cattle_offal", 0.15),
        ],
        "notes": (
            "Fresh hide weight 6-8% of live weight. EUDR tracks "
            "cattle hides as a covered derived product. Hide value "
            "significantly lower than beef carcass."
        ),
    },
    "cattle::live_to_offal": {
        "commodity": "cattle",
        "process_type": "live_to_offal",
        "description": "Edible offal recovery from slaughtered cattle",
        "input_product": "live_cattle",
        "output_product": "cattle_offal",
        "yield_ratio": {"min": 0.12, "typical": 0.15, "max": 0.18},
        "source": "OECD-FAO Agricultural Outlook: By-Products (2024)",
        "loss_tolerance_pct": 3.0,
        "by_products": [
            ("beef_carcass", 0.55),
            ("cattle_hide", 0.07),
        ],
        "notes": (
            "Includes liver, heart, kidneys, tongue, tripe, etc. "
            "Offal markets vary by region; higher utilization in "
            "Africa and Asia than in North America / Europe."
        ),
    },
    "cattle::carcass_to_cuts": {
        "commodity": "cattle",
        "process_type": "carcass_to_cuts",
        "description": "Fabricating beef carcass into primal and sub-primal cuts",
        "input_product": "beef_carcass",
        "output_product": "beef_cuts",
        "yield_ratio": {"min": 0.65, "typical": 0.70, "max": 0.75},
        "source": "OECD-FAO Agricultural Outlook: Meat Fabrication (2024)",
        "loss_tolerance_pct": 5.0,
        "by_products": [
            ("beef_trim", 0.12),
            ("beef_bone", 0.15),
            ("beef_fat", 0.08),
        ],
        "notes": (
            "Yield depends on cutting specifications (bone-in vs "
            "boneless) and trim level. Bone-in cuts yield higher total "
            "weight but include non-meat components."
        ),
    },
    "cattle::hide_to_leather": {
        "commodity": "cattle",
        "process_type": "hide_to_leather",
        "description": "Tanning raw cattle hide to finished leather",
        "input_product": "cattle_hide",
        "output_product": "finished_leather",
        "yield_ratio": {"min": 0.20, "typical": 0.25, "max": 0.30},
        "source": "International Council of Tanners: Yield Data (2024)",
        "loss_tolerance_pct": 5.0,
        "by_products": [
            ("leather_shavings", 0.10),
            ("tannery_sludge", 0.30),
        ],
        "notes": (
            "Tanning is highly lossy due to fleshing, splitting, "
            "shaving, and chemical treatment. Wet-blue stage retains "
            "~60% of raw hide weight; finished leather ~20-30%."
        ),
    },
}

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

TOTAL_CONVERSION_FACTORS = len(CONVERSION_FACTORS)

_COMMODITIES_IN_DATA = sorted(set(
    entry["commodity"]
    for entry in CONVERSION_FACTORS.values()
))

TOTAL_COMMODITIES = len(_COMMODITIES_IN_DATA)


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------


def get_conversion_factor(
    commodity: str,
    process_type: str,
) -> Optional[Dict[str, Any]]:
    """Look up a specific conversion factor entry.

    Args:
        commodity: EUDR commodity (cocoa, palm_oil, coffee, soya, rubber,
            wood, cattle).
        process_type: Transformation process type (e.g., beans_to_nibs).

    Returns:
        Conversion factor dictionary, or None if not found.

    Example:
        >>> factor = get_conversion_factor("cocoa", "beans_to_nibs")
        >>> factor["yield_ratio"]["typical"]
        0.87
    """
    key = f"{commodity}::{process_type}"
    return CONVERSION_FACTORS.get(key)


def get_expected_yield(
    commodity: str,
    process_type: str,
    percentile: str = "typical",
) -> Optional[float]:
    """Return the expected yield ratio for a commodity transformation.

    Args:
        commodity: EUDR commodity.
        process_type: Transformation process type.
        percentile: One of 'min', 'typical', or 'max'.

    Returns:
        Yield ratio as a float (0.0-1.0), or None if not found.

    Raises:
        ValueError: If percentile is not 'min', 'typical', or 'max'.

    Example:
        >>> get_expected_yield("palm_oil", "ffb_to_cpo", "typical")
        0.22
    """
    if percentile not in ("min", "typical", "max"):
        raise ValueError(
            f"percentile must be 'min', 'typical', or 'max', got '{percentile}'"
        )
    factor = get_conversion_factor(commodity, process_type)
    if factor is None:
        return None
    return factor["yield_ratio"].get(percentile)


def get_loss_tolerance(
    commodity: str,
    process_type: str,
) -> Optional[float]:
    """Return the acceptable loss tolerance percentage for a process.

    Args:
        commodity: EUDR commodity.
        process_type: Transformation process type.

    Returns:
        Loss tolerance as a percentage (e.g., 3.0 means 3%), or None.

    Example:
        >>> get_loss_tolerance("cocoa", "beans_to_nibs")
        3.0
    """
    factor = get_conversion_factor(commodity, process_type)
    if factor is None:
        return None
    return factor.get("loss_tolerance_pct")


def validate_yield(
    commodity: str,
    process_type: str,
    actual_yield: float,
) -> Dict[str, Any]:
    """Validate an actual yield ratio against reference data.

    Compares the actual yield with expected min/typical/max ranges and
    flags anomalies:
      - 'pass': actual_yield within [min, max] range
      - 'warning': actual_yield within tolerance zone beyond max
      - 'overdraft': actual_yield > max + tolerance (more output than input allows)
      - 'low_yield': actual_yield < min - tolerance (excessive loss)
      - 'unknown': no reference data available

    Args:
        commodity: EUDR commodity.
        process_type: Transformation process type.
        actual_yield: Observed yield ratio (0.0-1.0).

    Returns:
        Dictionary with keys: status, actual_yield, expected_min,
        expected_typical, expected_max, loss_tolerance_pct, deviation_pct.

    Example:
        >>> result = validate_yield("cocoa", "beans_to_nibs", 0.86)
        >>> result["status"]
        'pass'
    """
    factor = get_conversion_factor(commodity, process_type)
    if factor is None:
        return {
            "status": "unknown",
            "actual_yield": actual_yield,
            "message": (
                f"No conversion factor found for "
                f"{commodity}::{process_type}"
            ),
        }

    yr = factor["yield_ratio"]
    tolerance = factor.get("loss_tolerance_pct", 0.0) / 100.0
    expected_min = yr["min"]
    expected_typical = yr["typical"]
    expected_max = yr["max"]

    # Deviation from typical
    deviation_pct = round(
        ((actual_yield - expected_typical) / expected_typical) * 100, 2,
    ) if expected_typical > 0 else 0.0

    if actual_yield > expected_max + tolerance:
        status = "overdraft"
        message = (
            f"Yield {actual_yield:.4f} exceeds maximum {expected_max:.4f} "
            f"plus tolerance {tolerance:.4f}. Possible double-counting."
        )
    elif actual_yield < expected_min - tolerance:
        status = "low_yield"
        message = (
            f"Yield {actual_yield:.4f} below minimum {expected_min:.4f} "
            f"minus tolerance {tolerance:.4f}. Excessive loss detected."
        )
    elif expected_min <= actual_yield <= expected_max:
        status = "pass"
        message = "Yield within expected range."
    elif actual_yield > expected_max:
        status = "warning"
        message = (
            f"Yield {actual_yield:.4f} exceeds max {expected_max:.4f} "
            f"but within tolerance {tolerance:.4f}."
        )
    else:
        status = "warning"
        message = (
            f"Yield {actual_yield:.4f} below min {expected_min:.4f} "
            f"but within tolerance {tolerance:.4f}."
        )

    return {
        "status": status,
        "actual_yield": actual_yield,
        "expected_min": expected_min,
        "expected_typical": expected_typical,
        "expected_max": expected_max,
        "loss_tolerance_pct": factor.get("loss_tolerance_pct", 0.0),
        "deviation_pct": deviation_pct,
        "message": message,
    }


def get_by_products(
    commodity: str,
    process_type: str,
) -> List[Tuple[str, float]]:
    """Return by-products and their typical ratios for a process.

    Args:
        commodity: EUDR commodity.
        process_type: Transformation process type.

    Returns:
        List of (by_product_name, typical_ratio) tuples. Empty list if
        no conversion factor found or no by-products.

    Example:
        >>> get_by_products("cocoa", "beans_to_nibs")
        [('cocoa_shell', 0.13)]
    """
    factor = get_conversion_factor(commodity, process_type)
    if factor is None:
        return []
    return [
        (bp[0], bp[1]) for bp in factor.get("by_products", [])
    ]


def get_all_processes_for_commodity(
    commodity: str,
) -> List[Dict[str, Any]]:
    """Return all conversion factor entries for a given commodity.

    Args:
        commodity: EUDR commodity (e.g., 'cocoa', 'palm_oil').

    Returns:
        List of conversion factor dictionaries for the commodity.
        Empty list if commodity not found.

    Example:
        >>> processes = get_all_processes_for_commodity("cocoa")
        >>> len(processes) >= 4
        True
    """
    results = []
    for key, entry in CONVERSION_FACTORS.items():
        if entry["commodity"] == commodity:
            results.append(entry)
    return results


def get_commodities() -> List[str]:
    """Return all commodity types that have conversion factors defined.

    Returns:
        Sorted list of commodity names.

    Example:
        >>> 'cocoa' in get_commodities()
        True
    """
    return list(_COMMODITIES_IN_DATA)


def get_mass_balance_check(
    commodity: str,
    process_type: str,
    input_quantity_kg: float,
    output_quantity_kg: float,
    by_product_quantities_kg: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Perform a mass balance check on a transformation step.

    Verifies that output + by-products + acceptable loss account for the
    full input quantity.

    Args:
        commodity: EUDR commodity.
        process_type: Transformation process type.
        input_quantity_kg: Input material weight in kilograms.
        output_quantity_kg: Primary output weight in kilograms.
        by_product_quantities_kg: Optional dict mapping by-product names
            to their actual quantities in kilograms.

    Returns:
        Dictionary with: balanced (bool), primary_yield, total_accounted_kg,
        unaccounted_kg, unaccounted_pct, status.

    Example:
        >>> result = get_mass_balance_check(
        ...     "cocoa", "beans_to_nibs",
        ...     input_quantity_kg=1000.0,
        ...     output_quantity_kg=870.0,
        ...     by_product_quantities_kg={"cocoa_shell": 125.0},
        ... )
        >>> result["balanced"]
        True
    """
    factor = get_conversion_factor(commodity, process_type)
    if factor is None:
        return {
            "balanced": False,
            "status": "unknown",
            "message": f"No reference data for {commodity}::{process_type}",
        }

    by_prod_kg = by_product_quantities_kg or {}
    total_by_prod = sum(by_prod_kg.values())
    total_accounted = output_quantity_kg + total_by_prod
    unaccounted_kg = input_quantity_kg - total_accounted
    unaccounted_pct = (
        (unaccounted_kg / input_quantity_kg) * 100
        if input_quantity_kg > 0 else 0.0
    )

    primary_yield = (
        output_quantity_kg / input_quantity_kg
        if input_quantity_kg > 0 else 0.0
    )

    tolerance = factor.get("loss_tolerance_pct", 0.0)
    expected_loss_pct = (1.0 - factor["yield_ratio"]["typical"]) * 100
    max_acceptable_loss_pct = expected_loss_pct + tolerance

    # Balanced if unaccounted loss is within acceptable range
    balanced = 0 <= unaccounted_pct <= max_acceptable_loss_pct

    if unaccounted_pct < 0:
        status = "overdraft"
        message = (
            f"Output exceeds input by {abs(unaccounted_kg):.2f} kg. "
            f"Possible double-counting or measurement error."
        )
    elif unaccounted_pct > max_acceptable_loss_pct:
        status = "excessive_loss"
        message = (
            f"Unaccounted material {unaccounted_pct:.1f}% exceeds "
            f"acceptable {max_acceptable_loss_pct:.1f}%. "
            f"Missing {unaccounted_kg:.2f} kg."
        )
    else:
        status = "balanced"
        message = (
            f"Mass balance OK: {unaccounted_pct:.1f}% unaccounted "
            f"(max {max_acceptable_loss_pct:.1f}%)."
        )

    return {
        "balanced": balanced,
        "status": status,
        "primary_yield": round(primary_yield, 4),
        "input_quantity_kg": input_quantity_kg,
        "output_quantity_kg": output_quantity_kg,
        "by_product_quantities_kg": by_prod_kg,
        "total_accounted_kg": round(total_accounted, 2),
        "unaccounted_kg": round(unaccounted_kg, 2),
        "unaccounted_pct": round(unaccounted_pct, 2),
        "expected_loss_pct": round(expected_loss_pct, 2),
        "max_acceptable_loss_pct": round(max_acceptable_loss_pct, 2),
        "message": message,
    }
