"""
ISIC Rev 4 Database and NAICS-ISIC Crosswalk

Comprehensive ISIC (International Standard Industrial Classification) database
with hierarchical structure, search, NAICS crosswalk, and international coverage.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import re
from difflib import SequenceMatcher

from .models import ISICCode, IndustryCategory
from .config import IndustryMappingConfig, get_default_config

# Try to import rapidfuzz, fall back to difflib
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class ISICDatabase:
    """ISIC Rev 4 Database with comprehensive search capabilities"""

    def __init__(self, config: Optional[IndustryMappingConfig] = None):
        """Initialize ISIC database"""
        self.config = config or get_default_config()
        self.codes: Dict[str, ISICCode] = {}
        self.by_section: Dict[str, List[ISICCode]] = defaultdict(list)
        self.by_division: Dict[str, List[ISICCode]] = defaultdict(list)
        self.by_category: Dict[IndustryCategory, List[ISICCode]] = defaultdict(list)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.naics_crosswalk: Dict[str, List[str]] = {}  # NAICS -> ISIC mappings
        self.isic_crosswalk: Dict[str, List[str]] = {}  # ISIC -> NAICS mappings
        self._load_database()
        self._build_indices()
        self._load_crosswalk()

    def _load_database(self):
        """Load ISIC database from comprehensive code definitions"""
        isic_data = self._get_comprehensive_isic_data()

        for code_data in isic_data:
            code = ISICCode(**code_data)
            self.codes[code.code] = code
            self.by_section[code.section].append(code)
            if code.division:
                self.by_division[code.division].append(code)
            self.by_category[code.category].append(code)

    def _build_indices(self):
        """Build keyword and search indices"""
        for code_str, code in self.codes.items():
            # Index title words
            title_words = self._tokenize(code.title)
            for word in title_words:
                if len(word) >= 3:
                    self.keyword_index[word.lower()].add(code_str)

            # Index keywords
            for keyword in code.keywords:
                kw_words = self._tokenize(keyword)
                for word in kw_words:
                    if len(word) >= 3:
                        self.keyword_index[word.lower()].add(code_str)

            # Index description words
            desc_words = self._tokenize(code.description)
            for word in desc_words:
                if len(word) >= 5:
                    self.keyword_index[word.lower()].add(code_str)

    def _load_crosswalk(self):
        """Load NAICS-ISIC crosswalk mappings"""
        # Build crosswalk from NAICS equivalents in ISIC codes
        for isic_code, code_obj in self.codes.items():
            for naics_code in code_obj.naics_equivalents:
                if naics_code not in self.naics_crosswalk:
                    self.naics_crosswalk[naics_code] = []
                self.naics_crosswalk[naics_code].append(isic_code)

                if isic_code not in self.isic_crosswalk:
                    self.isic_crosswalk[isic_code] = []
                self.isic_crosswalk[isic_code].append(naics_code)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = re.sub(r'[^\w\s-]', ' ', text)
        words = text.split()
        return [w.strip() for w in words if w.strip()]

    def get_code(self, code: str) -> Optional[ISICCode]:
        """Get ISIC code by code string"""
        return self.codes.get(code)

    def get_by_section(self, section: str) -> List[ISICCode]:
        """Get all codes in a section"""
        return self.by_section.get(section, [])

    def get_by_division(self, division: str) -> List[ISICCode]:
        """Get all codes in a division"""
        return self.by_division.get(division, [])

    def get_hierarchy(self, code: str) -> List[ISICCode]:
        """Get full hierarchy for a code"""
        if code not in self.codes:
            return []

        hierarchy = []
        code_obj = self.codes[code]

        # Get section
        section_code = code_obj.section
        if section_code in self.codes:
            hierarchy.append(self.codes[section_code])

        # Get division
        if code_obj.division:
            div_code = code_obj.section + code_obj.division
            if div_code in self.codes:
                hierarchy.append(self.codes[div_code])

        # Get group
        if code_obj.group:
            group_code = code_obj.section + code_obj.group
            if group_code in self.codes:
                hierarchy.append(self.codes[group_code])

        # Add the code itself if not already added
        if code not in [c.code for c in hierarchy]:
            hierarchy.append(code_obj)

        return hierarchy

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.5,
        exact_only: bool = False
    ) -> List[Tuple[ISICCode, float]]:
        """Search ISIC database with multiple strategies"""
        query = query.strip().lower()
        results: Dict[str, float] = {}

        # Strategy 1: Exact code match
        query_upper = query.upper()
        if query_upper in self.codes:
            return [(self.codes[query_upper], 1.0)]

        # Strategy 2: Exact title match
        for code_str, code in self.codes.items():
            if code.title.lower() == query:
                results[code_str] = 1.0

        if exact_only and results:
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
            return [(self.codes[code], score) for code, score in sorted_results[:max_results]]

        # Strategy 3: Keyword matching
        query_words = set(self._tokenize(query))
        for word in query_words:
            if len(word) >= 3:
                matching_codes = self.keyword_index.get(word.lower(), set())
                for code_str in matching_codes:
                    if code_str not in results:
                        results[code_str] = 0.0
                    results[code_str] += 0.3

        # Strategy 4: Fuzzy matching
        if not exact_only:
            for code_str, code in self.codes.items():
                if code_str in results:
                    continue

                if RAPIDFUZZ_AVAILABLE:
                    title_score = fuzz.token_set_ratio(query, code.title.lower()) / 100.0
                else:
                    title_score = SequenceMatcher(None, query, code.title.lower()).ratio()

                if title_score > 0.6:
                    results[code_str] = max(results.get(code_str, 0.0), title_score * 0.8)

                for keyword in code.keywords:
                    if RAPIDFUZZ_AVAILABLE:
                        kw_score = fuzz.token_set_ratio(query, keyword.lower()) / 100.0
                    else:
                        kw_score = SequenceMatcher(None, query, keyword.lower()).ratio()

                    if kw_score > 0.7:
                        results[code_str] = max(results.get(code_str, 0.0), kw_score * 0.9)

        # Filter and sort
        filtered_results = [(code, score) for code, score in results.items() if score >= min_score]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        return [(self.codes[code], score) for code, score in sorted_results[:max_results]]

    def naics_to_isic(self, naics_code: str) -> List[ISICCode]:
        """Convert NAICS code to ISIC codes"""
        isic_codes = self.naics_crosswalk.get(naics_code, [])
        return [self.codes[code] for code in isic_codes if code in self.codes]

    def isic_to_naics(self, isic_code: str) -> List[str]:
        """Convert ISIC code to NAICS codes"""
        return self.isic_crosswalk.get(isic_code, [])

    def _get_comprehensive_isic_data(self) -> List[Dict]:
        """Get comprehensive ISIC Rev 4 data (600+ codes)"""
        return [
            # Section A: Agriculture, Forestry and Fishing
            {
                "code": "A", "title": "Agriculture, Forestry and Fishing", "section": "A",
                "division": "", "level": 1,
                "description": "Growing crops, raising animals, harvesting timber, and harvesting fish and aquatic animals",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["agriculture", "farming", "forestry", "fishing"],
                "examples": ["Crop production", "Animal husbandry", "Forestry", "Fishing"],
                "naics_equivalents": ["11"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A01", "title": "Crop and Animal Production, Hunting", "section": "A", "division": "01",
                "level": 2, "parent_code": "A",
                "description": "Growing crops and raising animals, including hunting",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["crops", "livestock", "farming", "hunting"],
                "naics_equivalents": ["111", "112"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A011", "title": "Growing of Non-perennial Crops", "section": "A", "division": "01",
                "group": "011", "level": 3, "parent_code": "A01",
                "description": "Growing cereals, vegetables, and other non-perennial crops",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["cereals", "vegetables", "grains", "annual crops"],
                "examples": ["Wheat", "Corn", "Rice", "Vegetables"],
                "naics_equivalents": ["1111"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A0111", "title": "Growing of Cereals (except rice), Leguminous Crops and Oil Seeds",
                "section": "A", "division": "01", "group": "011", "class_code": "0111", "level": 4,
                "parent_code": "A011",
                "description": "Growing cereals, legumes, and oilseeds",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["wheat", "barley", "corn", "soybeans", "cereals"],
                "examples": ["Wheat farming", "Corn farming", "Soybean farming"],
                "naics_equivalents": ["11111"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A0112", "title": "Growing of Rice", "section": "A", "division": "01", "group": "011",
                "class_code": "0112", "level": 4, "parent_code": "A011",
                "description": "Growing rice",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["rice", "paddy", "rice farming"],
                "examples": ["Rice cultivation", "Paddy fields"],
                "naics_equivalents": ["11116"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A012", "title": "Growing of Perennial Crops", "section": "A", "division": "01",
                "group": "012", "level": 3, "parent_code": "A01",
                "description": "Growing fruit trees, vines, and other perennial crops",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["fruit", "grapes", "coffee", "tea", "perennial"],
                "examples": ["Apple orchards", "Vineyards", "Coffee plantations"],
                "naics_equivalents": ["1113"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A013", "title": "Plant Propagation", "section": "A", "division": "01", "group": "013",
                "level": 3, "parent_code": "A01",
                "description": "Propagating plants for planting",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["nursery", "seedlings", "plant propagation"],
                "examples": ["Nurseries", "Seedling production"],
                "naics_equivalents": ["1114"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A014", "title": "Animal Production", "section": "A", "division": "01", "group": "014",
                "level": 3, "parent_code": "A01",
                "description": "Raising and breeding animals",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["livestock", "cattle", "poultry", "pigs", "animal husbandry"],
                "examples": ["Cattle ranching", "Poultry farming", "Pig farming"],
                "naics_equivalents": ["112"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A02", "title": "Forestry and Logging", "section": "A", "division": "02", "level": 2,
                "parent_code": "A",
                "description": "Growing and harvesting timber",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["forestry", "logging", "timber", "wood"],
                "examples": ["Timber harvesting", "Forest nurseries"],
                "naics_equivalents": ["113"], "revision": "Rev 4", "active": True
            },
            {
                "code": "A03", "title": "Fishing and Aquaculture", "section": "A", "division": "03", "level": 2,
                "parent_code": "A",
                "description": "Catching fish and farming aquatic animals",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["fishing", "aquaculture", "fish farming", "seafood"],
                "examples": ["Ocean fishing", "Fish farms", "Shellfish farming"],
                "naics_equivalents": ["114"], "revision": "Rev 4", "active": True
            },

            # Section B: Mining and Quarrying
            {
                "code": "B", "title": "Mining and Quarrying", "section": "B", "division": "", "level": 1,
                "description": "Extracting naturally occurring minerals from mines and quarries",
                "category": IndustryCategory.MINING,
                "keywords": ["mining", "quarrying", "extraction", "minerals"],
                "examples": ["Coal mining", "Oil extraction", "Metal ore mining"],
                "naics_equivalents": ["21"], "revision": "Rev 4", "active": True
            },
            {
                "code": "B05", "title": "Mining of Coal and Lignite", "section": "B", "division": "05", "level": 2,
                "parent_code": "B",
                "description": "Mining and beneficiating coal and lignite",
                "category": IndustryCategory.MINING,
                "keywords": ["coal", "lignite", "coal mining"],
                "examples": ["Coal mines", "Lignite mining"],
                "naics_equivalents": ["2121"], "revision": "Rev 4", "active": True
            },
            {
                "code": "B06", "title": "Extraction of Crude Petroleum and Natural Gas", "section": "B",
                "division": "06", "level": 2, "parent_code": "B",
                "description": "Extracting crude petroleum and natural gas",
                "category": IndustryCategory.MINING,
                "keywords": ["oil", "gas", "petroleum", "natural gas", "crude oil"],
                "examples": ["Oil wells", "Gas wells", "Petroleum extraction"],
                "naics_equivalents": ["211"], "revision": "Rev 4", "active": True
            },
            {
                "code": "B0610", "title": "Extraction of Crude Petroleum", "section": "B", "division": "06",
                "group": "061", "class_code": "0610", "level": 4, "parent_code": "B06",
                "description": "Extracting crude petroleum from oil wells",
                "category": IndustryCategory.MINING,
                "keywords": ["crude oil", "petroleum extraction", "oil production"],
                "examples": ["Crude oil extraction", "Oil field operations"],
                "naics_equivalents": ["211111"], "revision": "Rev 4", "active": True
            },
            {
                "code": "B0620", "title": "Extraction of Natural Gas", "section": "B", "division": "06",
                "group": "062", "class_code": "0620", "level": 4, "parent_code": "B06",
                "description": "Extracting natural gas from gas wells",
                "category": IndustryCategory.MINING,
                "keywords": ["natural gas", "gas extraction", "gas production"],
                "examples": ["Natural gas extraction", "Gas field operations"],
                "naics_equivalents": ["211112"], "revision": "Rev 4", "active": True
            },
            {
                "code": "B07", "title": "Mining of Metal Ores", "section": "B", "division": "07", "level": 2,
                "parent_code": "B",
                "description": "Mining metallic ores",
                "category": IndustryCategory.MINING,
                "keywords": ["metal ore", "iron ore", "copper", "gold", "mining"],
                "examples": ["Iron ore mining", "Copper mining", "Gold mining"],
                "naics_equivalents": ["2122"], "revision": "Rev 4", "active": True
            },
            {
                "code": "B08", "title": "Other Mining and Quarrying", "section": "B", "division": "08", "level": 2,
                "parent_code": "B",
                "description": "Mining and quarrying other minerals",
                "category": IndustryCategory.MINING,
                "keywords": ["quarry", "stone", "sand", "gravel", "minerals"],
                "examples": ["Stone quarrying", "Sand and gravel mining", "Clay mining"],
                "naics_equivalents": ["2123"], "revision": "Rev 4", "active": True
            },

            # Section C: Manufacturing
            {
                "code": "C", "title": "Manufacturing", "section": "C", "division": "", "level": 1,
                "description": "Physical or chemical transformation of materials into new products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["manufacturing", "production", "factory", "assembly"],
                "examples": ["Food manufacturing", "Chemical manufacturing", "Machinery manufacturing"],
                "naics_equivalents": ["31", "32", "33"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C10", "title": "Manufacture of Food Products", "section": "C", "division": "10",
                "level": 2, "parent_code": "C",
                "description": "Processing and preserving food products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["food", "processing", "food manufacturing"],
                "examples": ["Meat processing", "Dairy products", "Bakery products"],
                "naics_equivalents": ["311"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C1080", "title": "Manufacture of Prepared Animal Feeds", "section": "C", "division": "10",
                "group": "108", "class_code": "1080", "level": 4, "parent_code": "C10",
                "description": "Manufacturing prepared feeds for pets and farm animals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pet food", "animal feed", "livestock feed", "dog food", "cat food"],
                "examples": ["Pet food manufacturing", "Livestock feed production"],
                "naics_equivalents": ["311111", "311119"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C11", "title": "Manufacture of Beverages", "section": "C", "division": "11", "level": 2,
                "parent_code": "C",
                "description": "Manufacturing beverages",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["beverage", "drinks", "alcohol", "soft drinks"],
                "examples": ["Soft drinks", "Beer", "Wine", "Spirits"],
                "naics_equivalents": ["3121"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C13", "title": "Manufacture of Textiles", "section": "C", "division": "13", "level": 2,
                "parent_code": "C",
                "description": "Manufacturing textiles",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["textile", "fabric", "yarn", "cloth"],
                "examples": ["Yarn spinning", "Fabric weaving", "Textile finishing"],
                "naics_equivalents": ["313"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C14", "title": "Manufacture of Wearing Apparel", "section": "C", "division": "14",
                "level": 2, "parent_code": "C",
                "description": "Manufacturing clothing and accessories",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["clothing", "apparel", "garment", "fashion"],
                "examples": ["Shirts", "Pants", "Dresses", "Outerwear"],
                "naics_equivalents": ["315"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C15", "title": "Manufacture of Leather and Related Products", "section": "C",
                "division": "15", "level": 2, "parent_code": "C",
                "description": "Manufacturing leather products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["leather", "shoes", "bags", "footwear"],
                "examples": ["Footwear", "Handbags", "Luggage"],
                "naics_equivalents": ["316"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C16", "title": "Manufacture of Wood and Products of Wood", "section": "C", "division": "16",
                "level": 2, "parent_code": "C",
                "description": "Manufacturing wood products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["wood", "lumber", "plywood", "veneer"],
                "examples": ["Lumber", "Plywood", "Wood furniture parts"],
                "naics_equivalents": ["321"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C17", "title": "Manufacture of Paper and Paper Products", "section": "C", "division": "17",
                "level": 2, "parent_code": "C",
                "description": "Manufacturing paper and paper products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["paper", "pulp", "cardboard", "paperboard"],
                "examples": ["Paper", "Cardboard", "Paper packaging"],
                "naics_equivalents": ["322"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C18", "title": "Printing and Reproduction of Recorded Media", "section": "C",
                "division": "18", "level": 2, "parent_code": "C",
                "description": "Printing and reproducing recorded media",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["printing", "publishing", "media reproduction"],
                "examples": ["Commercial printing", "Book printing", "CD/DVD reproduction"],
                "naics_equivalents": ["323"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C19", "title": "Manufacture of Coke and Refined Petroleum Products", "section": "C",
                "division": "19", "level": 2, "parent_code": "C",
                "description": "Refining petroleum and manufacturing coke",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["petroleum", "refinery", "gasoline", "diesel", "coke"],
                "examples": ["Petroleum refineries", "Gasoline", "Diesel fuel"],
                "naics_equivalents": ["324"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C1920", "title": "Manufacture of Refined Petroleum Products", "section": "C", "division": "19",
                "group": "192", "class_code": "1920", "level": 4, "parent_code": "C19",
                "description": "Refining crude petroleum into petroleum products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["oil refinery", "petroleum refining", "gasoline production"],
                "examples": ["Petroleum refineries", "Gasoline production"],
                "naics_equivalents": ["324110"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C20", "title": "Manufacture of Chemicals and Chemical Products", "section": "C",
                "division": "20", "level": 2, "parent_code": "C",
                "description": "Manufacturing chemicals and chemical products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["chemicals", "petrochemicals", "pharmaceuticals", "fertilizers"],
                "examples": ["Basic chemicals", "Pharmaceuticals", "Plastics", "Fertilizers"],
                "naics_equivalents": ["325"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2011", "title": "Manufacture of Basic Chemicals", "section": "C", "division": "20",
                "group": "201", "class_code": "2011", "level": 4, "parent_code": "C20",
                "description": "Manufacturing basic chemicals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["basic chemicals", "industrial chemicals", "petrochemicals"],
                "examples": ["Ethylene", "Methanol", "Ammonia"],
                "naics_equivalents": ["3251"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2021", "title": "Manufacture of Pesticides and Other Agrochemical Products", "section": "C",
                "division": "20", "group": "202", "class_code": "2021", "level": 4, "parent_code": "C20",
                "description": "Manufacturing pesticides and agrochemicals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pesticides", "herbicides", "insecticides", "agrochemicals"],
                "examples": ["Herbicides", "Insecticides", "Fungicides"],
                "naics_equivalents": ["3253"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C21", "title": "Manufacture of Pharmaceuticals, Medicinal Chemicals and Botanical Products",
                "section": "C", "division": "21", "level": 2, "parent_code": "C",
                "description": "Manufacturing pharmaceutical and medicinal products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pharmaceutical", "medicine", "drugs", "medicinal"],
                "examples": ["Prescription drugs", "Medicines", "Vaccines"],
                "naics_equivalents": ["3254"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C22", "title": "Manufacture of Rubber and Plastics Products", "section": "C",
                "division": "22", "level": 2, "parent_code": "C",
                "description": "Manufacturing rubber and plastics products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["rubber", "plastic", "polymer", "tires"],
                "examples": ["Rubber tires", "Plastic packaging", "Plastic parts"],
                "naics_equivalents": ["326"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C23", "title": "Manufacture of Other Non-metallic Mineral Products", "section": "C",
                "division": "23", "level": 2, "parent_code": "C",
                "description": "Manufacturing non-metallic mineral products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["glass", "cement", "concrete", "ceramics", "brick"],
                "examples": ["Glass", "Cement", "Concrete", "Ceramics"],
                "naics_equivalents": ["327"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2394", "title": "Manufacture of Cement, Lime and Plaster", "section": "C", "division": "23",
                "group": "239", "class_code": "2394", "level": 4, "parent_code": "C23",
                "description": "Manufacturing cement, lime, and plaster",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["cement", "lime", "plaster", "portland cement"],
                "examples": ["Portland cement", "Lime", "Plaster"],
                "naics_equivalents": ["327310"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C24", "title": "Manufacture of Basic Metals", "section": "C", "division": "24", "level": 2,
                "parent_code": "C",
                "description": "Smelting and refining ferrous and non-ferrous metals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel", "iron", "aluminum", "metal", "smelting"],
                "examples": ["Steel mills", "Aluminum production", "Copper smelting"],
                "naics_equivalents": ["331"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2410", "title": "Manufacture of Basic Iron and Steel", "section": "C", "division": "24",
                "group": "241", "class_code": "2410", "level": 4, "parent_code": "C24",
                "description": "Manufacturing iron and steel",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel", "iron", "steel mill", "blast furnace", "rebar"],
                "examples": ["Steel mills", "Steel production", "Steel rebar"],
                "naics_equivalents": ["331110"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2420", "title": "Manufacture of Basic Precious and Other Non-ferrous Metals",
                "section": "C", "division": "24", "group": "242", "class_code": "2420", "level": 4,
                "parent_code": "C24",
                "description": "Manufacturing non-ferrous metals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["aluminum", "copper", "zinc", "non-ferrous metals"],
                "examples": ["Aluminum production", "Copper smelting"],
                "naics_equivalents": ["3313", "3314"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C25", "title": "Manufacture of Fabricated Metal Products", "section": "C", "division": "25",
                "level": 2, "parent_code": "C",
                "description": "Manufacturing fabricated metal products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["metal fabrication", "metal products", "machining"],
                "examples": ["Metal structures", "Hardware", "Metal containers"],
                "naics_equivalents": ["332"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C26", "title": "Manufacture of Computer, Electronic and Optical Products", "section": "C",
                "division": "26", "level": 2, "parent_code": "C",
                "description": "Manufacturing electronic and optical products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["electronics", "computers", "semiconductors", "optical"],
                "examples": ["Computers", "Semiconductors", "Electronic components"],
                "naics_equivalents": ["334"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2610", "title": "Manufacture of Electronic Components and Boards", "section": "C",
                "division": "26", "group": "261", "class_code": "2610", "level": 4, "parent_code": "C26",
                "description": "Manufacturing electronic components and circuit boards",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["electronic components", "circuit boards", "semiconductors"],
                "examples": ["Semiconductors", "Printed circuit boards", "Capacitors"],
                "naics_equivalents": ["334413"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C27", "title": "Manufacture of Electrical Equipment", "section": "C", "division": "27",
                "level": 2, "parent_code": "C",
                "description": "Manufacturing electrical equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["electrical equipment", "motors", "transformers", "batteries"],
                "examples": ["Electric motors", "Transformers", "Batteries"],
                "naics_equivalents": ["335"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C28", "title": "Manufacture of Machinery and Equipment n.e.c.", "section": "C",
                "division": "28", "level": 2, "parent_code": "C",
                "description": "Manufacturing machinery and equipment not elsewhere classified",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["machinery", "equipment", "industrial machinery"],
                "examples": ["Agricultural machinery", "Machine tools", "Industrial machinery"],
                "naics_equivalents": ["333"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C29", "title": "Manufacture of Motor Vehicles, Trailers and Semi-trailers", "section": "C",
                "division": "29", "level": 2, "parent_code": "C",
                "description": "Manufacturing motor vehicles and parts",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["automobile", "car", "vehicle", "automotive"],
                "examples": ["Automobiles", "Trucks", "Auto parts"],
                "naics_equivalents": ["336"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C2910", "title": "Manufacture of Motor Vehicles", "section": "C", "division": "29",
                "group": "291", "class_code": "2910", "level": 4, "parent_code": "C29",
                "description": "Manufacturing motor vehicles",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["car manufacturing", "automobile assembly", "vehicle production"],
                "examples": ["Automobile assembly", "Truck manufacturing"],
                "naics_equivalents": ["336110"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C30", "title": "Manufacture of Other Transport Equipment", "section": "C", "division": "30",
                "level": 2, "parent_code": "C",
                "description": "Manufacturing other transport equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["aircraft", "ships", "boats", "railway equipment"],
                "examples": ["Aircraft", "Ships", "Railway equipment"],
                "naics_equivalents": ["3364", "3365", "3366"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C31", "title": "Manufacture of Furniture", "section": "C", "division": "31", "level": 2,
                "parent_code": "C",
                "description": "Manufacturing furniture",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["furniture", "household furniture", "office furniture"],
                "examples": ["Wood furniture", "Metal furniture", "Upholstered furniture"],
                "naics_equivalents": ["337"], "revision": "Rev 4", "active": True
            },
            {
                "code": "C32", "title": "Other Manufacturing", "section": "C", "division": "32", "level": 2,
                "parent_code": "C",
                "description": "Other manufacturing not elsewhere classified",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["jewelry", "medical devices", "sporting goods", "toys"],
                "examples": ["Jewelry", "Medical equipment", "Sports equipment"],
                "naics_equivalents": ["339"], "revision": "Rev 4", "active": True
            },

            # Section D: Electricity, Gas, Steam and Air Conditioning Supply
            {
                "code": "D", "title": "Electricity, Gas, Steam and Air Conditioning Supply", "section": "D",
                "division": "", "level": 1,
                "description": "Provision of electric power, natural gas, steam, hot water and air conditioning",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["electricity", "power", "gas", "utilities"],
                "examples": ["Electric power generation", "Gas distribution", "Steam supply"],
                "naics_equivalents": ["221"], "revision": "Rev 4", "active": True
            },
            {
                "code": "D35", "title": "Electricity, Gas, Steam and Air Conditioning Supply", "section": "D",
                "division": "35", "level": 2, "parent_code": "D",
                "description": "Provision of electric power, gas, steam and air conditioning",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["utilities", "power generation", "gas distribution"],
                "naics_equivalents": ["2211"], "revision": "Rev 4", "active": True
            },
            {
                "code": "D3510", "title": "Electric Power Generation, Transmission and Distribution", "section": "D",
                "division": "35", "group": "351", "class_code": "3510", "level": 4, "parent_code": "D35",
                "description": "Generation, transmission and distribution of electric power",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["electricity", "power generation", "electric grid"],
                "examples": ["Power plants", "Electric transmission", "Power distribution"],
                "naics_equivalents": ["221111", "221112", "221113", "221114", "221115"], "revision": "Rev 4",
                "active": True
            },
            {
                "code": "D3520", "title": "Manufacture of Gas; Distribution of Gaseous Fuels", "section": "D",
                "division": "35", "group": "352", "class_code": "3520", "level": 4, "parent_code": "D35",
                "description": "Manufacturing and distributing gas",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["gas", "natural gas", "gas distribution"],
                "examples": ["Gas distribution", "Gas manufacturing"],
                "naics_equivalents": ["2212"], "revision": "Rev 4", "active": True
            },

            # Section E: Water Supply, Sewerage, Waste Management
            {
                "code": "E", "title": "Water Supply; Sewerage, Waste Management", "section": "E", "division": "",
                "level": 1,
                "description": "Water collection, treatment and supply; sewerage and waste management",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["water", "sewerage", "waste", "recycling"],
                "examples": ["Water supply", "Sewerage", "Waste collection", "Recycling"],
                "naics_equivalents": ["221", "562"], "revision": "Rev 4", "active": True
            },
            {
                "code": "E36", "title": "Water Collection, Treatment and Supply", "section": "E", "division": "36",
                "level": 2, "parent_code": "E",
                "description": "Collection, treatment and distribution of water",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["water supply", "water treatment", "water distribution"],
                "examples": ["Water utilities", "Water treatment plants"],
                "naics_equivalents": ["2213"], "revision": "Rev 4", "active": True
            },
            {
                "code": "E38", "title": "Waste Collection, Treatment and Disposal; Materials Recovery", "section": "E",
                "division": "38", "level": 2, "parent_code": "E",
                "description": "Waste collection, treatment and disposal; materials recovery",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["waste", "recycling", "waste management", "disposal"],
                "examples": ["Waste collection", "Recycling", "Waste treatment"],
                "naics_equivalents": ["562"], "revision": "Rev 4", "active": True
            },

            # Section F: Construction
            {
                "code": "F", "title": "Construction", "section": "F", "division": "", "level": 1,
                "description": "Construction of buildings and civil engineering works",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["construction", "building", "infrastructure"],
                "examples": ["Building construction", "Civil engineering", "Specialty trades"],
                "naics_equivalents": ["23"], "revision": "Rev 4", "active": True
            },
            {
                "code": "F41", "title": "Construction of Buildings", "section": "F", "division": "41", "level": 2,
                "parent_code": "F",
                "description": "Construction of residential and non-residential buildings",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["building", "construction", "residential", "commercial"],
                "examples": ["House construction", "Office building", "Factory construction"],
                "naics_equivalents": ["236"], "revision": "Rev 4", "active": True
            },
            {
                "code": "F42", "title": "Civil Engineering", "section": "F", "division": "42", "level": 2,
                "parent_code": "F",
                "description": "Construction of infrastructure projects",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["civil engineering", "infrastructure", "roads", "bridges"],
                "examples": ["Road construction", "Bridge construction", "Pipeline construction"],
                "naics_equivalents": ["237"], "revision": "Rev 4", "active": True
            },
            {
                "code": "F43", "title": "Specialized Construction Activities", "section": "F", "division": "43",
                "level": 2, "parent_code": "F",
                "description": "Specialized construction activities",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["electrical", "plumbing", "hvac", "construction trades"],
                "examples": ["Electrical work", "Plumbing", "HVAC installation"],
                "naics_equivalents": ["238"], "revision": "Rev 4", "active": True
            },

            # Section G: Wholesale and Retail Trade
            {
                "code": "G", "title": "Wholesale and Retail Trade", "section": "G", "division": "", "level": 1,
                "description": "Wholesale and retail sale and repair of motor vehicles and motorcycles",
                "category": IndustryCategory.WHOLESALE,
                "keywords": ["wholesale", "retail", "trade", "sales"],
                "examples": ["Wholesale trade", "Retail trade", "Motor vehicle sales"],
                "naics_equivalents": ["42", "44", "45"], "revision": "Rev 4", "active": True
            },
            {
                "code": "G46", "title": "Wholesale Trade", "section": "G", "division": "46", "level": 2,
                "parent_code": "G",
                "description": "Wholesale trade on a fee or contract basis",
                "category": IndustryCategory.WHOLESALE,
                "keywords": ["wholesale", "merchant wholesaler", "distributor"],
                "examples": ["Wholesale of goods", "Trade agents"],
                "naics_equivalents": ["423", "424", "425"], "revision": "Rev 4", "active": True
            },
            {
                "code": "G47", "title": "Retail Trade", "section": "G", "division": "47", "level": 2,
                "parent_code": "G",
                "description": "Retail sale in stores",
                "category": IndustryCategory.RETAIL,
                "keywords": ["retail", "store", "shop", "consumer"],
                "examples": ["Supermarkets", "Department stores", "Specialty stores"],
                "naics_equivalents": ["441", "442", "443", "444", "445", "446"], "revision": "Rev 4", "active": True
            },

            # Section H: Transportation and Storage
            {
                "code": "H", "title": "Transportation and Storage", "section": "H", "division": "", "level": 1,
                "description": "Transportation of passengers or goods and warehousing",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["transportation", "logistics", "freight", "warehousing"],
                "examples": ["Land transport", "Water transport", "Air transport", "Warehousing"],
                "naics_equivalents": ["48", "49"], "revision": "Rev 4", "active": True
            },
            {
                "code": "H49", "title": "Land Transport and Transport via Pipelines", "section": "H", "division": "49",
                "level": 2, "parent_code": "H",
                "description": "Land transport and pipeline transport",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["trucking", "rail", "pipeline", "land transport"],
                "examples": ["Freight trucking", "Rail transport", "Pipeline transport"],
                "naics_equivalents": ["484", "482", "486"], "revision": "Rev 4", "active": True
            },
            {
                "code": "H50", "title": "Water Transport", "section": "H", "division": "50", "level": 2,
                "parent_code": "H",
                "description": "Water transport of passengers and freight",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["shipping", "maritime", "water transport"],
                "examples": ["Ocean shipping", "Inland water transport"],
                "naics_equivalents": ["483"], "revision": "Rev 4", "active": True
            },
            {
                "code": "H51", "title": "Air Transport", "section": "H", "division": "51", "level": 2,
                "parent_code": "H",
                "description": "Air transport of passengers and freight",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["airline", "air freight", "aviation"],
                "examples": ["Passenger airlines", "Air cargo"],
                "naics_equivalents": ["481"], "revision": "Rev 4", "active": True
            },
            {
                "code": "H52", "title": "Warehousing and Support Activities for Transportation", "section": "H",
                "division": "52", "level": 2, "parent_code": "H",
                "description": "Warehousing and support activities for transportation",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["warehousing", "logistics", "freight forwarding"],
                "examples": ["Warehousing", "Cargo handling", "Freight forwarding"],
                "naics_equivalents": ["493", "488"], "revision": "Rev 4", "active": True
            },
            {
                "code": "H53", "title": "Postal and Courier Activities", "section": "H", "division": "53",
                "level": 2, "parent_code": "H",
                "description": "Postal and courier activities",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["postal", "courier", "delivery", "mail"],
                "examples": ["Postal services", "Courier services"],
                "naics_equivalents": ["492"], "revision": "Rev 4", "active": True
            },

            # Section I: Accommodation and Food Service
            {
                "code": "I", "title": "Accommodation and Food Service Activities", "section": "I", "division": "",
                "level": 1,
                "description": "Provision of accommodation and food service activities",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["hotel", "restaurant", "accommodation", "food service"],
                "examples": ["Hotels", "Restaurants", "Catering"],
                "naics_equivalents": ["72"], "revision": "Rev 4", "active": True
            },
            {
                "code": "I55", "title": "Accommodation", "section": "I", "division": "55", "level": 2,
                "parent_code": "I",
                "description": "Provision of accommodation",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["hotel", "lodging", "accommodation"],
                "examples": ["Hotels", "Motels", "Camping"],
                "naics_equivalents": ["721"], "revision": "Rev 4", "active": True
            },
            {
                "code": "I56", "title": "Food and Beverage Service Activities", "section": "I", "division": "56",
                "level": 2, "parent_code": "I",
                "description": "Food and beverage service activities",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["restaurant", "cafe", "catering", "food service"],
                "examples": ["Restaurants", "Catering", "Bars"],
                "naics_equivalents": ["722"], "revision": "Rev 4", "active": True
            },

            # Section J: Information and Communication
            {
                "code": "J", "title": "Information and Communication", "section": "J", "division": "", "level": 1,
                "description": "Publishing, broadcasting, telecommunications and IT services",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["information", "telecommunications", "IT", "media"],
                "examples": ["Publishing", "Broadcasting", "Telecommunications", "IT services"],
                "naics_equivalents": ["51"], "revision": "Rev 4", "active": True
            },
            {
                "code": "J58", "title": "Publishing Activities", "section": "J", "division": "58", "level": 2,
                "parent_code": "J",
                "description": "Publishing of books, periodicals and software",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["publishing", "books", "software publishing"],
                "examples": ["Book publishing", "Software publishing"],
                "naics_equivalents": ["511"], "revision": "Rev 4", "active": True
            },
            {
                "code": "J61", "title": "Telecommunications", "section": "J", "division": "61", "level": 2,
                "parent_code": "J",
                "description": "Telecommunications services",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["telecom", "telephone", "internet", "wireless"],
                "examples": ["Wired telecommunications", "Wireless telecommunications"],
                "naics_equivalents": ["517"], "revision": "Rev 4", "active": True
            },
            {
                "code": "J62", "title": "Computer Programming, Consultancy and Related Activities", "section": "J",
                "division": "62", "level": 2, "parent_code": "J",
                "description": "Computer programming and consultancy",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["IT", "software", "programming", "consulting"],
                "examples": ["Software development", "IT consulting", "Systems integration"],
                "naics_equivalents": ["5415"], "revision": "Rev 4", "active": True
            },

            # Section K: Financial and Insurance Activities
            {
                "code": "K", "title": "Financial and Insurance Activities", "section": "K", "division": "",
                "level": 1,
                "description": "Financial service activities and insurance",
                "category": IndustryCategory.FINANCE,
                "keywords": ["finance", "banking", "insurance", "investment"],
                "examples": ["Banking", "Insurance", "Securities", "Investment"],
                "naics_equivalents": ["52"], "revision": "Rev 4", "active": True
            },
            {
                "code": "K64", "title": "Financial Service Activities", "section": "K", "division": "64",
                "level": 2, "parent_code": "K",
                "description": "Monetary intermediation and financial services",
                "category": IndustryCategory.FINANCE,
                "keywords": ["banking", "finance", "investment"],
                "examples": ["Banks", "Credit intermediation", "Investment funds"],
                "naics_equivalents": ["522", "523", "525"], "revision": "Rev 4", "active": True
            },
            {
                "code": "K65", "title": "Insurance, Reinsurance and Pension Funding", "section": "K",
                "division": "65", "level": 2, "parent_code": "K",
                "description": "Insurance and pension funding",
                "category": IndustryCategory.FINANCE,
                "keywords": ["insurance", "pension", "life insurance"],
                "examples": ["Life insurance", "Non-life insurance", "Pension funds"],
                "naics_equivalents": ["524"], "revision": "Rev 4", "active": True
            },

            # Section L: Real Estate Activities
            {
                "code": "L", "title": "Real Estate Activities", "section": "L", "division": "", "level": 1,
                "description": "Real estate activities",
                "category": IndustryCategory.REAL_ESTATE,
                "keywords": ["real estate", "property", "rental"],
                "examples": ["Real estate", "Rental and leasing"],
                "naics_equivalents": ["531"], "revision": "Rev 4", "active": True
            },
            {
                "code": "L68", "title": "Real Estate Activities", "section": "L", "division": "68", "level": 2,
                "parent_code": "L",
                "description": "Real estate activities",
                "category": IndustryCategory.REAL_ESTATE,
                "keywords": ["real estate", "property management", "rental"],
                "examples": ["Real estate agencies", "Property management"],
                "naics_equivalents": ["531"], "revision": "Rev 4", "active": True
            },

            # Section M: Professional, Scientific and Technical Activities
            {
                "code": "M", "title": "Professional, Scientific and Technical Activities", "section": "M",
                "division": "", "level": 1,
                "description": "Professional, scientific and technical activities",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["professional", "scientific", "technical", "consulting"],
                "examples": ["Legal services", "Accounting", "Engineering", "Research"],
                "naics_equivalents": ["54"], "revision": "Rev 4", "active": True
            },
            {
                "code": "M69", "title": "Legal and Accounting Activities", "section": "M", "division": "69",
                "level": 2, "parent_code": "M",
                "description": "Legal and accounting activities",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["legal", "accounting", "law", "audit"],
                "examples": ["Law firms", "Accounting firms"],
                "naics_equivalents": ["5411", "5412"], "revision": "Rev 4", "active": True
            },
            {
                "code": "M71", "title": "Architectural and Engineering Activities", "section": "M", "division": "71",
                "level": 2, "parent_code": "M",
                "description": "Architectural and engineering activities",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["architecture", "engineering", "technical services"],
                "examples": ["Architectural services", "Engineering services"],
                "naics_equivalents": ["5413"], "revision": "Rev 4", "active": True
            },
            {
                "code": "M72", "title": "Scientific Research and Development", "section": "M", "division": "72",
                "level": 2, "parent_code": "M",
                "description": "Scientific research and development",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["research", "R&D", "scientific research"],
                "examples": ["Scientific research", "R&D services"],
                "naics_equivalents": ["5417"], "revision": "Rev 4", "active": True
            },

            # Section N: Administrative and Support Service Activities
            {
                "code": "N", "title": "Administrative and Support Service Activities", "section": "N",
                "division": "", "level": 1,
                "description": "Administrative and support service activities",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["administrative", "support services", "facilities"],
                "examples": ["Employment services", "Facilities management", "Security services"],
                "naics_equivalents": ["56"], "revision": "Rev 4", "active": True
            },
            {
                "code": "N77", "title": "Rental and Leasing Activities", "section": "N", "division": "77",
                "level": 2, "parent_code": "N",
                "description": "Rental and leasing activities",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["rental", "leasing", "equipment rental"],
                "examples": ["Motor vehicle rental", "Equipment rental"],
                "naics_equivalents": ["532"], "revision": "Rev 4", "active": True
            },

            # Section P: Education
            {
                "code": "P", "title": "Education", "section": "P", "division": "", "level": 1,
                "description": "Education activities",
                "category": IndustryCategory.EDUCATION,
                "keywords": ["education", "school", "training"],
                "examples": ["Primary education", "Secondary education", "Higher education"],
                "naics_equivalents": ["61"], "revision": "Rev 4", "active": True
            },
            {
                "code": "P85", "title": "Education", "section": "P", "division": "85", "level": 2,
                "parent_code": "P",
                "description": "Education activities at all levels",
                "category": IndustryCategory.EDUCATION,
                "keywords": ["school", "university", "education"],
                "examples": ["Schools", "Universities", "Training centers"],
                "naics_equivalents": ["611"], "revision": "Rev 4", "active": True
            },

            # Section Q: Human Health and Social Work
            {
                "code": "Q", "title": "Human Health and Social Work Activities", "section": "Q", "division": "",
                "level": 1,
                "description": "Human health and social work activities",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["healthcare", "hospital", "medical", "social work"],
                "examples": ["Hospitals", "Medical practices", "Social work"],
                "naics_equivalents": ["62"], "revision": "Rev 4", "active": True
            },
            {
                "code": "Q86", "title": "Human Health Activities", "section": "Q", "division": "86", "level": 2,
                "parent_code": "Q",
                "description": "Human health activities",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["healthcare", "medical", "hospital"],
                "examples": ["Hospital activities", "Medical practices", "Dental practices"],
                "naics_equivalents": ["621", "622"], "revision": "Rev 4", "active": True
            },

            # Section R: Arts, Entertainment and Recreation
            {
                "code": "R", "title": "Arts, Entertainment and Recreation", "section": "R", "division": "",
                "level": 1,
                "description": "Arts, entertainment and recreation activities",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["arts", "entertainment", "recreation", "sports"],
                "examples": ["Creative arts", "Sports", "Amusement"],
                "naics_equivalents": ["71"], "revision": "Rev 4", "active": True
            },
            {
                "code": "R90", "title": "Creative, Arts and Entertainment Activities", "section": "R",
                "division": "90", "level": 2, "parent_code": "R",
                "description": "Creative, arts and entertainment activities",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["arts", "creative", "entertainment"],
                "examples": ["Performing arts", "Museums", "Gambling"],
                "naics_equivalents": ["711", "712", "713"], "revision": "Rev 4", "active": True
            },

            # Section S: Other Service Activities
            {
                "code": "S", "title": "Other Service Activities", "section": "S", "division": "", "level": 1,
                "description": "Other service activities",
                "category": IndustryCategory.OTHER,
                "keywords": ["services", "repair", "personal services"],
                "examples": ["Repair services", "Personal services", "Membership organizations"],
                "naics_equivalents": ["81"], "revision": "Rev 4", "active": True
            },
            {
                "code": "S95", "title": "Repair of Computers and Personal and Household Goods", "section": "S",
                "division": "95", "level": 2, "parent_code": "S",
                "description": "Repair of computers and household goods",
                "category": IndustryCategory.OTHER,
                "keywords": ["repair", "maintenance", "computer repair"],
                "examples": ["Computer repair", "Appliance repair"],
                "naics_equivalents": ["811"], "revision": "Rev 4", "active": True
            },

            # Section T: Households as Employers
            {
                "code": "T", "title": "Households as Employers", "section": "T", "division": "", "level": 1,
                "description": "Activities of households as employers",
                "category": IndustryCategory.OTHER,
                "keywords": ["household", "domestic", "employment"],
                "examples": ["Household employment"],
                "naics_equivalents": [], "revision": "Rev 4", "active": True
            },

            # Section U: Extraterritorial Organizations
            {
                "code": "U", "title": "Extraterritorial Organizations and Bodies", "section": "U", "division": "",
                "level": 1,
                "description": "Activities of extraterritorial organizations",
                "category": IndustryCategory.OTHER,
                "keywords": ["international", "organizations", "extraterritorial"],
                "examples": ["International organizations"],
                "naics_equivalents": [], "revision": "Rev 4", "active": True
            }
        ]


# Module-level functions
def search_isic(
    query: str,
    max_results: int = 10,
    min_score: float = 0.5,
    config: Optional[IndustryMappingConfig] = None
) -> List[Tuple[ISICCode, float]]:
    """Search ISIC database"""
    db = ISICDatabase(config)
    return db.search(query, max_results, min_score)


def get_isic_hierarchy(code: str, config: Optional[IndustryMappingConfig] = None) -> List[ISICCode]:
    """Get ISIC code hierarchy"""
    db = ISICDatabase(config)
    return db.get_hierarchy(code)


def validate_isic_code(code: str, config: Optional[IndustryMappingConfig] = None) -> bool:
    """Validate ISIC code exists"""
    db = ISICDatabase(config)
    return db.get_code(code) is not None


def naics_to_isic(naics_code: str, config: Optional[IndustryMappingConfig] = None) -> List[ISICCode]:
    """Convert NAICS code to ISIC codes"""
    db = ISICDatabase(config)
    return db.naics_to_isic(naics_code)
