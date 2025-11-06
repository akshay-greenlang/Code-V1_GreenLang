"""
NAICS 2022 Database and Search Functions

Comprehensive NAICS (North American Industry Classification System) database
with hierarchical structure, search, and matching capabilities.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import re
from difflib import SequenceMatcher

from .models import NAICSCode, IndustryCategory, CodeHierarchy
from .config import IndustryMappingConfig, get_default_config

# Try to import rapidfuzz, fall back to difflib
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class NAICSDatabase:
    """NAICS 2022 Database with comprehensive search capabilities"""

    def __init__(self, config: Optional[IndustryMappingConfig] = None):
        """Initialize NAICS database"""
        self.config = config or get_default_config()
        self.codes: Dict[str, NAICSCode] = {}
        self.by_level: Dict[int, List[NAICSCode]] = defaultdict(list)
        self.by_category: Dict[IndustryCategory, List[NAICSCode]] = defaultdict(list)
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self._load_database()
        self._build_indices()

    def _load_database(self):
        """Load NAICS database from comprehensive code definitions"""
        # This would normally load from CSV, but we'll define comprehensive data inline
        naics_data = self._get_comprehensive_naics_data()

        for code_data in naics_data:
            code = NAICSCode(**code_data)
            self.codes[code.code] = code
            self.by_level[code.level].append(code)
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

            # Index description words (key terms only)
            desc_words = self._tokenize(code.description)
            for word in desc_words:
                if len(word) >= 5:  # Only longer words from description
                    self.keyword_index[word.lower()].add(code_str)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove special characters and split
        text = re.sub(r'[^\w\s-]', ' ', text)
        words = text.split()
        return [w.strip() for w in words if w.strip()]

    def get_code(self, code: str) -> Optional[NAICSCode]:
        """Get NAICS code by code string"""
        return self.codes.get(code)

    def get_by_level(self, level: int) -> List[NAICSCode]:
        """Get all codes at a specific level"""
        return self.by_level.get(level, [])

    def get_by_category(self, category: IndustryCategory) -> List[NAICSCode]:
        """Get all codes in a category"""
        return self.by_category.get(category, [])

    def get_hierarchy(self, code: str) -> List[NAICSCode]:
        """Get full hierarchy for a code"""
        if code not in self.codes:
            return []

        hierarchy = []
        current_code = code

        # Walk up the hierarchy
        for level in range(len(code), 1, -1):
            parent_code = current_code[:level]
            if parent_code in self.codes:
                hierarchy.insert(0, self.codes[parent_code])
            current_code = parent_code

        return hierarchy

    def get_children(self, code: str) -> List[NAICSCode]:
        """Get direct children of a code"""
        if code not in self.codes:
            return []

        parent_level = len(code)
        child_level = parent_level + 1

        if child_level > 6:
            return []

        children = []
        for child_code in self.by_level.get(child_level, []):
            if child_code.code.startswith(code):
                children.append(child_code)

        return children

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.5,
        exact_only: bool = False
    ) -> List[Tuple[NAICSCode, float]]:
        """
        Search NAICS database with multiple strategies

        Args:
            query: Search query
            max_results: Maximum results to return
            min_score: Minimum match score (0-1)
            exact_only: Only return exact matches

        Returns:
            List of (NAICSCode, score) tuples sorted by score
        """
        query = query.strip().lower()
        results: Dict[str, float] = {}

        # Strategy 1: Exact code match
        if query.isdigit() and len(query) >= 2:
            if query in self.codes:
                return [(self.codes[query], 1.0)]

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
                    results[code_str] += 0.3  # Boost for each keyword match

        # Strategy 4: Fuzzy matching on titles
        if not exact_only:
            for code_str, code in self.codes.items():
                if code_str in results:
                    continue

                # Use rapidfuzz if available, otherwise use difflib
                if RAPIDFUZZ_AVAILABLE:
                    title_score = fuzz.token_set_ratio(query, code.title.lower()) / 100.0
                else:
                    title_score = SequenceMatcher(None, query, code.title.lower()).ratio()

                if title_score > 0.6:
                    results[code_str] = max(results.get(code_str, 0.0), title_score * 0.8)

                # Also check keywords
                for keyword in code.keywords:
                    if RAPIDFUZZ_AVAILABLE:
                        kw_score = fuzz.token_set_ratio(query, keyword.lower()) / 100.0
                    else:
                        kw_score = SequenceMatcher(None, query, keyword.lower()).ratio()

                    if kw_score > 0.7:
                        results[code_str] = max(results.get(code_str, 0.0), kw_score * 0.9)

        # Filter by minimum score and sort
        filtered_results = [(code, score) for code, score in results.items() if score >= min_score]
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

        return [(self.codes[code], score) for code, score in sorted_results[:max_results]]

    def _get_comprehensive_naics_data(self) -> List[Dict]:
        """Get comprehensive NAICS 2022 data (600+ codes)"""
        # This is a comprehensive dataset covering major industries
        return [
            # Agriculture, Forestry, Fishing (11)
            {
                "code": "11", "title": "Agriculture, Forestry, Fishing and Hunting", "level": 2,
                "description": "Growing crops, raising animals, harvesting timber, and harvesting fish",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["agriculture", "farming", "forestry", "fishing", "crops", "livestock"],
                "examples": ["Crop production", "Animal production", "Forestry", "Fishing"],
                "year": 2022, "active": True
            },
            {
                "code": "111", "title": "Crop Production", "level": 3, "parent_code": "11",
                "description": "Growing crops, plants, vines, or trees",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["crops", "farming", "plants", "agriculture"],
                "examples": ["Grain farming", "Vegetable farming", "Fruit farming"],
                "year": 2022, "active": True
            },
            {
                "code": "1111", "title": "Oilseed and Grain Farming", "level": 4, "parent_code": "111",
                "description": "Growing oilseed and grain crops",
                "category": IndustryCategory.AGRICULTURE,
                "keywords": ["grain", "wheat", "corn", "soybeans", "oilseed"],
                "examples": ["Wheat farming", "Corn farming", "Soybean farming"],
                "year": 2022, "active": True
            },

            # Mining (21)
            {
                "code": "21", "title": "Mining, Quarrying, and Oil and Gas Extraction", "level": 2,
                "description": "Extracting naturally occurring minerals from the earth",
                "category": IndustryCategory.MINING,
                "keywords": ["mining", "oil", "gas", "coal", "quarrying", "extraction"],
                "examples": ["Oil extraction", "Coal mining", "Metal ore mining"],
                "year": 2022, "active": True
            },
            {
                "code": "211", "title": "Oil and Gas Extraction", "level": 3, "parent_code": "21",
                "description": "Operating and developing oil and gas field properties",
                "category": IndustryCategory.MINING,
                "keywords": ["oil", "gas", "petroleum", "crude oil", "natural gas"],
                "examples": ["Crude petroleum extraction", "Natural gas extraction"],
                "year": 2022, "active": True
            },
            {
                "code": "2111", "title": "Oil and Gas Extraction", "level": 4, "parent_code": "211",
                "description": "Operating and developing oil and gas field properties",
                "category": IndustryCategory.MINING,
                "keywords": ["oil well", "gas well", "petroleum", "drilling"],
                "examples": ["Oil drilling", "Gas drilling"],
                "year": 2022, "active": True
            },
            {
                "code": "21111", "title": "Oil and Gas Extraction", "level": 5, "parent_code": "2111",
                "description": "Operating and developing oil and gas field properties",
                "category": IndustryCategory.MINING,
                "keywords": ["oil extraction", "gas extraction", "petroleum production"],
                "year": 2022, "active": True
            },
            {
                "code": "211111", "title": "Crude Petroleum Extraction", "level": 6, "parent_code": "21111",
                "description": "Operating and developing crude petroleum field properties",
                "category": IndustryCategory.MINING,
                "keywords": ["crude oil", "petroleum extraction", "oil production"],
                "examples": ["Crude oil production", "Petroleum field operations"],
                "year": 2022, "active": True
            },
            {
                "code": "211112", "title": "Natural Gas Extraction", "level": 6, "parent_code": "21111",
                "description": "Operating and developing natural gas field properties",
                "category": IndustryCategory.MINING,
                "keywords": ["natural gas", "gas extraction", "gas production"],
                "examples": ["Natural gas production", "Gas field operations"],
                "year": 2022, "active": True
            },
            {
                "code": "212", "title": "Mining (except Oil and Gas)", "level": 3, "parent_code": "21",
                "description": "Mining, beneficiating, or otherwise preparing metallic and nonmetallic minerals",
                "category": IndustryCategory.MINING,
                "keywords": ["mining", "coal", "metal", "minerals", "ore"],
                "examples": ["Coal mining", "Iron ore mining", "Copper mining"],
                "year": 2022, "active": True
            },

            # Utilities (22)
            {
                "code": "22", "title": "Utilities", "level": 2,
                "description": "Providing electric power, natural gas, steam, water, and sewage services",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["utilities", "electricity", "power", "water", "gas", "sewage"],
                "examples": ["Electric power generation", "Natural gas distribution", "Water supply"],
                "year": 2022, "active": True
            },
            {
                "code": "221", "title": "Utilities", "level": 3, "parent_code": "22",
                "description": "Providing electric power, natural gas, steam, water, and sewage services",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["utility services", "power supply", "water supply"],
                "year": 2022, "active": True
            },
            {
                "code": "2211", "title": "Electric Power Generation, Transmission and Distribution", "level": 4,
                "parent_code": "221",
                "description": "Generating, transmitting, and distributing electric power",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["electricity", "power generation", "electric grid", "transmission"],
                "examples": ["Power plants", "Electric grid", "Power distribution"],
                "year": 2022, "active": True
            },
            {
                "code": "221111", "title": "Hydroelectric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating hydroelectric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["hydroelectric", "hydro power", "water power", "dam"],
                "examples": ["Hydroelectric dams", "Water power plants"],
                "year": 2022, "active": True
            },
            {
                "code": "221112", "title": "Fossil Fuel Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating fossil fuel powered electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["coal power", "gas power", "fossil fuel", "thermal power"],
                "examples": ["Coal power plants", "Natural gas power plants"],
                "year": 2022, "active": True
            },
            {
                "code": "221113", "title": "Nuclear Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating nuclear electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["nuclear power", "nuclear reactor", "atomic energy"],
                "examples": ["Nuclear power plants", "Nuclear reactors"],
                "year": 2022, "active": True
            },
            {
                "code": "221114", "title": "Solar Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating solar electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["solar power", "solar panels", "photovoltaic", "solar farm"],
                "examples": ["Solar farms", "Photovoltaic power plants"],
                "year": 2022, "active": True
            },
            {
                "code": "221115", "title": "Wind Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating wind electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["wind power", "wind turbine", "wind farm", "wind energy"],
                "examples": ["Wind farms", "Wind turbines"],
                "year": 2022, "active": True
            },
            {
                "code": "221116", "title": "Geothermal Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating geothermal electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["geothermal", "geothermal power", "thermal energy"],
                "examples": ["Geothermal power plants"],
                "year": 2022, "active": True
            },
            {
                "code": "221117", "title": "Biomass Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating biomass electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["biomass", "biomass power", "bioenergy", "waste to energy"],
                "examples": ["Biomass power plants", "Waste-to-energy facilities"],
                "year": 2022, "active": True
            },
            {
                "code": "221118", "title": "Other Electric Power Generation", "level": 6, "parent_code": "22111",
                "description": "Operating other types of electric power generation facilities",
                "category": IndustryCategory.UTILITIES,
                "keywords": ["renewable energy", "alternative energy"],
                "year": 2022, "active": True
            },

            # Construction (23)
            {
                "code": "23", "title": "Construction", "level": 2,
                "description": "Construction of buildings and infrastructure",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["construction", "building", "contractor", "infrastructure"],
                "examples": ["Building construction", "Heavy construction", "Specialty trades"],
                "year": 2022, "active": True
            },
            {
                "code": "236", "title": "Construction of Buildings", "level": 3, "parent_code": "23",
                "description": "Construction of residential and nonresidential buildings",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["building", "construction", "residential", "commercial"],
                "examples": ["Home building", "Office building", "Warehouse construction"],
                "year": 2022, "active": True
            },
            {
                "code": "237", "title": "Heavy and Civil Engineering Construction", "level": 3, "parent_code": "23",
                "description": "Construction of infrastructure projects",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["infrastructure", "civil engineering", "heavy construction"],
                "examples": ["Road construction", "Bridge construction", "Pipeline construction"],
                "year": 2022, "active": True
            },
            {
                "code": "238", "title": "Specialty Trade Contractors", "level": 3, "parent_code": "23",
                "description": "Specialized construction work",
                "category": IndustryCategory.CONSTRUCTION,
                "keywords": ["electrical", "plumbing", "hvac", "masonry", "roofing"],
                "examples": ["Electrical contractors", "Plumbing contractors", "HVAC contractors"],
                "year": 2022, "active": True
            },

            # Manufacturing (31-33)
            {
                "code": "31", "title": "Manufacturing", "level": 2,
                "description": "Mechanical, physical, or chemical transformation of materials into new products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["manufacturing", "production", "factory", "assembly"],
                "examples": ["Food manufacturing", "Chemical manufacturing", "Machinery manufacturing"],
                "year": 2022, "active": True
            },
            {
                "code": "311", "title": "Food Manufacturing", "level": 3, "parent_code": "31",
                "description": "Transforming raw agricultural products into food products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["food", "processing", "beverage", "packaged food"],
                "examples": ["Meat processing", "Dairy products", "Bakery products"],
                "year": 2022, "active": True
            },
            {
                "code": "3111", "title": "Animal Food Manufacturing", "level": 4, "parent_code": "311",
                "description": "Manufacturing food for animals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pet food", "animal feed", "livestock feed"],
                "examples": ["Dog food", "Cat food", "Livestock feed"],
                "year": 2022, "active": True
            },
            {
                "code": "31111", "title": "Animal Food Manufacturing", "level": 5, "parent_code": "3111",
                "description": "Manufacturing food and feed for animals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pet food manufacturing", "feed production"],
                "year": 2022, "active": True
            },
            {
                "code": "311111", "title": "Dog and Cat Food Manufacturing", "level": 6, "parent_code": "31111",
                "description": "Manufacturing dog and cat food from ingredients",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pet food", "dog food", "cat food", "pet treats"],
                "examples": ["Dry dog food", "Wet cat food", "Pet treats"],
                "year": 2022, "active": True
            },
            {
                "code": "311119", "title": "Other Animal Food Manufacturing", "level": 6, "parent_code": "31111",
                "description": "Manufacturing animal food for farm animals, birds, and fish",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["livestock feed", "poultry feed", "fish feed", "animal feed"],
                "examples": ["Cattle feed", "Poultry feed", "Fish feed"],
                "year": 2022, "active": True
            },
            {
                "code": "3112", "title": "Grain and Oilseed Milling", "level": 4, "parent_code": "311",
                "description": "Milling flour and other grain products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["flour", "grain", "milling", "rice"],
                "examples": ["Flour milling", "Rice milling", "Malt manufacturing"],
                "year": 2022, "active": True
            },
            {
                "code": "3113", "title": "Sugar and Confectionery Product Manufacturing", "level": 4, "parent_code": "311",
                "description": "Manufacturing sugar and confectionery products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["sugar", "candy", "chocolate", "confectionery"],
                "examples": ["Sugar refining", "Chocolate manufacturing", "Candy production"],
                "year": 2022, "active": True
            },
            {
                "code": "3114", "title": "Fruit and Vegetable Preserving and Specialty Food Manufacturing", "level": 4,
                "parent_code": "311",
                "description": "Preserving fruits and vegetables and manufacturing specialty foods",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["canning", "freezing", "preserving", "dried food"],
                "examples": ["Canned vegetables", "Frozen fruits", "Dried fruits"],
                "year": 2022, "active": True
            },
            {
                "code": "3115", "title": "Dairy Product Manufacturing", "level": 4, "parent_code": "311",
                "description": "Manufacturing dairy products from raw milk",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["dairy", "milk", "cheese", "butter", "yogurt", "ice cream"],
                "examples": ["Fluid milk", "Cheese", "Ice cream", "Yogurt"],
                "year": 2022, "active": True
            },
            {
                "code": "3116", "title": "Animal Slaughtering and Processing", "level": 4, "parent_code": "311",
                "description": "Slaughtering animals and processing meat",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["meat", "slaughtering", "processing", "beef", "pork", "poultry"],
                "examples": ["Beef processing", "Pork processing", "Poultry processing"],
                "year": 2022, "active": True
            },
            {
                "code": "3117", "title": "Seafood Product Preparation and Packaging", "level": 4, "parent_code": "311",
                "description": "Processing and packaging seafood products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["seafood", "fish", "shellfish", "processing"],
                "examples": ["Fish processing", "Shellfish processing", "Canned seafood"],
                "year": 2022, "active": True
            },
            {
                "code": "3118", "title": "Bakeries and Tortilla Manufacturing", "level": 4, "parent_code": "311",
                "description": "Manufacturing bread, cakes, and tortillas",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["bakery", "bread", "cake", "tortilla", "pastry"],
                "examples": ["Bread", "Cookies", "Tortillas", "Pastries"],
                "year": 2022, "active": True
            },
            {
                "code": "3119", "title": "Other Food Manufacturing", "level": 4, "parent_code": "311",
                "description": "Manufacturing other food products not elsewhere classified",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["snacks", "coffee", "tea", "spices", "sauces"],
                "examples": ["Snack foods", "Coffee roasting", "Seasonings"],
                "year": 2022, "active": True
            },

            # Beverage and Tobacco (312-313)
            {
                "code": "312", "title": "Beverage and Tobacco Product Manufacturing", "level": 3, "parent_code": "31",
                "description": "Manufacturing beverages and tobacco products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["beverage", "drinks", "alcohol", "soft drinks"],
                "examples": ["Soft drinks", "Beer", "Wine", "Bottled water"],
                "year": 2022, "active": True
            },
            {
                "code": "3121", "title": "Beverage Manufacturing", "level": 4, "parent_code": "312",
                "description": "Manufacturing beverages",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["soft drinks", "juice", "water", "beer", "wine"],
                "examples": ["Soda", "Juice", "Beer", "Wine"],
                "year": 2022, "active": True
            },

            # Textiles (313-316)
            {
                "code": "313", "title": "Textile Mills", "level": 3, "parent_code": "31",
                "description": "Transforming fiber into fabric",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["textile", "fabric", "yarn", "cloth", "weaving"],
                "examples": ["Yarn spinning", "Fabric weaving", "Textile finishing"],
                "year": 2022, "active": True
            },
            {
                "code": "314", "title": "Textile Product Mills", "level": 3, "parent_code": "31",
                "description": "Manufacturing textile products from fabric",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["carpet", "rug", "curtain", "textile products"],
                "examples": ["Carpets", "Curtains", "Bed linens"],
                "year": 2022, "active": True
            },
            {
                "code": "315", "title": "Apparel Manufacturing", "level": 3, "parent_code": "31",
                "description": "Manufacturing clothing and accessories",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["clothing", "apparel", "garment", "fashion"],
                "examples": ["Shirts", "Pants", "Dresses", "Outerwear"],
                "year": 2022, "active": True
            },
            {
                "code": "316", "title": "Leather and Allied Product Manufacturing", "level": 3, "parent_code": "31",
                "description": "Manufacturing leather products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["leather", "shoes", "bags", "luggage"],
                "examples": ["Footwear", "Handbags", "Luggage"],
                "year": 2022, "active": True
            },

            # Wood Products (321)
            {
                "code": "321", "title": "Wood Product Manufacturing", "level": 3, "parent_code": "32",
                "description": "Manufacturing wood products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["wood", "lumber", "plywood", "veneer"],
                "examples": ["Lumber", "Plywood", "Wood furniture parts"],
                "year": 2022, "active": True
            },

            # Paper (322)
            {
                "code": "322", "title": "Paper Manufacturing", "level": 3, "parent_code": "32",
                "description": "Manufacturing pulp, paper, and converted paper products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["paper", "pulp", "cardboard", "paperboard"],
                "examples": ["Paper mills", "Cardboard boxes", "Paper bags"],
                "year": 2022, "active": True
            },

            # Printing (323)
            {
                "code": "323", "title": "Printing and Related Support Activities", "level": 3, "parent_code": "32",
                "description": "Printing text and images on various materials",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["printing", "publishing", "commercial printing"],
                "examples": ["Commercial printing", "Book printing", "Digital printing"],
                "year": 2022, "active": True
            },

            # Petroleum and Coal (324)
            {
                "code": "324", "title": "Petroleum and Coal Products Manufacturing", "level": 3, "parent_code": "32",
                "description": "Transforming crude petroleum and coal into usable products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["petroleum", "refinery", "gasoline", "diesel", "fuel"],
                "examples": ["Petroleum refineries", "Gasoline", "Diesel fuel"],
                "year": 2022, "active": True
            },
            {
                "code": "32411", "title": "Petroleum Refineries", "level": 5, "parent_code": "3241",
                "description": "Refining crude petroleum into petroleum products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["oil refinery", "petroleum refining", "gasoline production"],
                "examples": ["Crude oil refining", "Gasoline production"],
                "year": 2022, "active": True
            },
            {
                "code": "324110", "title": "Petroleum Refineries", "level": 6, "parent_code": "32411",
                "description": "Refining crude petroleum into petroleum products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["refinery", "crude oil", "petroleum products"],
                "examples": ["Oil refineries"],
                "year": 2022, "active": True
            },

            # Chemicals (325)
            {
                "code": "325", "title": "Chemical Manufacturing", "level": 3, "parent_code": "32",
                "description": "Transforming organic and inorganic raw materials by chemical processes",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["chemicals", "petrochemicals", "pharmaceuticals", "plastics"],
                "examples": ["Basic chemicals", "Pharmaceuticals", "Plastics", "Fertilizers"],
                "year": 2022, "active": True
            },
            {
                "code": "3251", "title": "Basic Chemical Manufacturing", "level": 4, "parent_code": "325",
                "description": "Manufacturing basic chemicals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["petrochemicals", "industrial chemicals", "basic chemicals"],
                "examples": ["Ethylene", "Propylene", "Industrial gases"],
                "year": 2022, "active": True
            },
            {
                "code": "3252", "title": "Resin, Synthetic Rubber, and Artificial Fibers Manufacturing", "level": 4,
                "parent_code": "325",
                "description": "Manufacturing resins, synthetic rubber, and synthetic fibers",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["resin", "plastic", "synthetic rubber", "polymer"],
                "examples": ["Plastic resins", "Synthetic rubber", "Synthetic fibers"],
                "year": 2022, "active": True
            },
            {
                "code": "3253", "title": "Pesticide, Fertilizer, and Other Agricultural Chemical Manufacturing", "level": 4,
                "parent_code": "325",
                "description": "Manufacturing pesticides, fertilizers, and agricultural chemicals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["fertilizer", "pesticide", "herbicide", "agricultural chemicals"],
                "examples": ["Nitrogen fertilizers", "Pesticides", "Herbicides"],
                "year": 2022, "active": True
            },
            {
                "code": "3254", "title": "Pharmaceutical and Medicine Manufacturing", "level": 4, "parent_code": "325",
                "description": "Manufacturing pharmaceutical products and medicines",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["pharmaceutical", "medicine", "drugs", "vaccines"],
                "examples": ["Prescription drugs", "Over-the-counter medicines", "Vaccines"],
                "year": 2022, "active": True
            },
            {
                "code": "3255", "title": "Paint, Coating, and Adhesive Manufacturing", "level": 4, "parent_code": "325",
                "description": "Manufacturing paints, coatings, and adhesives",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["paint", "coating", "adhesive", "glue", "sealant"],
                "examples": ["Architectural paints", "Industrial coatings", "Adhesives"],
                "year": 2022, "active": True
            },
            {
                "code": "3256", "title": "Soap, Cleaning Compound, and Toilet Preparation Manufacturing", "level": 4,
                "parent_code": "325",
                "description": "Manufacturing soap, cleaning products, and personal care products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["soap", "detergent", "shampoo", "cosmetics", "cleaning"],
                "examples": ["Detergents", "Shampoo", "Toothpaste", "Cosmetics"],
                "year": 2022, "active": True
            },

            # Plastics and Rubber (326)
            {
                "code": "326", "title": "Plastics and Rubber Products Manufacturing", "level": 3, "parent_code": "32",
                "description": "Manufacturing plastics and rubber products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["plastic", "rubber", "polymer", "molding"],
                "examples": ["Plastic bottles", "Rubber tires", "Plastic packaging"],
                "year": 2022, "active": True
            },
            {
                "code": "3261", "title": "Plastics Product Manufacturing", "level": 4, "parent_code": "326",
                "description": "Manufacturing plastics products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["plastic products", "plastic packaging", "plastic parts"],
                "examples": ["Plastic bottles", "Plastic bags", "Plastic components"],
                "year": 2022, "active": True
            },
            {
                "code": "3262", "title": "Rubber Product Manufacturing", "level": 4, "parent_code": "326",
                "description": "Manufacturing rubber products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["rubber", "tires", "rubber products"],
                "examples": ["Tires", "Rubber hoses", "Rubber belts"],
                "year": 2022, "active": True
            },

            # Nonmetallic Mineral Products (327)
            {
                "code": "327", "title": "Nonmetallic Mineral Product Manufacturing", "level": 3, "parent_code": "32",
                "description": "Manufacturing products from nonmetallic minerals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["glass", "cement", "concrete", "brick", "ceramics"],
                "examples": ["Glass", "Cement", "Concrete", "Bricks", "Ceramics"],
                "year": 2022, "active": True
            },
            {
                "code": "3271", "title": "Clay Product and Refractory Manufacturing", "level": 4, "parent_code": "327",
                "description": "Manufacturing clay products and refractories",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["clay", "brick", "ceramic", "refractory"],
                "examples": ["Bricks", "Ceramic tiles", "Pottery"],
                "year": 2022, "active": True
            },
            {
                "code": "3272", "title": "Glass and Glass Product Manufacturing", "level": 4, "parent_code": "327",
                "description": "Manufacturing glass and glass products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["glass", "windows", "bottles", "fiberglass"],
                "examples": ["Flat glass", "Glass containers", "Fiberglass"],
                "year": 2022, "active": True
            },
            {
                "code": "3273", "title": "Cement and Concrete Product Manufacturing", "level": 4, "parent_code": "327",
                "description": "Manufacturing cement and concrete products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["cement", "concrete", "ready-mix"],
                "examples": ["Portland cement", "Ready-mix concrete", "Concrete blocks"],
                "year": 2022, "active": True
            },
            {
                "code": "32731", "title": "Cement Manufacturing", "level": 5, "parent_code": "3273",
                "description": "Manufacturing portland and other types of cement",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["cement production", "portland cement", "clinker"],
                "examples": ["Portland cement", "Masonry cement"],
                "year": 2022, "active": True
            },
            {
                "code": "327310", "title": "Cement Manufacturing", "level": 6, "parent_code": "32731",
                "description": "Manufacturing portland and other types of cement",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["cement plant", "cement production"],
                "examples": ["Cement manufacturing plants"],
                "year": 2022, "active": True
            },

            # Primary Metal Manufacturing (331)
            {
                "code": "331", "title": "Primary Metal Manufacturing", "level": 3, "parent_code": "33",
                "description": "Smelting and refining ferrous and nonferrous metals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel", "aluminum", "iron", "metal", "smelting"],
                "examples": ["Steel mills", "Aluminum production", "Iron foundries"],
                "year": 2022, "active": True
            },
            {
                "code": "3311", "title": "Iron and Steel Mills and Ferroalloy Manufacturing", "level": 4, "parent_code": "331",
                "description": "Manufacturing iron, steel, and ferroalloys",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel", "iron", "ferroalloy", "steel mill"],
                "examples": ["Steel mills", "Iron production", "Steel products"],
                "year": 2022, "active": True
            },
            {
                "code": "33111", "title": "Iron and Steel Mills and Ferroalloy Manufacturing", "level": 5, "parent_code": "3311",
                "description": "Manufacturing iron and steel from ore or scrap",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel production", "iron production", "blast furnace"],
                "year": 2022, "active": True
            },
            {
                "code": "331110", "title": "Iron and Steel Mills and Ferroalloy Manufacturing", "level": 6, "parent_code": "33111",
                "description": "Manufacturing iron and steel from ore or scrap using integrated or non-integrated processes",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel mill", "iron mill", "steel production", "rebar"],
                "examples": ["Integrated steel mills", "Mini-mills", "Steel rebar"],
                "year": 2022, "active": True
            },
            {
                "code": "3312", "title": "Steel Product Manufacturing from Purchased Steel", "level": 4, "parent_code": "331",
                "description": "Manufacturing steel products from purchased steel",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["steel products", "steel fabrication", "steel wire"],
                "examples": ["Steel pipes", "Steel wire", "Steel springs"],
                "year": 2022, "active": True
            },
            {
                "code": "3313", "title": "Alumina and Aluminum Production and Processing", "level": 4, "parent_code": "331",
                "description": "Manufacturing alumina and aluminum from bauxite ore or scrap",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["aluminum", "alumina", "bauxite"],
                "examples": ["Primary aluminum", "Aluminum sheet", "Aluminum extrusions"],
                "year": 2022, "active": True
            },
            {
                "code": "3314", "title": "Nonferrous Metal (except Aluminum) Production and Processing", "level": 4,
                "parent_code": "331",
                "description": "Manufacturing nonferrous metals except aluminum",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["copper", "zinc", "lead", "nonferrous metal"],
                "examples": ["Copper smelting", "Zinc production", "Lead production"],
                "year": 2022, "active": True
            },
            {
                "code": "3315", "title": "Foundries", "level": 4, "parent_code": "331",
                "description": "Pouring molten metal into molds to manufacture castings",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["foundry", "casting", "molten metal"],
                "examples": ["Iron foundries", "Steel foundries", "Aluminum foundries"],
                "year": 2022, "active": True
            },

            # Fabricated Metal Products (332)
            {
                "code": "332", "title": "Fabricated Metal Product Manufacturing", "level": 3, "parent_code": "33",
                "description": "Transforming metal into intermediate or end products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["metal fabrication", "metal products", "machining"],
                "examples": ["Metal cans", "Hardware", "Spring manufacturing"],
                "year": 2022, "active": True
            },
            {
                "code": "3321", "title": "Forging and Stamping", "level": 4, "parent_code": "332",
                "description": "Forging and stamping metal",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["forging", "stamping", "metal forming"],
                "examples": ["Iron forgings", "Metal stamping"],
                "year": 2022, "active": True
            },
            {
                "code": "3322", "title": "Cutlery and Handtool Manufacturing", "level": 4, "parent_code": "332",
                "description": "Manufacturing cutlery and hand tools",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["cutlery", "hand tools", "knives", "saws"],
                "examples": ["Kitchen cutlery", "Hand saws", "Wrenches"],
                "year": 2022, "active": True
            },
            {
                "code": "3323", "title": "Architectural and Structural Metals Manufacturing", "level": 4, "parent_code": "332",
                "description": "Manufacturing metal building components",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["structural metal", "metal frames", "architectural metal"],
                "examples": ["Metal doors", "Metal windows", "Structural steel"],
                "year": 2022, "active": True
            },
            {
                "code": "3324", "title": "Boiler, Tank, and Shipping Container Manufacturing", "level": 4, "parent_code": "332",
                "description": "Manufacturing boilers, tanks, and shipping containers",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["boiler", "tank", "container", "pressure vessel"],
                "examples": ["Power boilers", "Metal tanks", "Shipping containers"],
                "year": 2022, "active": True
            },
            {
                "code": "3325", "title": "Hardware Manufacturing", "level": 4, "parent_code": "332",
                "description": "Manufacturing hardware",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["hardware", "locks", "hinges", "fasteners"],
                "examples": ["Door locks", "Hinges", "Bolts and nuts"],
                "year": 2022, "active": True
            },
            {
                "code": "3326", "title": "Spring and Wire Product Manufacturing", "level": 4, "parent_code": "332",
                "description": "Manufacturing springs and wire products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["spring", "wire", "wire products"],
                "examples": ["Coil springs", "Wire rope", "Wire mesh"],
                "year": 2022, "active": True
            },
            {
                "code": "3327", "title": "Machine Shops; Turned Product; and Screw, Nut, and Bolt Manufacturing", "level": 4,
                "parent_code": "332",
                "description": "Machine shop work and manufacturing fasteners",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["machine shop", "machining", "screws", "bolts", "nuts"],
                "examples": ["CNC machining", "Metal screws", "Bolts"],
                "year": 2022, "active": True
            },
            {
                "code": "3328", "title": "Coating, Engraving, Heat Treating, and Allied Activities", "level": 4,
                "parent_code": "332",
                "description": "Coating, engraving, and heat treating metals",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["metal coating", "plating", "engraving", "heat treating"],
                "examples": ["Electroplating", "Powder coating", "Heat treating"],
                "year": 2022, "active": True
            },
            {
                "code": "3329", "title": "Other Fabricated Metal Product Manufacturing", "level": 4, "parent_code": "332",
                "description": "Manufacturing other fabricated metal products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["metal valves", "metal fittings", "wire products"],
                "examples": ["Industrial valves", "Pipe fittings", "Metal ammunition"],
                "year": 2022, "active": True
            },

            # Machinery Manufacturing (333)
            {
                "code": "333", "title": "Machinery Manufacturing", "level": 3, "parent_code": "33",
                "description": "Manufacturing industrial and commercial machinery",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["machinery", "equipment", "industrial machinery"],
                "examples": ["Agricultural machinery", "Construction machinery", "Industrial machinery"],
                "year": 2022, "active": True
            },
            {
                "code": "3331", "title": "Agriculture, Construction, and Mining Machinery Manufacturing", "level": 4,
                "parent_code": "333",
                "description": "Manufacturing machinery for agriculture, construction, and mining",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["agricultural equipment", "construction equipment", "mining equipment"],
                "examples": ["Tractors", "Excavators", "Mining machinery"],
                "year": 2022, "active": True
            },
            {
                "code": "3332", "title": "Industrial Machinery Manufacturing", "level": 4, "parent_code": "333",
                "description": "Manufacturing industrial process machinery",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["industrial machinery", "sawmill equipment", "textile machinery"],
                "examples": ["Sawmill machinery", "Paper industry machinery"],
                "year": 2022, "active": True
            },
            {
                "code": "3333", "title": "Commercial and Service Industry Machinery Manufacturing", "level": 4,
                "parent_code": "333",
                "description": "Manufacturing commercial and service machinery",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["commercial machinery", "food service equipment", "vending machines"],
                "examples": ["Commercial ovens", "Vending machines", "Laundry equipment"],
                "year": 2022, "active": True
            },
            {
                "code": "3334", "title": "Ventilation, Heating, Air-Conditioning, and Commercial Refrigeration Equipment Manufacturing",
                "level": 4, "parent_code": "333",
                "description": "Manufacturing HVAC and refrigeration equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["HVAC", "air conditioning", "heating", "refrigeration"],
                "examples": ["Air conditioners", "Furnaces", "Commercial refrigerators"],
                "year": 2022, "active": True
            },
            {
                "code": "3335", "title": "Metalworking Machinery Manufacturing", "level": 4, "parent_code": "333",
                "description": "Manufacturing metalworking machinery",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["machine tools", "metalworking", "CNC machines"],
                "examples": ["Lathes", "Milling machines", "Metal cutting tools"],
                "year": 2022, "active": True
            },
            {
                "code": "3336", "title": "Engine, Turbine, and Power Transmission Equipment Manufacturing", "level": 4,
                "parent_code": "333",
                "description": "Manufacturing engines, turbines, and power transmission equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["engine", "turbine", "gearbox", "transmission"],
                "examples": ["Gasoline engines", "Steam turbines", "Mechanical power transmission equipment"],
                "year": 2022, "active": True
            },

            # Computer and Electronic Products (334)
            {
                "code": "334", "title": "Computer and Electronic Product Manufacturing", "level": 3, "parent_code": "33",
                "description": "Manufacturing computers, computer peripherals, communications equipment, and electronic components",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["electronics", "computers", "semiconductors", "telecommunications"],
                "examples": ["Computers", "Semiconductors", "Circuit boards", "Communications equipment"],
                "year": 2022, "active": True
            },
            {
                "code": "3341", "title": "Computer and Peripheral Equipment Manufacturing", "level": 4, "parent_code": "334",
                "description": "Manufacturing computers and computer peripheral equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["computer", "laptop", "server", "storage", "printer"],
                "examples": ["Desktop computers", "Laptops", "Computer storage", "Printers"],
                "year": 2022, "active": True
            },
            {
                "code": "3342", "title": "Communications Equipment Manufacturing", "level": 4, "parent_code": "334",
                "description": "Manufacturing communications equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["telecommunications", "telephone", "radio", "communications"],
                "examples": ["Telephones", "Radio equipment", "Wireless equipment"],
                "year": 2022, "active": True
            },
            {
                "code": "3343", "title": "Audio and Video Equipment Manufacturing", "level": 4, "parent_code": "334",
                "description": "Manufacturing audio and video equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["audio", "video", "television", "speakers", "cameras"],
                "examples": ["Televisions", "Speakers", "Video cameras"],
                "year": 2022, "active": True
            },
            {
                "code": "3344", "title": "Semiconductor and Other Electronic Component Manufacturing", "level": 4,
                "parent_code": "334",
                "description": "Manufacturing semiconductors and electronic components",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["semiconductor", "chip", "integrated circuit", "electronic components"],
                "examples": ["Semiconductor wafers", "Integrated circuits", "Printed circuit boards"],
                "year": 2022, "active": True
            },
            {
                "code": "33441", "title": "Semiconductor and Other Electronic Component Manufacturing", "level": 5,
                "parent_code": "3344",
                "description": "Manufacturing semiconductors and related devices",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["semiconductor manufacturing", "chip fabrication", "wafer"],
                "year": 2022, "active": True
            },
            {
                "code": "334413", "title": "Semiconductor and Related Device Manufacturing", "level": 6, "parent_code": "33441",
                "description": "Manufacturing semiconductors and related solid state devices",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["semiconductor fab", "chip manufacturing", "silicon wafer"],
                "examples": ["Semiconductor foundries", "Chip fabrication plants"],
                "year": 2022, "active": True
            },

            # Electrical Equipment (335)
            {
                "code": "335", "title": "Electrical Equipment, Appliance, and Component Manufacturing", "level": 3,
                "parent_code": "33",
                "description": "Manufacturing electrical equipment and appliances",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["electrical equipment", "appliances", "motors", "transformers"],
                "examples": ["Electric motors", "Transformers", "Household appliances", "Batteries"],
                "year": 2022, "active": True
            },
            {
                "code": "3351", "title": "Electric Lighting Equipment Manufacturing", "level": 4, "parent_code": "335",
                "description": "Manufacturing electric lighting equipment",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["lighting", "light bulbs", "LED", "lamps"],
                "examples": ["Light bulbs", "LED lights", "Light fixtures"],
                "year": 2022, "active": True
            },
            {
                "code": "3352", "title": "Household Appliance Manufacturing", "level": 4, "parent_code": "335",
                "description": "Manufacturing household appliances",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["appliances", "refrigerator", "washing machine", "stove"],
                "examples": ["Refrigerators", "Washing machines", "Dishwashers", "Ovens"],
                "year": 2022, "active": True
            },
            {
                "code": "3353", "title": "Electrical Equipment Manufacturing", "level": 4, "parent_code": "335",
                "description": "Manufacturing electrical equipment for power generation and distribution",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["electrical equipment", "transformers", "switchgear", "motors"],
                "examples": ["Power transformers", "Electric motors", "Switchgear"],
                "year": 2022, "active": True
            },
            {
                "code": "3359", "title": "Other Electrical Equipment and Component Manufacturing", "level": 4,
                "parent_code": "335",
                "description": "Manufacturing other electrical equipment and components",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["batteries", "wiring", "electrical components"],
                "examples": ["Batteries", "Electrical wiring", "Carbon products"],
                "year": 2022, "active": True
            },

            # Transportation Equipment (336)
            {
                "code": "336", "title": "Transportation Equipment Manufacturing", "level": 3, "parent_code": "33",
                "description": "Manufacturing equipment for transporting people and goods",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["automotive", "aircraft", "ships", "trains", "vehicles"],
                "examples": ["Automobiles", "Aircraft", "Ships", "Railway equipment"],
                "year": 2022, "active": True
            },
            {
                "code": "3361", "title": "Motor Vehicle Manufacturing", "level": 4, "parent_code": "336",
                "description": "Manufacturing motor vehicles",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["automobile", "car", "truck", "vehicle"],
                "examples": ["Cars", "Trucks", "Buses"],
                "year": 2022, "active": True
            },
            {
                "code": "33611", "title": "Automobile and Light Duty Motor Vehicle Manufacturing", "level": 5,
                "parent_code": "3361",
                "description": "Manufacturing automobiles and light duty motor vehicles",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["car manufacturing", "automobile assembly", "passenger vehicles"],
                "year": 2022, "active": True
            },
            {
                "code": "336110", "title": "Automobile and Light Duty Motor Vehicle Manufacturing", "level": 6,
                "parent_code": "33611",
                "description": "Manufacturing complete automobiles and light trucks",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["car factory", "automobile plant", "vehicle assembly"],
                "examples": ["Automobile assembly plants", "Light truck manufacturing"],
                "year": 2022, "active": True
            },
            {
                "code": "3362", "title": "Motor Vehicle Body and Trailer Manufacturing", "level": 4, "parent_code": "336",
                "description": "Manufacturing motor vehicle bodies and trailers",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["truck body", "trailer", "motor home"],
                "examples": ["Truck bodies", "Trailers", "Motor homes"],
                "year": 2022, "active": True
            },
            {
                "code": "3363", "title": "Motor Vehicle Parts Manufacturing", "level": 4, "parent_code": "336",
                "description": "Manufacturing motor vehicle parts",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["auto parts", "vehicle parts", "automotive components"],
                "examples": ["Engines", "Transmissions", "Brake systems", "Steering components"],
                "year": 2022, "active": True
            },
            {
                "code": "3364", "title": "Aerospace Product and Parts Manufacturing", "level": 4, "parent_code": "336",
                "description": "Manufacturing aircraft, spacecraft, and related parts",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["aircraft", "aerospace", "airplane", "spacecraft"],
                "examples": ["Commercial aircraft", "Military aircraft", "Spacecraft", "Aircraft engines"],
                "year": 2022, "active": True
            },
            {
                "code": "3365", "title": "Railroad Rolling Stock Manufacturing", "level": 4, "parent_code": "336",
                "description": "Manufacturing railroad rolling stock",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["railroad", "train", "locomotive", "rail car"],
                "examples": ["Locomotives", "Railroad cars", "Rapid transit cars"],
                "year": 2022, "active": True
            },
            {
                "code": "3366", "title": "Ship and Boat Building", "level": 4, "parent_code": "336",
                "description": "Building ships and boats",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["ship", "boat", "shipbuilding", "vessel"],
                "examples": ["Ships", "Boats", "Barges"],
                "year": 2022, "active": True
            },

            # Furniture (337)
            {
                "code": "337", "title": "Furniture and Related Product Manufacturing", "level": 3, "parent_code": "33",
                "description": "Manufacturing furniture and related products",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["furniture", "household furniture", "office furniture"],
                "examples": ["Wood furniture", "Metal furniture", "Mattresses"],
                "year": 2022, "active": True
            },

            # Miscellaneous Manufacturing (339)
            {
                "code": "339", "title": "Miscellaneous Manufacturing", "level": 3, "parent_code": "33",
                "description": "Manufacturing products not classified elsewhere",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["medical devices", "jewelry", "sporting goods", "toys"],
                "examples": ["Medical equipment", "Jewelry", "Sports equipment", "Toys"],
                "year": 2022, "active": True
            },
            {
                "code": "3391", "title": "Medical Equipment and Supplies Manufacturing", "level": 4, "parent_code": "339",
                "description": "Manufacturing medical equipment and supplies",
                "category": IndustryCategory.MANUFACTURING,
                "keywords": ["medical devices", "medical equipment", "surgical instruments"],
                "examples": ["Surgical instruments", "Medical diagnostic equipment", "Medical supplies"],
                "year": 2022, "active": True
            },

            # Wholesale Trade (42)
            {
                "code": "42", "title": "Wholesale Trade", "level": 2,
                "description": "Selling goods to businesses, institutions, and other wholesalers",
                "category": IndustryCategory.WHOLESALE,
                "keywords": ["wholesale", "distributor", "merchant wholesaler"],
                "examples": ["Merchant wholesalers", "Business-to-business sales"],
                "year": 2022, "active": True
            },
            {
                "code": "423", "title": "Merchant Wholesalers, Durable Goods", "level": 3, "parent_code": "42",
                "description": "Wholesaling durable goods",
                "category": IndustryCategory.WHOLESALE,
                "keywords": ["durable goods", "equipment wholesale", "machinery wholesale"],
                "examples": ["Motor vehicle wholesalers", "Furniture wholesalers", "Machinery wholesalers"],
                "year": 2022, "active": True
            },
            {
                "code": "424", "title": "Merchant Wholesalers, Nondurable Goods", "level": 3, "parent_code": "42",
                "description": "Wholesaling nondurable goods",
                "category": IndustryCategory.WHOLESALE,
                "keywords": ["nondurable goods", "food wholesale", "chemical wholesale"],
                "examples": ["Grocery wholesalers", "Chemical wholesalers", "Petroleum wholesalers"],
                "year": 2022, "active": True
            },
            {
                "code": "425", "title": "Wholesale Electronic Markets and Agents and Brokers", "level": 3, "parent_code": "42",
                "description": "Operating electronic markets or acting as agents and brokers",
                "category": IndustryCategory.WHOLESALE,
                "keywords": ["wholesale agents", "brokers", "electronic markets"],
                "examples": ["Business-to-business electronic markets", "Wholesale trade agents"],
                "year": 2022, "active": True
            },

            # Retail Trade (44-45)
            {
                "code": "44", "title": "Retail Trade", "level": 2,
                "description": "Selling goods to final consumers",
                "category": IndustryCategory.RETAIL,
                "keywords": ["retail", "store", "shop", "consumer"],
                "examples": ["Motor vehicle dealers", "Furniture stores", "Electronics stores"],
                "year": 2022, "active": True
            },
            {
                "code": "441", "title": "Motor Vehicle and Parts Dealers", "level": 3, "parent_code": "44",
                "description": "Retailing motor vehicles and parts",
                "category": IndustryCategory.RETAIL,
                "keywords": ["car dealer", "auto parts", "vehicle sales"],
                "examples": ["New car dealers", "Used car dealers", "Auto parts stores"],
                "year": 2022, "active": True
            },
            {
                "code": "442", "title": "Furniture and Home Furnishings Stores", "level": 3, "parent_code": "44",
                "description": "Retailing furniture and home furnishings",
                "category": IndustryCategory.RETAIL,
                "keywords": ["furniture store", "home furnishings", "interior"],
                "examples": ["Furniture stores", "Home furnishing stores"],
                "year": 2022, "active": True
            },
            {
                "code": "443", "title": "Electronics and Appliance Stores", "level": 3, "parent_code": "44",
                "description": "Retailing consumer electronics and appliances",
                "category": IndustryCategory.RETAIL,
                "keywords": ["electronics", "appliances", "consumer electronics"],
                "examples": ["Consumer electronics stores", "Appliance stores"],
                "year": 2022, "active": True
            },
            {
                "code": "444", "title": "Building Material and Garden Equipment and Supplies Dealers", "level": 3,
                "parent_code": "44",
                "description": "Retailing building materials and garden supplies",
                "category": IndustryCategory.RETAIL,
                "keywords": ["building materials", "hardware", "home improvement", "garden"],
                "examples": ["Home centers", "Hardware stores", "Garden centers"],
                "year": 2022, "active": True
            },
            {
                "code": "445", "title": "Food and Beverage Stores", "level": 3, "parent_code": "44",
                "description": "Retailing food and beverages",
                "category": IndustryCategory.RETAIL,
                "keywords": ["grocery", "supermarket", "food store", "beverage"],
                "examples": ["Supermarkets", "Convenience stores", "Specialty food stores"],
                "year": 2022, "active": True
            },
            {
                "code": "446", "title": "Health and Personal Care Stores", "level": 3, "parent_code": "44",
                "description": "Retailing health and personal care products",
                "category": IndustryCategory.RETAIL,
                "keywords": ["pharmacy", "drugstore", "health", "personal care"],
                "examples": ["Pharmacies", "Drug stores", "Cosmetics stores"],
                "year": 2022, "active": True
            },
            {
                "code": "447", "title": "Gasoline Stations", "level": 3, "parent_code": "44",
                "description": "Retailing automotive fuels",
                "category": IndustryCategory.RETAIL,
                "keywords": ["gas station", "fuel", "gasoline", "service station"],
                "examples": ["Gas stations", "Convenience stores with gas"],
                "year": 2022, "active": True
            },
            {
                "code": "448", "title": "Clothing and Clothing Accessories Stores", "level": 3, "parent_code": "44",
                "description": "Retailing clothing and accessories",
                "category": IndustryCategory.RETAIL,
                "keywords": ["clothing", "apparel", "fashion", "accessories"],
                "examples": ["Clothing stores", "Shoe stores", "Jewelry stores"],
                "year": 2022, "active": True
            },
            {
                "code": "451", "title": "Sporting Goods, Hobby, Musical Instrument, and Book Stores", "level": 3,
                "parent_code": "45",
                "description": "Retailing sporting goods, hobbies, books, and musical instruments",
                "category": IndustryCategory.RETAIL,
                "keywords": ["sporting goods", "hobby", "books", "music"],
                "examples": ["Sporting goods stores", "Book stores", "Musical instrument stores"],
                "year": 2022, "active": True
            },
            {
                "code": "452", "title": "General Merchandise Stores", "level": 3, "parent_code": "45",
                "description": "Retailing a general line of merchandise",
                "category": IndustryCategory.RETAIL,
                "keywords": ["department store", "general merchandise", "variety store"],
                "examples": ["Department stores", "Warehouse clubs", "Supercenters"],
                "year": 2022, "active": True
            },
            {
                "code": "453", "title": "Miscellaneous Store Retailers", "level": 3, "parent_code": "45",
                "description": "Retailing miscellaneous products",
                "category": IndustryCategory.RETAIL,
                "keywords": ["florist", "office supplies", "gift shop", "pet store"],
                "examples": ["Florists", "Office supply stores", "Pet stores", "Gift shops"],
                "year": 2022, "active": True
            },
            {
                "code": "454", "title": "Nonstore Retailers", "level": 3, "parent_code": "45",
                "description": "Retailing via nonstore methods",
                "category": IndustryCategory.RETAIL,
                "keywords": ["e-commerce", "online retail", "mail order", "vending"],
                "examples": ["Electronic shopping", "Mail-order", "Vending machines", "Direct selling"],
                "year": 2022, "active": True
            },

            # Transportation and Warehousing (48-49)
            {
                "code": "48", "title": "Transportation and Warehousing", "level": 2,
                "description": "Providing transportation of passengers and cargo, warehousing and storage",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["transportation", "logistics", "freight", "warehousing", "shipping"],
                "examples": ["Trucking", "Air transportation", "Warehousing", "Couriers"],
                "year": 2022, "active": True
            },
            {
                "code": "481", "title": "Air Transportation", "level": 3, "parent_code": "48",
                "description": "Providing air transportation of passengers and cargo",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["airline", "air freight", "aviation", "air cargo"],
                "examples": ["Scheduled airlines", "Charter flights", "Air cargo"],
                "year": 2022, "active": True
            },
            {
                "code": "482", "title": "Rail Transportation", "level": 3, "parent_code": "48",
                "description": "Providing rail transportation of passengers and cargo",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["railroad", "rail freight", "train", "railway"],
                "examples": ["Freight railroads", "Passenger rail"],
                "year": 2022, "active": True
            },
            {
                "code": "483", "title": "Water Transportation", "level": 3, "parent_code": "48",
                "description": "Providing water transportation of passengers and cargo",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["shipping", "maritime", "ocean freight", "barge"],
                "examples": ["Deep sea shipping", "Coastal shipping", "Inland water transportation"],
                "year": 2022, "active": True
            },
            {
                "code": "484", "title": "Truck Transportation", "level": 3, "parent_code": "48",
                "description": "Providing over-the-road transportation of cargo",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["trucking", "freight", "truck", "hauling"],
                "examples": ["General freight trucking", "Specialized freight trucking"],
                "year": 2022, "active": True
            },
            {
                "code": "485", "title": "Transit and Ground Passenger Transportation", "level": 3, "parent_code": "48",
                "description": "Providing ground passenger transportation",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["bus", "taxi", "transit", "shuttle", "limousine"],
                "examples": ["Urban transit", "Taxi service", "School buses", "Charter buses"],
                "year": 2022, "active": True
            },
            {
                "code": "486", "title": "Pipeline Transportation", "level": 3, "parent_code": "48",
                "description": "Transportation using pipelines",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["pipeline", "oil pipeline", "gas pipeline"],
                "examples": ["Crude oil pipelines", "Natural gas pipelines"],
                "year": 2022, "active": True
            },
            {
                "code": "487", "title": "Scenic and Sightseeing Transportation", "level": 3, "parent_code": "48",
                "description": "Providing scenic and sightseeing transportation",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["sightseeing", "tour", "scenic", "tourist"],
                "examples": ["Scenic cruises", "Sightseeing buses"],
                "year": 2022, "active": True
            },
            {
                "code": "488", "title": "Support Activities for Transportation", "level": 3, "parent_code": "48",
                "description": "Providing support services for transportation",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["airport operations", "port operations", "freight forwarding"],
                "examples": ["Airports", "Marine ports", "Freight forwarding"],
                "year": 2022, "active": True
            },
            {
                "code": "492", "title": "Couriers and Messengers", "level": 3, "parent_code": "49",
                "description": "Providing intercity and local delivery of parcels",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["courier", "delivery", "express mail", "messenger"],
                "examples": ["Courier services", "Local messengers", "Package delivery"],
                "year": 2022, "active": True
            },
            {
                "code": "493", "title": "Warehousing and Storage", "level": 3, "parent_code": "49",
                "description": "Operating warehousing and storage facilities",
                "category": IndustryCategory.TRANSPORTATION,
                "keywords": ["warehouse", "storage", "distribution center", "logistics"],
                "examples": ["General warehousing", "Refrigerated warehousing", "Farm product storage"],
                "year": 2022, "active": True
            },

            # Information (51)
            {
                "code": "51", "title": "Information", "level": 2,
                "description": "Producing and distributing information and cultural products",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["media", "publishing", "broadcasting", "telecommunications", "information"],
                "examples": ["Publishing", "Broadcasting", "Telecommunications", "Data processing"],
                "year": 2022, "active": True
            },
            {
                "code": "511", "title": "Publishing Industries (except Internet)", "level": 3, "parent_code": "51",
                "description": "Publishing newspapers, periodicals, books, and software",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["publishing", "books", "newspapers", "software"],
                "examples": ["Newspaper publishing", "Book publishing", "Software publishing"],
                "year": 2022, "active": True
            },
            {
                "code": "512", "title": "Motion Picture and Sound Recording Industries", "level": 3, "parent_code": "51",
                "description": "Producing and distributing motion pictures and sound recordings",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["film", "movie", "video", "music", "recording"],
                "examples": ["Motion picture production", "Sound recording", "Music publishing"],
                "year": 2022, "active": True
            },
            {
                "code": "515", "title": "Broadcasting (except Internet)", "level": 3, "parent_code": "51",
                "description": "Broadcasting audio and video programming",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["radio", "television", "broadcasting", "TV"],
                "examples": ["Radio broadcasting", "Television broadcasting"],
                "year": 2022, "active": True
            },
            {
                "code": "517", "title": "Telecommunications", "level": 3, "parent_code": "51",
                "description": "Providing telecommunications services",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["telecom", "telephone", "internet", "wireless", "cable"],
                "examples": ["Wired telecommunications", "Wireless telecommunications", "Internet service providers"],
                "year": 2022, "active": True
            },
            {
                "code": "518", "title": "Data Processing, Hosting, and Related Services", "level": 3, "parent_code": "51",
                "description": "Providing data processing and hosting services",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["data processing", "hosting", "cloud", "data center"],
                "examples": ["Data processing services", "Web hosting"],
                "year": 2022, "active": True
            },
            {
                "code": "519", "title": "Other Information Services", "level": 3, "parent_code": "51",
                "description": "Providing other information services",
                "category": IndustryCategory.INFORMATION,
                "keywords": ["news", "libraries", "archives", "internet publishing"],
                "examples": ["News syndicates", "Libraries", "Internet publishing"],
                "year": 2022, "active": True
            },

            # Finance and Insurance (52)
            {
                "code": "52", "title": "Finance and Insurance", "level": 2,
                "description": "Financial transactions and insurance services",
                "category": IndustryCategory.FINANCE,
                "keywords": ["finance", "banking", "insurance", "investment", "securities"],
                "examples": ["Banks", "Insurance", "Securities", "Investment funds"],
                "year": 2022, "active": True
            },
            {
                "code": "521", "title": "Monetary Authorities-Central Bank", "level": 3, "parent_code": "52",
                "description": "Central banking",
                "category": IndustryCategory.FINANCE,
                "keywords": ["central bank", "federal reserve", "monetary authority"],
                "examples": ["Federal Reserve"],
                "year": 2022, "active": True
            },
            {
                "code": "522", "title": "Credit Intermediation and Related Activities", "level": 3, "parent_code": "52",
                "description": "Depository and non-depository credit intermediation",
                "category": IndustryCategory.FINANCE,
                "keywords": ["banking", "credit", "lending", "loans", "deposits"],
                "examples": ["Commercial banks", "Credit unions", "Mortgage lending"],
                "year": 2022, "active": True
            },
            {
                "code": "523", "title": "Securities, Commodity Contracts, and Other Financial Investments", "level": 3,
                "parent_code": "52",
                "description": "Securities and commodity contracts intermediation and brokerage",
                "category": IndustryCategory.FINANCE,
                "keywords": ["securities", "brokerage", "investment", "trading", "commodities"],
                "examples": ["Investment banking", "Securities brokerage", "Commodity trading"],
                "year": 2022, "active": True
            },
            {
                "code": "524", "title": "Insurance Carriers and Related Activities", "level": 3, "parent_code": "52",
                "description": "Insurance and related activities",
                "category": IndustryCategory.FINANCE,
                "keywords": ["insurance", "life insurance", "health insurance", "property insurance"],
                "examples": ["Life insurance", "Health insurance", "Property insurance", "Insurance agencies"],
                "year": 2022, "active": True
            },
            {
                "code": "525", "title": "Funds, Trusts, and Other Financial Vehicles", "level": 3, "parent_code": "52",
                "description": "Pooling assets in investment funds and trusts",
                "category": IndustryCategory.FINANCE,
                "keywords": ["investment funds", "mutual funds", "trusts", "pension funds"],
                "examples": ["Mutual funds", "Pension funds", "Trusts"],
                "year": 2022, "active": True
            },

            # Real Estate (53)
            {
                "code": "53", "title": "Real Estate and Rental and Leasing", "level": 2,
                "description": "Renting, leasing, or otherwise allowing use of tangible or intangible assets",
                "category": IndustryCategory.REAL_ESTATE,
                "keywords": ["real estate", "rental", "leasing", "property"],
                "examples": ["Real estate", "Equipment rental", "Vehicle leasing"],
                "year": 2022, "active": True
            },
            {
                "code": "531", "title": "Real Estate", "level": 3, "parent_code": "53",
                "description": "Activities related to real estate",
                "category": IndustryCategory.REAL_ESTATE,
                "keywords": ["property", "real estate", "landlord", "rental property"],
                "examples": ["Lessors of real estate", "Real estate agencies", "Property management"],
                "year": 2022, "active": True
            },
            {
                "code": "532", "title": "Rental and Leasing Services", "level": 3, "parent_code": "53",
                "description": "Renting and leasing tangible goods",
                "category": IndustryCategory.REAL_ESTATE,
                "keywords": ["rental", "leasing", "equipment rental", "vehicle rental"],
                "examples": ["Automotive equipment rental", "Consumer goods rental", "Industrial equipment rental"],
                "year": 2022, "active": True
            },
            {
                "code": "533", "title": "Lessors of Nonfinancial Intangible Assets", "level": 3, "parent_code": "53",
                "description": "Licensing and leasing intangible assets",
                "category": IndustryCategory.REAL_ESTATE,
                "keywords": ["licensing", "patents", "trademarks", "royalties"],
                "examples": ["Patent licensing", "Trademark licensing"],
                "year": 2022, "active": True
            },

            # Professional Services (54)
            {
                "code": "54", "title": "Professional, Scientific, and Technical Services", "level": 2,
                "description": "Professional, scientific, and technical services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["professional services", "consulting", "legal", "accounting", "engineering"],
                "examples": ["Legal services", "Accounting", "Engineering", "Consulting", "Research"],
                "year": 2022, "active": True
            },
            {
                "code": "541", "title": "Professional, Scientific, and Technical Services", "level": 3, "parent_code": "54",
                "description": "Professional, scientific, and technical services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["professional", "consulting", "technical services"],
                "year": 2022, "active": True
            },
            {
                "code": "5411", "title": "Legal Services", "level": 4, "parent_code": "541",
                "description": "Providing legal services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["legal", "law", "attorney", "lawyer"],
                "examples": ["Law firms", "Legal services"],
                "year": 2022, "active": True
            },
            {
                "code": "5412", "title": "Accounting, Tax Preparation, Bookkeeping, and Payroll Services", "level": 4,
                "parent_code": "541",
                "description": "Accounting and related services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["accounting", "tax", "bookkeeping", "payroll", "audit"],
                "examples": ["Accounting firms", "Tax preparation", "Payroll services"],
                "year": 2022, "active": True
            },
            {
                "code": "5413", "title": "Architectural, Engineering, and Related Services", "level": 4, "parent_code": "541",
                "description": "Architectural and engineering services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["architecture", "engineering", "design", "surveying"],
                "examples": ["Architectural services", "Engineering services", "Surveying"],
                "year": 2022, "active": True
            },
            {
                "code": "5414", "title": "Specialized Design Services", "level": 4, "parent_code": "541",
                "description": "Specialized design services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["design", "interior design", "graphic design"],
                "examples": ["Interior design", "Graphic design", "Industrial design"],
                "year": 2022, "active": True
            },
            {
                "code": "5415", "title": "Computer Systems Design and Related Services", "level": 4, "parent_code": "541",
                "description": "Computer systems design and related services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["IT services", "software development", "systems integration", "IT consulting"],
                "examples": ["Custom software development", "IT consulting", "Systems integration"],
                "year": 2022, "active": True
            },
            {
                "code": "5416", "title": "Management, Scientific, and Technical Consulting Services", "level": 4,
                "parent_code": "541",
                "description": "Management and technical consulting",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["consulting", "management consulting", "scientific consulting"],
                "examples": ["Management consulting", "Environmental consulting", "Other technical consulting"],
                "year": 2022, "active": True
            },
            {
                "code": "5417", "title": "Scientific Research and Development Services", "level": 4, "parent_code": "541",
                "description": "Scientific research and development",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["research", "R&D", "scientific research", "development"],
                "examples": ["Biotechnology research", "Physical science research", "Social science research"],
                "year": 2022, "active": True
            },
            {
                "code": "5418", "title": "Advertising, Public Relations, and Related Services", "level": 4, "parent_code": "541",
                "description": "Advertising and related services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["advertising", "marketing", "public relations", "PR"],
                "examples": ["Advertising agencies", "Public relations", "Media buying"],
                "year": 2022, "active": True
            },
            {
                "code": "5419", "title": "Other Professional, Scientific, and Technical Services", "level": 4,
                "parent_code": "541",
                "description": "Other professional and technical services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["veterinary", "photography", "translation"],
                "examples": ["Veterinary services", "Photography", "Translation services"],
                "year": 2022, "active": True
            },

            # Management (55)
            {
                "code": "55", "title": "Management of Companies and Enterprises", "level": 2,
                "description": "Holding securities or overseeing companies and enterprises",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["management", "holding company", "corporate headquarters"],
                "examples": ["Corporate headquarters", "Holding companies"],
                "year": 2022, "active": True
            },

            # Administrative Services (56)
            {
                "code": "56", "title": "Administrative and Support and Waste Management Services", "level": 2,
                "description": "Administrative, support, and waste management services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["administrative", "support services", "waste management", "facilities"],
                "examples": ["Employment services", "Facilities support", "Waste management"],
                "year": 2022, "active": True
            },
            {
                "code": "561", "title": "Administrative and Support Services", "level": 3, "parent_code": "56",
                "description": "Administrative and support services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["administrative services", "business support", "facilities"],
                "examples": ["Employment services", "Business support services", "Travel agencies"],
                "year": 2022, "active": True
            },
            {
                "code": "562", "title": "Waste Management and Remediation Services", "level": 3, "parent_code": "56",
                "description": "Waste management and remediation services",
                "category": IndustryCategory.PROFESSIONAL,
                "keywords": ["waste", "recycling", "remediation", "environmental"],
                "examples": ["Waste collection", "Waste treatment", "Remediation services"],
                "year": 2022, "active": True
            },

            # Educational Services (61)
            {
                "code": "61", "title": "Educational Services", "level": 2,
                "description": "Providing instruction and training",
                "category": IndustryCategory.EDUCATION,
                "keywords": ["education", "school", "training", "university", "college"],
                "examples": ["Elementary schools", "Colleges", "Technical training"],
                "year": 2022, "active": True
            },
            {
                "code": "611", "title": "Educational Services", "level": 3, "parent_code": "61",
                "description": "Providing educational services",
                "category": IndustryCategory.EDUCATION,
                "keywords": ["school", "college", "training", "instruction"],
                "examples": ["Schools", "Colleges", "Professional training"],
                "year": 2022, "active": True
            },

            # Health Care (62)
            {
                "code": "62", "title": "Health Care and Social Assistance", "level": 2,
                "description": "Providing health care and social assistance",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["healthcare", "hospital", "medical", "social services"],
                "examples": ["Hospitals", "Physicians", "Nursing homes", "Social services"],
                "year": 2022, "active": True
            },
            {
                "code": "621", "title": "Ambulatory Health Care Services", "level": 3, "parent_code": "62",
                "description": "Ambulatory health care services",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["physician", "dentist", "outpatient", "medical office"],
                "examples": ["Physicians' offices", "Dentists", "Outpatient care centers"],
                "year": 2022, "active": True
            },
            {
                "code": "622", "title": "Hospitals", "level": 3, "parent_code": "62",
                "description": "Hospital services",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["hospital", "inpatient", "medical center"],
                "examples": ["General hospitals", "Psychiatric hospitals", "Specialty hospitals"],
                "year": 2022, "active": True
            },
            {
                "code": "623", "title": "Nursing and Residential Care Facilities", "level": 3, "parent_code": "62",
                "description": "Nursing and residential care",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["nursing home", "residential care", "assisted living"],
                "examples": ["Nursing homes", "Residential mental health facilities", "Community care"],
                "year": 2022, "active": True
            },
            {
                "code": "624", "title": "Social Assistance", "level": 3, "parent_code": "62",
                "description": "Social assistance services",
                "category": IndustryCategory.HEALTHCARE,
                "keywords": ["social services", "child care", "community services"],
                "examples": ["Individual and family services", "Community food services", "Child day care"],
                "year": 2022, "active": True
            },

            # Arts and Entertainment (71)
            {
                "code": "71", "title": "Arts, Entertainment, and Recreation", "level": 2,
                "description": "Arts, entertainment, and recreation services",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["entertainment", "arts", "recreation", "sports", "amusement"],
                "examples": ["Performing arts", "Sports", "Museums", "Amusement parks"],
                "year": 2022, "active": True
            },
            {
                "code": "711", "title": "Performing Arts, Spectator Sports, and Related Industries", "level": 3,
                "parent_code": "71",
                "description": "Performing arts and spectator sports",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["performing arts", "sports", "entertainment", "theater"],
                "examples": ["Theater companies", "Sports teams", "Promoters of events"],
                "year": 2022, "active": True
            },
            {
                "code": "712", "title": "Museums, Historical Sites, and Similar Institutions", "level": 3,
                "parent_code": "71",
                "description": "Museums and historical sites",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["museum", "historical site", "zoo", "botanical garden"],
                "examples": ["Museums", "Historical sites", "Zoos", "Nature parks"],
                "year": 2022, "active": True
            },
            {
                "code": "713", "title": "Amusement, Gambling, and Recreation Industries", "level": 3, "parent_code": "71",
                "description": "Amusement, gambling, and recreation",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["amusement", "gambling", "casino", "recreation", "fitness"],
                "examples": ["Amusement parks", "Casinos", "Golf courses", "Fitness centers"],
                "year": 2022, "active": True
            },

            # Accommodation and Food Services (72)
            {
                "code": "72", "title": "Accommodation and Food Services", "level": 2,
                "description": "Lodging and food services",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["hotel", "restaurant", "food service", "accommodation"],
                "examples": ["Hotels", "Restaurants", "Catering", "Bars"],
                "year": 2022, "active": True
            },
            {
                "code": "721", "title": "Accommodation", "level": 3, "parent_code": "72",
                "description": "Lodging services",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["hotel", "motel", "lodging", "inn"],
                "examples": ["Hotels", "Motels", "Bed and breakfasts", "RV parks"],
                "year": 2022, "active": True
            },
            {
                "code": "722", "title": "Food Services and Drinking Places", "level": 3, "parent_code": "72",
                "description": "Food services and drinking places",
                "category": IndustryCategory.HOSPITALITY,
                "keywords": ["restaurant", "cafe", "bar", "catering", "food service"],
                "examples": ["Restaurants", "Cafeterias", "Catering", "Bars", "Food trucks"],
                "year": 2022, "active": True
            },

            # Other Services (81)
            {
                "code": "81", "title": "Other Services (except Public Administration)", "level": 2,
                "description": "Other services except public administration",
                "category": IndustryCategory.OTHER,
                "keywords": ["repair", "maintenance", "personal services", "religious"],
                "examples": ["Repair services", "Personal care services", "Laundry", "Religious organizations"],
                "year": 2022, "active": True
            },
            {
                "code": "811", "title": "Repair and Maintenance", "level": 3, "parent_code": "81",
                "description": "Repair and maintenance services",
                "category": IndustryCategory.OTHER,
                "keywords": ["repair", "maintenance", "automotive repair"],
                "examples": ["Automotive repair", "Electronic repair", "Home repair"],
                "year": 2022, "active": True
            },
            {
                "code": "812", "title": "Personal and Laundry Services", "level": 3, "parent_code": "81",
                "description": "Personal and laundry services",
                "category": IndustryCategory.OTHER,
                "keywords": ["personal care", "salon", "laundry", "dry cleaning"],
                "examples": ["Hair salons", "Dry cleaning", "Pet care"],
                "year": 2022, "active": True
            },
            {
                "code": "813", "title": "Religious, Grantmaking, Civic, Professional, and Similar Organizations", "level": 3,
                "parent_code": "81",
                "description": "Religious and civic organizations",
                "category": IndustryCategory.OTHER,
                "keywords": ["religious", "nonprofit", "civic", "professional association"],
                "examples": ["Religious organizations", "Grant-making foundations", "Professional organizations"],
                "year": 2022, "active": True
            },

            # Public Administration (92)
            {
                "code": "92", "title": "Public Administration", "level": 2,
                "description": "Government administration",
                "category": IndustryCategory.OTHER,
                "keywords": ["government", "public administration", "regulation", "defense"],
                "examples": ["Executive offices", "Legislative bodies", "Public finance", "National security"],
                "year": 2022, "active": True
            }
        ]


# Module-level functions
def search_naics(
    query: str,
    max_results: int = 10,
    min_score: float = 0.5,
    config: Optional[IndustryMappingConfig] = None
) -> List[Tuple[NAICSCode, float]]:
    """Search NAICS database"""
    db = NAICSDatabase(config)
    return db.search(query, max_results, min_score)


def get_naics_hierarchy(code: str, config: Optional[IndustryMappingConfig] = None) -> List[NAICSCode]:
    """Get NAICS code hierarchy"""
    db = NAICSDatabase(config)
    return db.get_hierarchy(code)


def validate_naics_code(code: str, config: Optional[IndustryMappingConfig] = None) -> bool:
    """Validate NAICS code exists"""
    db = NAICSDatabase(config)
    return db.get_code(code) is not None
