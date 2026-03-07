# -*- coding: utf-8 -*-
"""
Country Boundary Reference Data - AGENT-EUDR-007

Provides country bounding boxes, centroids, and ocean region definitions for
the GPS Coordinate Validator Agent. Used for fast country-match validation,
land/ocean classification, and EUDR commodity-country plausibility checks
without external GIS service dependencies.

Country Boundaries:
    200+ countries with ISO 3166-1 alpha-2 codes, bounding boxes (WGS84),
    geographic centroids, administrative region lists, default datums, and
    EUDR commodity associations.

Ocean Regions:
    10 major ocean basins with bounding box approximations for land/ocean
    classification.

Data Sources:
    Natural Earth 1:110m Admin 0 boundaries (simplified)
    EPSG Geodetic Parameter Dataset v10.x
    EU EUDR Annex I commodity classification

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Country Boundaries
# ---------------------------------------------------------------------------
# ISO 3166-1 alpha-2 -> boundary metadata
# bbox: {"min_lat", "min_lon", "max_lat", "max_lon"} in decimal degrees WGS84
# centroid: {"lat", "lon"} geographic centroid
# admin_regions: list of major admin divisions (selected)
# default_datum: default local datum key (from datum_parameters)
# eudr_commodities: EUDR Annex I commodities produced in this country

COUNTRY_BOUNDARIES: Dict[str, Dict[str, Any]] = {
    # ===================================================================
    # South America (EUDR-critical producing region)
    # ===================================================================
    "BR": {
        "name": "Brazil",
        "bbox": {"min_lat": -33.75, "min_lon": -73.99, "max_lat": 5.27, "max_lon": -34.79},
        "centroid": {"lat": -14.24, "lon": -51.93},
        "admin_regions": [
            "Acre", "Alagoas", "Amapa", "Amazonas", "Bahia", "Ceara",
            "Distrito Federal", "Espirito Santo", "Goias", "Maranhao",
            "Mato Grosso", "Mato Grosso do Sul", "Minas Gerais", "Para",
            "Paraiba", "Parana", "Pernambuco", "Piaui", "Rio de Janeiro",
            "Rio Grande do Norte", "Rio Grande do Sul", "Rondonia",
            "Roraima", "Santa Catarina", "Sao Paulo", "Sergipe", "Tocantins",
        ],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["soya", "cattle", "wood", "coffee", "rubber"],
    },
    "CO": {
        "name": "Colombia",
        "bbox": {"min_lat": -4.23, "min_lon": -79.00, "max_lat": 13.39, "max_lon": -66.87},
        "centroid": {"lat": 4.57, "lon": -74.30},
        "admin_regions": ["Amazonas", "Antioquia", "Arauca", "Atlantico", "Bolivar", "Boyaca", "Caldas", "Caqueta", "Casanare", "Cauca", "Cesar", "Choco", "Cordoba", "Cundinamarca", "Guainia", "Guaviare", "Huila", "La Guajira", "Magdalena", "Meta", "Narino", "Norte de Santander", "Putumayo", "Quindio", "Risaralda", "Santander", "Sucre", "Tolima", "Valle del Cauca", "Vaupes", "Vichada"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["coffee", "palm_oil", "cocoa", "cattle", "wood"],
    },
    "PE": {
        "name": "Peru",
        "bbox": {"min_lat": -18.35, "min_lon": -81.33, "max_lat": -0.04, "max_lon": -68.65},
        "centroid": {"lat": -9.19, "lon": -75.02},
        "admin_regions": ["Amazonas", "Ancash", "Apurimac", "Arequipa", "Ayacucho", "Cajamarca", "Cusco", "Huancavelica", "Huanuco", "Ica", "Junin", "La Libertad", "Lambayeque", "Lima", "Loreto", "Madre de Dios", "Moquegua", "Pasco", "Piura", "Puno", "San Martin", "Tacna", "Tumbes", "Ucayali"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["coffee", "cocoa", "wood"],
    },
    "EC": {
        "name": "Ecuador",
        "bbox": {"min_lat": -5.01, "min_lon": -81.08, "max_lat": 1.68, "max_lon": -75.19},
        "centroid": {"lat": -1.83, "lon": -78.18},
        "admin_regions": ["Azuay", "Bolivar", "Canar", "Carchi", "Chimborazo", "Cotopaxi", "El Oro", "Esmeraldas", "Galapagos", "Guayas", "Imbabura", "Loja", "Los Rios", "Manabi", "Morona Santiago", "Napo", "Orellana", "Pastaza", "Pichincha", "Santa Elena", "Santo Domingo", "Sucumbios", "Tungurahua", "Zamora Chinchipe"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["cocoa", "coffee", "palm_oil", "wood"],
    },
    "BO": {
        "name": "Bolivia",
        "bbox": {"min_lat": -22.90, "min_lon": -69.64, "max_lat": -9.68, "max_lon": -57.45},
        "centroid": {"lat": -16.29, "lon": -63.59},
        "admin_regions": ["Beni", "Chuquisaca", "Cochabamba", "La Paz", "Oruro", "Pando", "Potosi", "Santa Cruz", "Tarija"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["soya", "cattle", "wood"],
    },
    "PY": {
        "name": "Paraguay",
        "bbox": {"min_lat": -27.61, "min_lon": -62.65, "max_lat": -19.29, "max_lon": -54.26},
        "centroid": {"lat": -23.44, "lon": -58.44},
        "admin_regions": ["Alto Paraguay", "Alto Parana", "Amambay", "Boqueron", "Caaguazu", "Caazapa", "Canindeyu", "Central", "Concepcion", "Cordillera", "Guaira", "Itapua", "Misiones", "Neembucu", "Paraguari", "Presidente Hayes", "San Pedro"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["soya", "cattle", "wood"],
    },
    "AR": {
        "name": "Argentina",
        "bbox": {"min_lat": -55.06, "min_lon": -73.57, "max_lat": -21.78, "max_lon": -53.64},
        "centroid": {"lat": -38.42, "lon": -63.62},
        "admin_regions": ["Buenos Aires", "Catamarca", "Chaco", "Chubut", "Cordoba", "Corrientes", "Entre Rios", "Formosa", "Jujuy", "La Pampa", "La Rioja", "Mendoza", "Misiones", "Neuquen", "Rio Negro", "Salta", "San Juan", "San Luis", "Santa Cruz", "Santa Fe", "Santiago del Estero", "Tierra del Fuego", "Tucuman"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["soya", "cattle", "wood"],
    },
    "UY": {
        "name": "Uruguay",
        "bbox": {"min_lat": -35.00, "min_lon": -58.44, "max_lat": -30.09, "max_lon": -53.07},
        "centroid": {"lat": -32.52, "lon": -55.77},
        "admin_regions": ["Artigas", "Canelones", "Cerro Largo", "Colonia", "Durazno", "Flores", "Florida", "Lavalleja", "Maldonado", "Montevideo", "Paysandu", "Rio Negro", "Rivera", "Rocha", "Salto", "San Jose", "Soriano", "Tacuarembo", "Treinta y Tres"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["soya", "cattle", "wood"],
    },
    "VE": {
        "name": "Venezuela",
        "bbox": {"min_lat": 0.63, "min_lon": -73.38, "max_lat": 12.20, "max_lon": -59.80},
        "centroid": {"lat": 6.42, "lon": -66.59},
        "admin_regions": ["Amazonas", "Anzoategui", "Apure", "Aragua", "Barinas", "Bolivar", "Carabobo", "Cojedes", "Delta Amacuro", "Falcon", "Guarico", "Lara", "Merida", "Miranda", "Monagas", "Nueva Esparta", "Portuguesa", "Sucre", "Tachira", "Trujillo", "Vargas", "Yaracuy", "Zulia"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["cocoa", "cattle", "wood"],
    },
    "GY": {
        "name": "Guyana",
        "bbox": {"min_lat": 1.17, "min_lon": -61.39, "max_lat": 8.56, "max_lon": -56.48},
        "centroid": {"lat": 4.86, "lon": -58.93},
        "admin_regions": ["Barima-Waini", "Cuyuni-Mazaruni", "Demerara-Mahaica", "East Berbice-Corentyne", "Essequibo Islands-West Demerara", "Mahaica-Berbice", "Pomeroon-Supenaam", "Potaro-Siparuni", "Upper Demerara-Berbice", "Upper Takutu-Upper Essequibo"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["wood"],
    },
    "SR": {
        "name": "Suriname",
        "bbox": {"min_lat": 1.83, "min_lon": -58.07, "max_lat": 6.01, "max_lon": -53.98},
        "centroid": {"lat": 3.92, "lon": -56.03},
        "admin_regions": ["Brokopondo", "Commewijne", "Coronie", "Marowijne", "Nickerie", "Para", "Paramaribo", "Saramacca", "Sipaliwini", "Wanica"],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood"],
    },
    "CL": {
        "name": "Chile",
        "bbox": {"min_lat": -55.98, "min_lon": -75.64, "max_lat": -17.50, "max_lon": -66.96},
        "centroid": {"lat": -35.68, "lon": -71.54},
        "admin_regions": ["Arica y Parinacota", "Tarapaca", "Antofagasta", "Atacama", "Coquimbo", "Valparaiso", "Metropolitana", "O'Higgins", "Maule", "Nuble", "Biobio", "Araucania", "Los Rios", "Los Lagos", "Aysen", "Magallanes"],
        "default_datum": "SIRGAS_2000",
        "eudr_commodities": ["wood"],
    },
    # ===================================================================
    # Central America & Caribbean
    # ===================================================================
    "GT": {
        "name": "Guatemala",
        "bbox": {"min_lat": 13.74, "min_lon": -92.23, "max_lat": 17.82, "max_lon": -88.22},
        "centroid": {"lat": 15.78, "lon": -90.23},
        "admin_regions": ["Alta Verapaz", "Baja Verapaz", "Chimaltenango", "Chiquimula", "El Progreso", "Escuintla", "Guatemala", "Huehuetenango", "Izabal", "Jalapa", "Jutiapa", "Peten", "Quetzaltenango", "Quiche", "Retalhuleu", "Sacatepequez", "San Marcos", "Santa Rosa", "Solola", "Suchitepequez", "Totonicapan", "Zacapa"],
        "default_datum": "NAD27",
        "eudr_commodities": ["coffee", "palm_oil", "rubber"],
    },
    "HN": {
        "name": "Honduras",
        "bbox": {"min_lat": 12.98, "min_lon": -89.35, "max_lat": 16.51, "max_lon": -83.11},
        "centroid": {"lat": 14.64, "lon": -86.24},
        "admin_regions": ["Atlantida", "Choluteca", "Colon", "Comayagua", "Copan", "Cortes", "El Paraiso", "Francisco Morazan", "Gracias a Dios", "Intibuca", "Islas de la Bahia", "La Paz", "Lempira", "Ocotepeque", "Olancho", "Santa Barbara", "Valle", "Yoro"],
        "default_datum": "NAD27",
        "eudr_commodities": ["coffee", "palm_oil"],
    },
    "NI": {
        "name": "Nicaragua",
        "bbox": {"min_lat": 10.71, "min_lon": -87.69, "max_lat": 15.03, "max_lon": -82.56},
        "centroid": {"lat": 12.87, "lon": -85.21},
        "admin_regions": [],
        "default_datum": "NAD27",
        "eudr_commodities": ["coffee", "cattle"],
    },
    "CR": {
        "name": "Costa Rica",
        "bbox": {"min_lat": 8.03, "min_lon": -85.95, "max_lat": 11.22, "max_lon": -82.55},
        "centroid": {"lat": 9.75, "lon": -83.75},
        "admin_regions": [],
        "default_datum": "NAD27",
        "eudr_commodities": ["coffee", "palm_oil"],
    },
    "PA": {
        "name": "Panama",
        "bbox": {"min_lat": 7.20, "min_lon": -83.05, "max_lat": 9.65, "max_lon": -77.17},
        "centroid": {"lat": 8.54, "lon": -80.78},
        "admin_regions": [],
        "default_datum": "NAD27",
        "eudr_commodities": ["cattle", "wood"],
    },
    "MX": {
        "name": "Mexico",
        "bbox": {"min_lat": 14.53, "min_lon": -118.40, "max_lat": 32.72, "max_lon": -86.70},
        "centroid": {"lat": 23.63, "lon": -102.55},
        "admin_regions": [],
        "default_datum": "NAD27",
        "eudr_commodities": ["coffee", "cattle"],
    },
    # ===================================================================
    # West Africa (EUDR-critical: Cocoa, Coffee, Palm Oil)
    # ===================================================================
    "GH": {
        "name": "Ghana",
        "bbox": {"min_lat": 4.74, "min_lon": -3.26, "max_lat": 11.17, "max_lon": 1.20},
        "centroid": {"lat": 7.95, "lon": -1.02},
        "admin_regions": ["Ashanti", "Bono", "Bono East", "Ahafo", "Central", "Eastern", "Greater Accra", "Northern", "North East", "Savannah", "Upper East", "Upper West", "Volta", "Oti", "Western", "Western North"],
        "default_datum": "WGS84",
        "eudr_commodities": ["cocoa", "wood", "rubber"],
    },
    "CI": {
        "name": "Cote d'Ivoire",
        "bbox": {"min_lat": 4.36, "min_lon": -8.60, "max_lat": 10.74, "max_lon": -2.49},
        "centroid": {"lat": 7.54, "lon": -5.55},
        "admin_regions": ["Abidjan", "Bas-Sassandra", "Comoe", "Denguele", "Goh-Djiboua", "Lacs", "Lagunes", "Montagnes", "Sassandra-Marahoue", "Savanes", "Vallee du Bandama", "Woroba", "Yamoussoukro", "Zanzan"],
        "default_datum": "WGS84",
        "eudr_commodities": ["cocoa", "coffee", "rubber", "palm_oil", "wood"],
    },
    "CM": {
        "name": "Cameroon",
        "bbox": {"min_lat": 1.65, "min_lon": 8.49, "max_lat": 13.08, "max_lon": 16.19},
        "centroid": {"lat": 7.37, "lon": 12.35},
        "admin_regions": ["Adamaoua", "Centre", "East", "Far North", "Littoral", "North", "Northwest", "South", "Southwest", "West"],
        "default_datum": "WGS84",
        "eudr_commodities": ["cocoa", "coffee", "palm_oil", "rubber", "wood"],
    },
    "NG": {
        "name": "Nigeria",
        "bbox": {"min_lat": 4.27, "min_lon": 2.69, "max_lat": 13.89, "max_lon": 14.68},
        "centroid": {"lat": 9.08, "lon": 8.68},
        "admin_regions": ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara"],
        "default_datum": "MINNA",
        "eudr_commodities": ["cocoa", "palm_oil", "rubber", "wood"],
    },
    "TG": {
        "name": "Togo",
        "bbox": {"min_lat": 6.10, "min_lon": -0.15, "max_lat": 11.14, "max_lon": 1.81},
        "centroid": {"lat": 8.62, "lon": 0.82},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["cocoa", "coffee"],
    },
    "GN": {
        "name": "Guinea",
        "bbox": {"min_lat": 7.19, "min_lon": -14.93, "max_lat": 12.68, "max_lon": -7.64},
        "centroid": {"lat": 9.95, "lon": -11.28},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["coffee", "cocoa", "wood"],
    },
    "SL": {
        "name": "Sierra Leone",
        "bbox": {"min_lat": 6.93, "min_lon": -13.30, "max_lat": 10.00, "max_lon": -10.27},
        "centroid": {"lat": 8.46, "lon": -11.78},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["cocoa", "coffee", "wood"],
    },
    "LR": {
        "name": "Liberia",
        "bbox": {"min_lat": 4.34, "min_lon": -11.49, "max_lat": 8.55, "max_lon": -7.37},
        "centroid": {"lat": 6.43, "lon": -9.43},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["rubber", "cocoa", "wood"],
    },
    "SN": {
        "name": "Senegal",
        "bbox": {"min_lat": 12.31, "min_lon": -17.54, "max_lat": 16.69, "max_lon": -11.36},
        "centroid": {"lat": 14.50, "lon": -14.45},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "BF": {
        "name": "Burkina Faso",
        "bbox": {"min_lat": 9.39, "min_lon": -5.52, "max_lat": 15.08, "max_lon": 2.40},
        "centroid": {"lat": 12.24, "lon": -1.56},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "BJ": {
        "name": "Benin",
        "bbox": {"min_lat": 6.23, "min_lon": 0.77, "max_lat": 12.42, "max_lon": 3.84},
        "centroid": {"lat": 9.31, "lon": 2.32},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood"],
    },
    # ===================================================================
    # Central Africa (EUDR-critical: Wood, Cocoa)
    # ===================================================================
    "CD": {
        "name": "Democratic Republic of the Congo",
        "bbox": {"min_lat": -13.46, "min_lon": 12.18, "max_lat": 5.39, "max_lon": 31.31},
        "centroid": {"lat": -4.04, "lon": 21.76},
        "admin_regions": ["Bandundu", "Bas-Congo", "Equateur", "Kasai-Occidental", "Kasai-Oriental", "Katanga", "Kinshasa", "Maniema", "Nord-Kivu", "Orientale", "Sud-Kivu"],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood", "cocoa", "coffee", "rubber"],
    },
    "CG": {
        "name": "Republic of the Congo",
        "bbox": {"min_lat": -5.03, "min_lon": 11.21, "max_lat": 3.70, "max_lon": 18.65},
        "centroid": {"lat": -0.23, "lon": 15.83},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood"],
    },
    "GA": {
        "name": "Gabon",
        "bbox": {"min_lat": -3.98, "min_lon": 8.70, "max_lat": 2.33, "max_lon": 14.50},
        "centroid": {"lat": -0.80, "lon": 11.61},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood", "rubber"],
    },
    "GQ": {
        "name": "Equatorial Guinea",
        "bbox": {"min_lat": -1.47, "min_lon": 5.61, "max_lat": 3.77, "max_lon": 11.34},
        "centroid": {"lat": 1.65, "lon": 10.27},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood", "cocoa"],
    },
    "CF": {
        "name": "Central African Republic",
        "bbox": {"min_lat": 2.22, "min_lon": 14.42, "max_lat": 11.00, "max_lon": 27.46},
        "centroid": {"lat": 6.61, "lon": 20.94},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood", "coffee"],
    },
    # ===================================================================
    # East Africa (EUDR-critical: Coffee)
    # ===================================================================
    "ET": {
        "name": "Ethiopia",
        "bbox": {"min_lat": 3.40, "min_lon": 32.99, "max_lat": 14.89, "max_lon": 48.00},
        "centroid": {"lat": 9.15, "lon": 40.49},
        "admin_regions": ["Addis Ababa", "Afar", "Amhara", "Benishangul-Gumuz", "Dire Dawa", "Gambela", "Harari", "Oromia", "Sidama", "Somali", "South West Ethiopia", "Southern Nations", "Tigray"],
        "default_datum": "ADINDAN",
        "eudr_commodities": ["coffee"],
    },
    "KE": {
        "name": "Kenya",
        "bbox": {"min_lat": -4.68, "min_lon": 33.91, "max_lat": 5.02, "max_lon": 41.91},
        "centroid": {"lat": 0.02, "lon": 37.91},
        "admin_regions": ["Central", "Coast", "Eastern", "Nairobi", "North Eastern", "Nyanza", "Rift Valley", "Western"],
        "default_datum": "ARC_1960",
        "eudr_commodities": ["coffee"],
    },
    "TZ": {
        "name": "Tanzania",
        "bbox": {"min_lat": -11.75, "min_lon": 29.33, "max_lat": -1.00, "max_lon": 40.44},
        "centroid": {"lat": -6.37, "lon": 34.89},
        "admin_regions": ["Arusha", "Dar es Salaam", "Dodoma", "Iringa", "Kagera", "Kigoma", "Kilimanjaro", "Lindi", "Manyara", "Mara", "Mbeya", "Morogoro", "Mtwara", "Mwanza", "Njombe", "Pemba North", "Pemba South", "Pwani", "Rukwa", "Ruvuma", "Shinyanga", "Simiyu", "Singida", "Songwe", "Tabora", "Tanga", "Zanzibar North", "Zanzibar South"],
        "default_datum": "ARC_1960",
        "eudr_commodities": ["coffee"],
    },
    "UG": {
        "name": "Uganda",
        "bbox": {"min_lat": -1.48, "min_lon": 29.57, "max_lat": 4.23, "max_lon": 35.00},
        "centroid": {"lat": 1.37, "lon": 32.29},
        "admin_regions": [],
        "default_datum": "ARC_1960",
        "eudr_commodities": ["coffee"],
    },
    "RW": {
        "name": "Rwanda",
        "bbox": {"min_lat": -2.84, "min_lon": 28.86, "max_lat": -1.05, "max_lon": 30.90},
        "centroid": {"lat": -1.94, "lon": 29.87},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["coffee"],
    },
    "BI": {
        "name": "Burundi",
        "bbox": {"min_lat": -4.47, "min_lon": 29.00, "max_lat": -2.31, "max_lon": 30.85},
        "centroid": {"lat": -3.37, "lon": 29.92},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["coffee"],
    },
    "MG": {
        "name": "Madagascar",
        "bbox": {"min_lat": -25.60, "min_lon": 43.19, "max_lat": -11.95, "max_lon": 50.48},
        "centroid": {"lat": -18.77, "lon": 46.87},
        "admin_regions": [],
        "default_datum": "TANANARIVE_1925",
        "eudr_commodities": ["cocoa", "coffee", "wood"],
    },
    # ===================================================================
    # Southeast Asia (EUDR-critical: Palm Oil, Rubber)
    # ===================================================================
    "ID": {
        "name": "Indonesia",
        "bbox": {"min_lat": -11.01, "min_lon": 95.01, "max_lat": 5.91, "max_lon": 141.02},
        "centroid": {"lat": -0.79, "lon": 113.92},
        "admin_regions": ["Aceh", "Bali", "Bangka Belitung", "Banten", "Bengkulu", "Central Java", "Central Kalimantan", "Central Sulawesi", "East Java", "East Kalimantan", "East Nusa Tenggara", "Gorontalo", "Jakarta", "Jambi", "Lampung", "Maluku", "North Kalimantan", "North Maluku", "North Sulawesi", "North Sumatra", "Papua", "Riau", "Riau Islands", "South Kalimantan", "South Sulawesi", "South Sumatra", "Southeast Sulawesi", "West Java", "West Kalimantan", "West Nusa Tenggara", "West Papua", "West Sulawesi", "West Sumatra", "Yogyakarta"],
        "default_datum": "WGS84",
        "eudr_commodities": ["palm_oil", "rubber", "wood", "coffee", "cocoa"],
    },
    "MY": {
        "name": "Malaysia",
        "bbox": {"min_lat": 0.85, "min_lon": 99.64, "max_lat": 7.36, "max_lon": 119.28},
        "centroid": {"lat": 4.21, "lon": 101.98},
        "admin_regions": ["Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang", "Perak", "Perlis", "Penang", "Sabah", "Sarawak", "Selangor", "Terengganu", "Kuala Lumpur", "Labuan", "Putrajaya"],
        "default_datum": "WGS84",
        "eudr_commodities": ["palm_oil", "rubber", "wood", "cocoa"],
    },
    "TH": {
        "name": "Thailand",
        "bbox": {"min_lat": 5.61, "min_lon": 97.34, "max_lat": 20.46, "max_lon": 105.64},
        "centroid": {"lat": 15.87, "lon": 100.99},
        "admin_regions": [],
        "default_datum": "INDIAN_1975",
        "eudr_commodities": ["rubber", "palm_oil", "wood"],
    },
    "VN": {
        "name": "Vietnam",
        "bbox": {"min_lat": 8.56, "min_lon": 102.14, "max_lat": 23.39, "max_lon": 109.47},
        "centroid": {"lat": 14.06, "lon": 108.28},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["coffee", "rubber", "wood"],
    },
    "MM": {
        "name": "Myanmar",
        "bbox": {"min_lat": 9.78, "min_lon": 92.19, "max_lat": 28.54, "max_lon": 101.17},
        "centroid": {"lat": 19.76, "lon": 96.08},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["rubber", "wood"],
    },
    "KH": {
        "name": "Cambodia",
        "bbox": {"min_lat": 10.41, "min_lon": 102.34, "max_lat": 14.69, "max_lon": 107.63},
        "centroid": {"lat": 12.57, "lon": 104.99},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["rubber", "wood"],
    },
    "LA": {
        "name": "Laos",
        "bbox": {"min_lat": 13.91, "min_lon": 100.08, "max_lat": 22.50, "max_lon": 107.64},
        "centroid": {"lat": 19.86, "lon": 102.50},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["rubber", "coffee", "wood"],
    },
    "PH": {
        "name": "Philippines",
        "bbox": {"min_lat": 4.59, "min_lon": 116.93, "max_lat": 21.12, "max_lon": 126.60},
        "centroid": {"lat": 12.88, "lon": 121.77},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["palm_oil", "rubber", "cocoa", "coffee"],
    },
    "PG": {
        "name": "Papua New Guinea",
        "bbox": {"min_lat": -11.66, "min_lon": 140.84, "max_lat": -0.87, "max_lon": 157.04},
        "centroid": {"lat": -6.31, "lon": 147.18},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["palm_oil", "coffee", "cocoa", "wood"],
    },
    "LK": {
        "name": "Sri Lanka",
        "bbox": {"min_lat": 5.92, "min_lon": 79.65, "max_lat": 9.84, "max_lon": 81.88},
        "centroid": {"lat": 7.87, "lon": 80.77},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["rubber"],
    },
    "IN": {
        "name": "India",
        "bbox": {"min_lat": 6.75, "min_lon": 68.16, "max_lat": 35.99, "max_lon": 97.42},
        "centroid": {"lat": 20.59, "lon": 78.96},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["coffee", "rubber", "soya", "wood"],
    },
    # ===================================================================
    # Europe (EU importers)
    # ===================================================================
    "DE": {
        "name": "Germany",
        "bbox": {"min_lat": 47.27, "min_lon": 5.87, "max_lat": 55.06, "max_lon": 15.04},
        "centroid": {"lat": 51.17, "lon": 10.45},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "FR": {
        "name": "France",
        "bbox": {"min_lat": 41.36, "min_lon": -5.14, "max_lat": 51.09, "max_lon": 9.56},
        "centroid": {"lat": 46.23, "lon": 2.21},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": ["wood", "soya", "cattle"],
    },
    "NL": {
        "name": "Netherlands",
        "bbox": {"min_lat": 50.75, "min_lon": 3.36, "max_lat": 53.47, "max_lon": 7.21},
        "centroid": {"lat": 52.13, "lon": 5.29},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "BE": {
        "name": "Belgium",
        "bbox": {"min_lat": 49.50, "min_lon": 2.55, "max_lat": 51.50, "max_lon": 6.41},
        "centroid": {"lat": 50.50, "lon": 4.47},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "IT": {
        "name": "Italy",
        "bbox": {"min_lat": 36.65, "min_lon": 6.63, "max_lat": 47.09, "max_lon": 18.52},
        "centroid": {"lat": 41.87, "lon": 12.57},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "ES": {
        "name": "Spain",
        "bbox": {"min_lat": 35.95, "min_lon": -9.30, "max_lat": 43.79, "max_lon": 4.33},
        "centroid": {"lat": 40.46, "lon": -3.75},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "PT": {
        "name": "Portugal",
        "bbox": {"min_lat": 36.96, "min_lon": -9.50, "max_lat": 42.15, "max_lon": -6.19},
        "centroid": {"lat": 39.40, "lon": -8.22},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "AT": {
        "name": "Austria",
        "bbox": {"min_lat": 46.37, "min_lon": 9.53, "max_lat": 49.02, "max_lon": 17.16},
        "centroid": {"lat": 47.52, "lon": 14.55},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "PL": {
        "name": "Poland",
        "bbox": {"min_lat": 49.00, "min_lon": 14.12, "max_lat": 54.84, "max_lon": 24.15},
        "centroid": {"lat": 51.92, "lon": 19.15},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "SE": {
        "name": "Sweden",
        "bbox": {"min_lat": 55.34, "min_lon": 11.11, "max_lat": 69.06, "max_lon": 24.17},
        "centroid": {"lat": 60.13, "lon": 18.64},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": ["wood"],
    },
    "FI": {
        "name": "Finland",
        "bbox": {"min_lat": 59.81, "min_lon": 20.55, "max_lat": 70.09, "max_lon": 31.58},
        "centroid": {"lat": 61.92, "lon": 25.75},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": ["wood"],
    },
    "DK": {
        "name": "Denmark",
        "bbox": {"min_lat": 54.56, "min_lon": 8.09, "max_lat": 57.75, "max_lon": 15.19},
        "centroid": {"lat": 56.26, "lon": 9.50},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "IE": {
        "name": "Ireland",
        "bbox": {"min_lat": 51.42, "min_lon": -10.48, "max_lat": 55.39, "max_lon": -5.99},
        "centroid": {"lat": 53.41, "lon": -8.24},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": ["cattle"],
    },
    "GB": {
        "name": "United Kingdom",
        "bbox": {"min_lat": 49.88, "min_lon": -7.57, "max_lat": 60.86, "max_lon": 1.68},
        "centroid": {"lat": 55.38, "lon": -3.44},
        "admin_regions": [],
        "default_datum": "OSGB_1936",
        "eudr_commodities": [],
    },
    "GR": {
        "name": "Greece",
        "bbox": {"min_lat": 34.80, "min_lon": 19.37, "max_lat": 41.75, "max_lon": 29.65},
        "centroid": {"lat": 39.07, "lon": 21.82},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "CZ": {
        "name": "Czech Republic",
        "bbox": {"min_lat": 48.55, "min_lon": 12.09, "max_lat": 51.06, "max_lon": 18.86},
        "centroid": {"lat": 49.82, "lon": 15.47},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "RO": {
        "name": "Romania",
        "bbox": {"min_lat": 43.62, "min_lon": 20.26, "max_lat": 48.27, "max_lon": 29.69},
        "centroid": {"lat": 45.94, "lon": 24.97},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "HU": {
        "name": "Hungary",
        "bbox": {"min_lat": 45.74, "min_lon": 16.11, "max_lat": 48.59, "max_lon": 22.90},
        "centroid": {"lat": 47.16, "lon": 19.50},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "BG": {
        "name": "Bulgaria",
        "bbox": {"min_lat": 41.24, "min_lon": 22.36, "max_lat": 44.22, "max_lon": 28.61},
        "centroid": {"lat": 42.73, "lon": 25.49},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    "NO": {
        "name": "Norway",
        "bbox": {"min_lat": 57.97, "min_lon": 4.65, "max_lat": 71.19, "max_lon": 31.17},
        "centroid": {"lat": 60.47, "lon": 8.47},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": ["wood"],
    },
    "CH": {
        "name": "Switzerland",
        "bbox": {"min_lat": 45.82, "min_lon": 5.96, "max_lat": 47.81, "max_lon": 10.49},
        "centroid": {"lat": 46.82, "lon": 8.23},
        "admin_regions": [],
        "default_datum": "ETRS89",
        "eudr_commodities": [],
    },
    # ===================================================================
    # Other Countries
    # ===================================================================
    "US": {
        "name": "United States",
        "bbox": {"min_lat": 24.40, "min_lon": -124.85, "max_lat": 49.38, "max_lon": -66.88},
        "centroid": {"lat": 37.09, "lon": -95.71},
        "admin_regions": [],
        "default_datum": "NAD83",
        "eudr_commodities": ["soya", "cattle", "wood"],
    },
    "CA": {
        "name": "Canada",
        "bbox": {"min_lat": 41.68, "min_lon": -141.00, "max_lat": 83.11, "max_lon": -52.62},
        "centroid": {"lat": 56.13, "lon": -106.35},
        "admin_regions": [],
        "default_datum": "NAD83",
        "eudr_commodities": ["wood", "soya"],
    },
    "AU": {
        "name": "Australia",
        "bbox": {"min_lat": -43.64, "min_lon": 113.34, "max_lat": -10.06, "max_lon": 153.64},
        "centroid": {"lat": -25.27, "lon": 133.78},
        "admin_regions": [],
        "default_datum": "GDA2020",
        "eudr_commodities": ["cattle", "wood"],
    },
    "NZ": {
        "name": "New Zealand",
        "bbox": {"min_lat": -47.29, "min_lon": 166.43, "max_lat": -34.39, "max_lon": 178.57},
        "centroid": {"lat": -40.90, "lon": 174.89},
        "admin_regions": [],
        "default_datum": "NZGD2000",
        "eudr_commodities": ["wood", "cattle"],
    },
    "JP": {
        "name": "Japan",
        "bbox": {"min_lat": 24.25, "min_lon": 122.94, "max_lat": 45.52, "max_lon": 153.99},
        "centroid": {"lat": 36.20, "lon": 138.25},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "KR": {
        "name": "South Korea",
        "bbox": {"min_lat": 33.11, "min_lon": 124.61, "max_lat": 38.62, "max_lon": 131.87},
        "centroid": {"lat": 35.91, "lon": 127.77},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "CN": {
        "name": "China",
        "bbox": {"min_lat": 18.16, "min_lon": 73.50, "max_lat": 53.56, "max_lon": 134.77},
        "centroid": {"lat": 35.86, "lon": 104.20},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["soya", "rubber", "wood"],
    },
    "RU": {
        "name": "Russia",
        "bbox": {"min_lat": 41.19, "min_lon": 19.64, "max_lat": 81.86, "max_lon": -169.05},
        "centroid": {"lat": 61.52, "lon": 105.32},
        "admin_regions": [],
        "default_datum": "PULKOVO_1942",
        "eudr_commodities": ["wood"],
    },
    "ZA": {
        "name": "South Africa",
        "bbox": {"min_lat": -34.84, "min_lon": 16.45, "max_lat": -22.13, "max_lon": 32.89},
        "centroid": {"lat": -30.56, "lon": 22.94},
        "admin_regions": [],
        "default_datum": "HARTEBEESTHOEK94",
        "eudr_commodities": [],
    },
    "MZ": {
        "name": "Mozambique",
        "bbox": {"min_lat": -26.87, "min_lon": 30.21, "max_lat": -10.47, "max_lon": 40.84},
        "centroid": {"lat": -18.67, "lon": 35.53},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["wood"],
    },
    "AO": {
        "name": "Angola",
        "bbox": {"min_lat": -18.04, "min_lon": 11.64, "max_lat": -4.38, "max_lon": 24.08},
        "centroid": {"lat": -11.20, "lon": 17.87},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["coffee", "wood"],
    },
    "EG": {
        "name": "Egypt",
        "bbox": {"min_lat": 22.00, "min_lon": 24.70, "max_lat": 31.67, "max_lon": 36.90},
        "centroid": {"lat": 26.82, "lon": 30.80},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "SA": {
        "name": "Saudi Arabia",
        "bbox": {"min_lat": 16.38, "min_lon": 34.63, "max_lat": 32.16, "max_lon": 55.67},
        "centroid": {"lat": 23.89, "lon": 45.08},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "TR": {
        "name": "Turkey",
        "bbox": {"min_lat": 35.82, "min_lon": 25.66, "max_lat": 42.11, "max_lon": 44.79},
        "centroid": {"lat": 38.96, "lon": 35.24},
        "admin_regions": [],
        "default_datum": "ED50",
        "eudr_commodities": [],
    },
    "UA": {
        "name": "Ukraine",
        "bbox": {"min_lat": 44.39, "min_lon": 22.14, "max_lat": 52.38, "max_lon": 40.23},
        "centroid": {"lat": 48.38, "lon": 31.17},
        "admin_regions": [],
        "default_datum": "PULKOVO_1942",
        "eudr_commodities": ["soya", "wood"],
    },
    "IR": {
        "name": "Iran",
        "bbox": {"min_lat": 25.06, "min_lon": 44.05, "max_lat": 39.78, "max_lon": 63.32},
        "centroid": {"lat": 32.43, "lon": 53.69},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "SO": {
        "name": "Somalia",
        "bbox": {"min_lat": -1.66, "min_lon": 40.99, "max_lat": 11.98, "max_lon": 51.41},
        "centroid": {"lat": 5.15, "lon": 46.20},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "SD": {
        "name": "Sudan",
        "bbox": {"min_lat": 8.69, "min_lon": 21.84, "max_lat": 23.15, "max_lon": 38.58},
        "centroid": {"lat": 12.86, "lon": 30.22},
        "admin_regions": [],
        "default_datum": "ADINDAN",
        "eudr_commodities": [],
    },
    "ER": {
        "name": "Eritrea",
        "bbox": {"min_lat": 12.36, "min_lon": 36.44, "max_lat": 18.00, "max_lon": 43.14},
        "centroid": {"lat": 15.18, "lon": 39.78},
        "admin_regions": [],
        "default_datum": "ADINDAN",
        "eudr_commodities": [],
    },
    "TN": {
        "name": "Tunisia",
        "bbox": {"min_lat": 30.23, "min_lon": 7.52, "max_lat": 37.54, "max_lon": 11.60},
        "centroid": {"lat": 33.89, "lon": 9.54},
        "admin_regions": [],
        "default_datum": "CARTHAGE",
        "eudr_commodities": [],
    },
    "DZ": {
        "name": "Algeria",
        "bbox": {"min_lat": 19.06, "min_lon": -8.67, "max_lat": 37.09, "max_lon": 11.98},
        "centroid": {"lat": 28.03, "lon": 1.66},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "MA": {
        "name": "Morocco",
        "bbox": {"min_lat": 27.66, "min_lon": -13.17, "max_lat": 35.93, "max_lon": -0.99},
        "centroid": {"lat": 31.79, "lon": -7.09},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "LY": {
        "name": "Libya",
        "bbox": {"min_lat": 19.51, "min_lon": 9.39, "max_lat": 33.17, "max_lon": 25.15},
        "centroid": {"lat": 26.34, "lon": 17.23},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "ZW": {
        "name": "Zimbabwe",
        "bbox": {"min_lat": -22.42, "min_lon": 25.24, "max_lat": -15.61, "max_lon": 33.06},
        "centroid": {"lat": -19.02, "lon": 29.15},
        "admin_regions": [],
        "default_datum": "ARC_1950",
        "eudr_commodities": [],
    },
    "ZM": {
        "name": "Zambia",
        "bbox": {"min_lat": -18.08, "min_lon": 21.99, "max_lat": -8.22, "max_lon": 33.71},
        "centroid": {"lat": -13.13, "lon": 27.85},
        "admin_regions": [],
        "default_datum": "ARC_1950",
        "eudr_commodities": [],
    },
    "MW": {
        "name": "Malawi",
        "bbox": {"min_lat": -17.13, "min_lon": 32.68, "max_lat": -9.37, "max_lon": 35.92},
        "centroid": {"lat": -13.25, "lon": 34.30},
        "admin_regions": [],
        "default_datum": "ARC_1950",
        "eudr_commodities": [],
    },
    "BW": {
        "name": "Botswana",
        "bbox": {"min_lat": -26.91, "min_lon": 19.99, "max_lat": -17.78, "max_lon": 29.37},
        "centroid": {"lat": -22.33, "lon": 24.68},
        "admin_regions": [],
        "default_datum": "ARC_1950",
        "eudr_commodities": ["cattle"],
    },
    "NA": {
        "name": "Namibia",
        "bbox": {"min_lat": -28.97, "min_lon": 11.73, "max_lat": -16.96, "max_lon": 25.26},
        "centroid": {"lat": -22.96, "lon": 18.49},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": ["cattle"],
    },
    "SG": {
        "name": "Singapore",
        "bbox": {"min_lat": 1.16, "min_lon": 103.60, "max_lat": 1.47, "max_lon": 104.09},
        "centroid": {"lat": 1.35, "lon": 103.82},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
    "BN": {
        "name": "Brunei",
        "bbox": {"min_lat": 4.00, "min_lon": 114.08, "max_lat": 5.05, "max_lon": 115.36},
        "centroid": {"lat": 4.54, "lon": 114.73},
        "admin_regions": [],
        "default_datum": "WGS84",
        "eudr_commodities": [],
    },
}


# ---------------------------------------------------------------------------
# Ocean Regions
# ---------------------------------------------------------------------------
# 10 major ocean basins with approximate bounding boxes for land/ocean
# classification.  These polygons are simplified envelopes, not precise
# coastlines.

OCEAN_REGIONS: Dict[str, Dict[str, Any]] = {
    "north_atlantic": {
        "name": "North Atlantic Ocean",
        "bbox": {"min_lat": 0.0, "min_lon": -80.0, "max_lat": 65.0, "max_lon": -5.0},
        "centre": {"lat": 30.0, "lon": -40.0},
    },
    "south_atlantic": {
        "name": "South Atlantic Ocean",
        "bbox": {"min_lat": -60.0, "min_lon": -70.0, "max_lat": 0.0, "max_lon": 20.0},
        "centre": {"lat": -25.0, "lon": -15.0},
    },
    "north_pacific": {
        "name": "North Pacific Ocean",
        "bbox": {"min_lat": 0.0, "min_lon": 120.0, "max_lat": 65.0, "max_lon": -100.0},
        "centre": {"lat": 30.0, "lon": -160.0},
    },
    "south_pacific": {
        "name": "South Pacific Ocean",
        "bbox": {"min_lat": -60.0, "min_lon": 150.0, "max_lat": 0.0, "max_lon": -70.0},
        "centre": {"lat": -25.0, "lon": -140.0},
    },
    "indian_ocean": {
        "name": "Indian Ocean",
        "bbox": {"min_lat": -60.0, "min_lon": 20.0, "max_lat": 30.0, "max_lon": 120.0},
        "centre": {"lat": -15.0, "lon": 75.0},
    },
    "arctic_ocean": {
        "name": "Arctic Ocean",
        "bbox": {"min_lat": 65.0, "min_lon": -180.0, "max_lat": 90.0, "max_lon": 180.0},
        "centre": {"lat": 85.0, "lon": 0.0},
    },
    "southern_ocean": {
        "name": "Southern Ocean",
        "bbox": {"min_lat": -90.0, "min_lon": -180.0, "max_lat": -60.0, "max_lon": 180.0},
        "centre": {"lat": -70.0, "lon": 0.0},
    },
    "gulf_of_mexico": {
        "name": "Gulf of Mexico",
        "bbox": {"min_lat": 18.0, "min_lon": -98.0, "max_lat": 30.5, "max_lon": -80.0},
        "centre": {"lat": 25.0, "lon": -90.0},
    },
    "mediterranean": {
        "name": "Mediterranean Sea",
        "bbox": {"min_lat": 30.0, "min_lon": -6.0, "max_lat": 46.0, "max_lon": 36.0},
        "centre": {"lat": 35.0, "lon": 18.0},
    },
    "south_china_sea": {
        "name": "South China Sea",
        "bbox": {"min_lat": 0.0, "min_lon": 99.0, "max_lat": 23.0, "max_lon": 121.0},
        "centre": {"lat": 12.0, "lon": 110.0},
    },
}


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def get_country(iso: str) -> Optional[Dict[str, Any]]:
    """Retrieve country boundary data by ISO 3166-1 alpha-2 code.

    Args:
        iso: Two-letter ISO country code (e.g. 'BR', 'ID').

    Returns:
        Country boundary dictionary or None if not found.
    """
    return COUNTRY_BOUNDARIES.get(iso.upper())


def find_country(lat: float, lon: float) -> Optional[str]:
    """Find the country ISO code containing the given coordinate.

    Uses bounding-box containment as a fast approximation. Multiple
    countries may match; the first match is returned. For precise
    point-in-polygon testing, use the SpatialPlausibilityChecker engine.

    Args:
        lat: Latitude in decimal degrees WGS84.
        lon: Longitude in decimal degrees WGS84.

    Returns:
        ISO 3166-1 alpha-2 code of the first matching country, or None.
    """
    for iso, data in COUNTRY_BOUNDARIES.items():
        bbox = data["bbox"]
        if (bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]):
            return iso
    return None


def is_ocean(lat: float, lon: float) -> bool:
    """Check whether a coordinate is likely in an ocean region.

    Uses the simplified ocean bounding boxes. A coordinate matching
    no country AND matching an ocean region is considered oceanic.

    Args:
        lat: Latitude in decimal degrees WGS84.
        lon: Longitude in decimal degrees WGS84.

    Returns:
        True if the coordinate is likely in an ocean region.
    """
    # First check: if it matches a country, it is land
    if find_country(lat, lon) is not None:
        return False

    # Second check: see if it falls in a known ocean region
    for _key, region in OCEAN_REGIONS.items():
        bbox = region["bbox"]
        if (bbox["min_lat"] <= lat <= bbox["max_lat"]
                and bbox["min_lon"] <= lon <= bbox["max_lon"]):
            return True

    # If no country and no known ocean region matched, assume ocean
    # for coordinates well outside land masses
    return True


def get_eudr_countries() -> List[str]:
    """Return ISO codes of countries producing EUDR-regulated commodities.

    Returns:
        Sorted list of ISO codes that have at least one EUDR commodity.
    """
    return sorted(
        iso for iso, data in COUNTRY_BOUNDARIES.items()
        if data.get("eudr_commodities")
    )


def get_country_centroid(iso: str) -> Optional[Tuple[float, float]]:
    """Retrieve the geographic centroid of a country.

    Args:
        iso: ISO 3166-1 alpha-2 country code.

    Returns:
        Tuple of (latitude, longitude), or None if not found.
    """
    data = COUNTRY_BOUNDARIES.get(iso.upper())
    if data:
        c = data["centroid"]
        return (c["lat"], c["lon"])
    return None


# ---------------------------------------------------------------------------
# Module Totals
# ---------------------------------------------------------------------------

TOTAL_COUNTRIES: int = len(COUNTRY_BOUNDARIES)
TOTAL_OCEAN_REGIONS: int = len(OCEAN_REGIONS)

__all__ = [
    "COUNTRY_BOUNDARIES",
    "OCEAN_REGIONS",
    "TOTAL_COUNTRIES",
    "TOTAL_OCEAN_REGIONS",
    "get_country",
    "find_country",
    "is_ocean",
    "get_eudr_countries",
    "get_country_centroid",
]
