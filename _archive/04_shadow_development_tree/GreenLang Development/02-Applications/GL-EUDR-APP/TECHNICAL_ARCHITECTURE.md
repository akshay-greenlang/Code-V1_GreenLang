# GL-EUDR-APP: Technical Architecture Document
## 5-Agent Pipeline for EU Deforestation Regulation Compliance

---

## 1. SYSTEM OVERVIEW

### 1.1 Architecture Principles

- **Microservices**: Each agent is an independent service
- **Event-Driven**: Async communication via message queues
- **Cloud-Native**: Containerized, Kubernetes-orchestrated
- **Scalable**: Horizontal scaling for each agent
- **Resilient**: Circuit breakers, retry logic, failover
- **Observable**: Distributed tracing, metrics, logging

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        External Systems                         │
├────────────┬────────────┬────────────┬────────────┬───────────┤
│   ERP      │  Satellite │    LLM     │  EU Portal │  Users    │
│  Systems   │    APIs    │    APIs    │    API     │  (Web)    │
└─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┴─────┬─────┘
      │            │            │            │            │
      ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API Gateway (Kong/AWS API GW)                │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    5-Agent Processing Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Agent 1  │→ │ Agent 2  │→ │ Agent 3  │→ │ Agent 4  │      │
│  │ Supplier │  │   Geo    │  │  Defor.  │  │   Doc    │      │
│  │  Intake  │  │  Valid.  │  │   Risk   │  │  Verif.  │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                    │                            │
│                                    ▼                            │
│                            ┌──────────────┐                    │
│                            │   Agent 5    │                    │
│                            │ DDS Reporting│                    │
│                            └──────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                              │
├─────────────┬─────────────┬─────────────┬──────────────────────┤
│ PostgreSQL  │   PostGIS   │   Redis     │   S3/Blob Storage    │
│ TimescaleDB │   Spatial   │   Cache     │   Documents/Images   │
└─────────────┴─────────────┴─────────────┴──────────────────────┘
```

---

## 2. AGENT SPECIFICATIONS

### 2.1 Agent 1: SupplierDataIntakeAgent

**Purpose**: Ingest and normalize supplier data from multiple sources

**Technology Stack**:
- Runtime: Python 3.11
- Framework: FastAPI
- Integration: Apache NiFi
- Queue: Apache Kafka
- Database: PostgreSQL

**API Endpoints**:
```python
# supplier_intake_agent.py
from fastapi import FastAPI, BackgroundTasks
from typing import List, Dict
import asyncio

app = FastAPI(title="Supplier Data Intake Agent")

@app.post("/api/v1/intake/erp/sync")
async def sync_erp_data(
    erp_config: ERPConfig,
    background_tasks: BackgroundTasks
):
    """
    Initiate ERP data synchronization

    Request:
    {
        "erp_type": "SAP_S4HANA",
        "connection": {
            "host": "sap.company.com",
            "client": "100",
            "credentials": "encrypted_token"
        },
        "sync_options": {
            "full_sync": false,
            "from_date": "2024-01-01",
            "commodities": ["COCOA", "COFFEE"]
        }
    }
    """
    background_tasks.add_task(sync_erp_async, erp_config)
    return {"status": "sync_initiated", "job_id": generate_job_id()}

@app.post("/api/v1/intake/manual/upload")
async def upload_supplier_data(
    file: UploadFile,
    format: str = "csv"
):
    """
    Manual supplier data upload (CSV/Excel)
    """
    data = await parse_file(file, format)
    validated = await validate_supplier_data(data)
    await queue_for_processing(validated)
    return {"records_processed": len(validated)}

async def sync_erp_async(config: ERPConfig):
    """
    Background ERP synchronization
    """
    connector = ERPConnectorFactory.create(config.erp_type)

    # Extract data in batches
    async for batch in connector.extract_batches():
        # Transform to standard format
        normalized = transform_to_eudr_format(batch)

        # Validate data quality
        validated = validate_batch(normalized)

        # Send to next agent via Kafka
        await kafka_producer.send(
            "supplier-data-topic",
            validated
        )

        # Update sync status
        await update_sync_status(config.job_id, batch.last_record_id)
```

**Data Schema**:
```sql
-- Supplier Master Data
CREATE TABLE suppliers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(100) UNIQUE,
    name VARCHAR(500) NOT NULL,
    tax_id VARCHAR(50),
    country_code CHAR(2) NOT NULL,
    address JSONB,
    commodities TEXT[],
    risk_category VARCHAR(20) DEFAULT 'STANDARD',
    erp_source VARCHAR(50),
    last_sync TIMESTAMP,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Procurement Data
CREATE TABLE procurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID REFERENCES suppliers(id),
    po_number VARCHAR(100) UNIQUE,
    commodity_type VARCHAR(50) NOT NULL,
    quantity DECIMAL(15,3),
    unit VARCHAR(20),
    harvest_date DATE,
    shipment_date DATE,
    origin_plot_ids UUID[],
    documents JSONB,
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### 2.2 Agent 2: GeoValidationAgent

**Purpose**: Deterministic validation of geographic coordinates

**Technology Stack**:
- Runtime: Python 3.11
- GIS: PostGIS, GDAL, Shapely
- Validation: Custom algorithms
- Cache: Redis

**Core Functions**:
```python
# geo_validation_agent.py
from shapely.geometry import Point, Polygon
import geopandas as gpd
from typing import Tuple, List

class GeoValidationAgent:
    """
    Zero-hallucination geographic validation
    All operations are deterministic
    """

    def __init__(self):
        self.country_boundaries = self._load_country_boundaries()
        self.water_bodies = self._load_water_bodies()
        self.protected_areas = self._load_protected_areas()

    def validate_coordinates(
        self,
        lat: float,
        lon: float,
        country_code: str
    ) -> ValidationResult:
        """
        Deterministic coordinate validation

        Rules:
        1. Must be 6+ decimal places
        2. Must be within valid lat/lon range
        3. Must be within claimed country
        4. Must not be in water body
        5. Must not overlap protected areas
        """
        errors = []
        warnings = []

        # Rule 1: Precision check
        if not self._check_precision(lat, 6) or not self._check_precision(lon, 6):
            errors.append("INSUFFICIENT_PRECISION")

        # Rule 2: Range validation
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            errors.append("INVALID_COORDINATES")
            return ValidationResult(valid=False, errors=errors)

        # Rule 3: Country boundary check
        point = Point(lon, lat)
        if not self._point_in_country(point, country_code):
            errors.append(f"NOT_IN_COUNTRY_{country_code}")

        # Rule 4: Water body check
        if self._point_in_water(point):
            errors.append("LOCATION_IN_WATER")

        # Rule 5: Protected area check
        if self._point_in_protected_area(point):
            warnings.append("IN_PROTECTED_AREA")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "precision_lat": self._get_precision(lat),
                "precision_lon": self._get_precision(lon),
                "country_verified": country_code
            }
        )

    def validate_polygon(
        self,
        coordinates: List[Tuple[float, float]],
        min_area_hectares: float = 0.01
    ) -> ValidationResult:
        """
        Validate polygon for plot boundaries

        Rules:
        1. Minimum 3 points for valid polygon
        2. No self-intersections
        3. Area above minimum threshold
        4. All points follow coordinate rules
        """
        if len(coordinates) < 3:
            return ValidationResult(
                valid=False,
                errors=["INSUFFICIENT_POINTS"]
            )

        # Create polygon
        polygon = Polygon(coordinates)

        # Check validity
        if not polygon.is_valid:
            return ValidationResult(
                valid=False,
                errors=["INVALID_POLYGON", explain_validity(polygon)]
            )

        # Calculate area
        area_m2 = self._calculate_area(polygon)
        area_hectares = area_m2 / 10000

        if area_hectares < min_area_hectares:
            return ValidationResult(
                valid=False,
                errors=[f"AREA_TOO_SMALL: {area_hectares:.4f} ha"]
            )

        # Validate each point
        for lat, lon in coordinates:
            point_result = self.validate_coordinates(lat, lon, country_code)
            if not point_result.valid:
                return point_result

        return ValidationResult(
            valid=True,
            metadata={
                "area_hectares": area_hectares,
                "perimeter_km": polygon.length * 111,  # Rough conversion
                "centroid": (polygon.centroid.y, polygon.centroid.x)
            }
        )

    def detect_overlaps(
        self,
        new_plot: Polygon,
        existing_plots: List[Polygon]
    ) -> List[OverlapResult]:
        """
        Detect overlapping plots (potential double-claiming)
        """
        overlaps = []

        for existing in existing_plots:
            if new_plot.intersects(existing):
                intersection = new_plot.intersection(existing)
                overlap_area = self._calculate_area(intersection)
                overlap_percent = (overlap_area / self._calculate_area(new_plot)) * 100

                overlaps.append(OverlapResult(
                    plot_id=existing.plot_id,
                    overlap_area_hectares=overlap_area / 10000,
                    overlap_percentage=overlap_percent,
                    severity="HIGH" if overlap_percent > 10 else "MEDIUM"
                ))

        return overlaps

    @staticmethod
    def _check_precision(value: float, min_decimals: int) -> bool:
        """Check if coordinate has minimum decimal precision"""
        str_value = f"{value:.10f}"
        decimal_part = str_value.split('.')[1]
        # Count non-zero decimal places
        significant_decimals = len(decimal_part.rstrip('0'))
        return significant_decimals >= min_decimals

    def _calculate_area(self, polygon: Polygon) -> float:
        """Calculate accurate area using geographic projection"""
        # Convert to appropriate UTM zone for accurate area calculation
        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
        utm_crs = gdf.estimate_utm_crs()
        gdf_projected = gdf.to_crs(utm_crs)
        return gdf_projected.geometry[0].area
```

**Validation Database**:
```sql
-- Validated plot geometry
CREATE TABLE validated_plots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supplier_id UUID REFERENCES suppliers(id),
    plot_identifier VARCHAR(255) UNIQUE,
    geometry GEOMETRY(GEOMETRY, 4326) NOT NULL,
    area_hectares DECIMAL(10,2),
    country_code CHAR(2) NOT NULL,
    commodity_type VARCHAR(50),
    validation_status VARCHAR(50) NOT NULL,
    validation_errors JSONB,
    validation_warnings JSONB,
    overlap_conflicts JSONB,
    validated_at TIMESTAMP DEFAULT NOW(),

    -- Spatial indexes
    CONSTRAINT valid_geometry CHECK (ST_IsValid(geometry))
);

CREATE INDEX idx_validated_plots_geometry ON validated_plots USING GIST (geometry);
CREATE INDEX idx_validated_plots_country ON validated_plots (country_code);
CREATE INDEX idx_validated_plots_status ON validated_plots (validation_status);

-- Spatial queries for overlap detection
CREATE OR REPLACE FUNCTION find_overlapping_plots(
    new_geometry GEOMETRY,
    commodity VARCHAR(50)
) RETURNS TABLE (
    plot_id UUID,
    overlap_area_m2 FLOAT,
    overlap_percentage FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vp.id,
        ST_Area(ST_Intersection(vp.geometry, new_geometry)::geography),
        (ST_Area(ST_Intersection(vp.geometry, new_geometry)::geography) /
         ST_Area(new_geometry::geography)) * 100
    FROM validated_plots vp
    WHERE ST_Intersects(vp.geometry, new_geometry)
      AND vp.commodity_type = commodity
      AND vp.validation_status = 'VALIDATED';
END;
$$ LANGUAGE plpgsql;
```

---

### 2.3 Agent 3: DeforestationRiskAgent

**Purpose**: AI/ML-based satellite imagery analysis for deforestation detection

**Technology Stack**:
- Runtime: Python 3.11
- ML Framework: TensorFlow 2.14, PyTorch 2.0
- Image Processing: GDAL, Rasterio
- Satellite APIs: Sentinel Hub, Google Earth Engine
- GPU: NVIDIA V100/A100

**ML Architecture**:
```python
# deforestation_risk_agent.py
import tensorflow as tf
import numpy as np
from sentinel2 import SentinelHubAPI
import rasterio

class DeforestationRiskAgent:
    """
    Satellite-based deforestation detection using ensemble ML
    """

    def __init__(self):
        self.models = self._load_ensemble_models()
        self.sentinel_api = SentinelHubAPI()
        self.baseline_year = 2020

    def assess_deforestation_risk(
        self,
        plot_geometry: Polygon,
        assessment_date: date
    ) -> RiskAssessment:
        """
        Main risk assessment pipeline
        """
        # Step 1: Retrieve satellite imagery
        imagery = self._fetch_satellite_imagery(
            plot_geometry,
            date_range=(self.baseline_year, assessment_date)
        )

        # Step 2: Preprocess imagery
        processed = self._preprocess_imagery(imagery)

        # Step 3: Run ensemble models
        predictions = self._run_ensemble(processed)

        # Step 4: Calculate risk score
        risk_score = self._calculate_risk_score(predictions)

        # Step 5: Generate evidence
        evidence = self._generate_evidence(
            imagery,
            predictions,
            plot_geometry
        )

        return RiskAssessment(
            plot_id=plot_geometry.id,
            risk_score=risk_score,
            risk_level=self._classify_risk(risk_score),
            deforestation_detected=risk_score > 0.7,
            evidence=evidence,
            confidence=self._calculate_confidence(predictions),
            assessment_date=datetime.now()
        )

    def _fetch_satellite_imagery(
        self,
        geometry: Polygon,
        date_range: Tuple[int, date]
    ) -> SatelliteImagery:
        """
        Fetch multi-temporal satellite imagery
        """
        bbox = geometry.bounds  # (minx, miny, maxx, maxy)

        # Sentinel-2 configuration
        config = {
            'bbox': bbox,
            'time_range': date_range,
            'bands': ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'],  # RGB + NIR + SWIR
            'resolution': 10,  # meters
            'cloud_coverage': 20  # max %
        }

        # Fetch baseline (2020)
        baseline = self.sentinel_api.get_imagery(
            config,
            date=f"{self.baseline_year}-12-31"
        )

        # Fetch current
        current = self.sentinel_api.get_imagery(
            config,
            date=assessment_date
        )

        # Calculate vegetation indices
        baseline_ndvi = self._calculate_ndvi(baseline)
        current_ndvi = self._calculate_ndvi(current)

        return SatelliteImagery(
            baseline=baseline,
            current=current,
            baseline_ndvi=baseline_ndvi,
            current_ndvi=current_ndvi,
            bbox=bbox,
            crs='EPSG:4326'
        )

    def _calculate_ndvi(self, imagery: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index
        NDVI = (NIR - Red) / (NIR + Red)
        """
        nir = imagery[:, :, 3]  # Band 8
        red = imagery[:, :, 2]  # Band 4

        ndvi = np.where(
            (nir + red) == 0,
            0,
            (nir - red) / (nir + red)
        )

        return ndvi

    def _run_ensemble(self, processed_imagery: ProcessedImagery) -> List[Prediction]:
        """
        Run ensemble of models for robust detection
        """
        predictions = []

        # Model 1: U-Net for semantic segmentation
        unet_pred = self.models['unet'].predict(
            processed_imagery.stacked_bands
        )
        predictions.append(Prediction(
            model='unet',
            forest_mask=unet_pred,
            confidence=self._calculate_mask_confidence(unet_pred)
        ))

        # Model 2: Change Detection CNN
        change_pred = self.models['change_detection'].predict([
            processed_imagery.baseline,
            processed_imagery.current
        ])
        predictions.append(Prediction(
            model='change_detection',
            change_mask=change_pred,
            confidence=np.mean(change_pred)
        ))

        # Model 3: Random Forest on vegetation indices
        rf_features = np.stack([
            processed_imagery.ndvi_change,
            processed_imagery.evi_change,
            processed_imagery.savi_change
        ], axis=-1)

        rf_pred = self.models['random_forest'].predict(
            rf_features.reshape(-1, 3)
        ).reshape(rf_features.shape[:2])

        predictions.append(Prediction(
            model='random_forest',
            deforestation_probability=rf_pred,
            confidence=np.mean(rf_pred > 0.5)
        ))

        # Model 4: Vision Transformer for patch classification
        patches = self._extract_patches(processed_imagery.current)
        vit_pred = self.models['vision_transformer'].predict(patches)

        predictions.append(Prediction(
            model='vision_transformer',
            patch_classifications=vit_pred,
            confidence=np.mean(np.max(vit_pred, axis=1))
        ))

        return predictions

    def _calculate_risk_score(self, predictions: List[Prediction]) -> float:
        """
        Weighted ensemble voting for final risk score
        """
        weights = {
            'unet': 0.3,
            'change_detection': 0.35,
            'random_forest': 0.2,
            'vision_transformer': 0.15
        }

        weighted_score = 0
        total_weight = 0

        for pred in predictions:
            model_score = self._normalize_prediction(pred)
            weight = weights.get(pred.model, 0.1)
            weighted_score += model_score * weight * pred.confidence
            total_weight += weight * pred.confidence

        final_score = weighted_score / total_weight if total_weight > 0 else 0
        return min(max(final_score, 0), 1)  # Clamp to [0, 1]

    def _generate_evidence(
        self,
        imagery: SatelliteImagery,
        predictions: List[Prediction],
        geometry: Polygon
    ) -> DeforestationEvidence:
        """
        Generate visual and statistical evidence
        """
        # Calculate forest loss area
        forest_loss_pixels = np.sum(predictions[0].forest_mask < 0.5)
        pixel_area_m2 = 100  # 10m x 10m
        forest_loss_hectares = (forest_loss_pixels * pixel_area_m2) / 10000

        # Generate change map
        change_map = self._create_change_visualization(
            imagery.baseline,
            imagery.current,
            predictions[1].change_mask
        )

        # Statistical analysis
        stats = {
            'ndvi_baseline_mean': np.mean(imagery.baseline_ndvi),
            'ndvi_current_mean': np.mean(imagery.current_ndvi),
            'ndvi_change': np.mean(imagery.current_ndvi) - np.mean(imagery.baseline_ndvi),
            'forest_loss_hectares': forest_loss_hectares,
            'forest_loss_percentage': (forest_loss_hectares / geometry.area_hectares) * 100
        }

        return DeforestationEvidence(
            satellite_images={
                'baseline': self._encode_image(imagery.baseline),
                'current': self._encode_image(imagery.current),
                'change_map': self._encode_image(change_map)
            },
            statistics=stats,
            model_outputs={
                model.model: model.to_dict() for model in predictions
            },
            acquisition_dates={
                'baseline': f"{self.baseline_year}-12-31",
                'current': str(assessment_date)
            }
        )


class UNetModel(tf.keras.Model):
    """
    U-Net architecture for forest segmentation
    """

    def __init__(self, input_shape=(256, 256, 6)):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(32)
        self.enc2 = self._conv_block(64)
        self.enc3 = self._conv_block(128)
        self.enc4 = self._conv_block(256)

        # Bottleneck
        self.bottleneck = self._conv_block(512)

        # Decoder
        self.dec4 = self._upconv_block(256)
        self.dec3 = self._upconv_block(128)
        self.dec2 = self._upconv_block(64)
        self.dec1 = self._upconv_block(32)

        # Output
        self.output_conv = tf.keras.layers.Conv2D(
            1,
            kernel_size=1,
            activation='sigmoid'
        )

    def _conv_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2)
        ])

    def _upconv_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding='same'),
            tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization()
        ])
```

**Risk Assessment Database**:
```sql
-- Deforestation risk assessments
CREATE TABLE risk_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES validated_plots(id),
    assessment_date TIMESTAMP DEFAULT NOW(),
    risk_score DECIMAL(3,2) CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_level VARCHAR(20) NOT NULL,
    deforestation_detected BOOLEAN DEFAULT FALSE,
    forest_loss_hectares DECIMAL(10,2),
    forest_loss_percentage DECIMAL(5,2),
    ndvi_change DECIMAL(5,3),
    confidence_score DECIMAL(3,2),
    evidence JSONB,
    model_versions JSONB,
    processing_time_seconds INTEGER,

    CONSTRAINT valid_risk_level CHECK (
        risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    )
);

CREATE INDEX idx_risk_plot ON risk_assessments (plot_id);
CREATE INDEX idx_risk_level ON risk_assessments (risk_level);
CREATE INDEX idx_risk_date ON risk_assessments (assessment_date);

-- Satellite imagery cache
CREATE TABLE satellite_imagery_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plot_id UUID REFERENCES validated_plots(id),
    acquisition_date DATE NOT NULL,
    satellite_source VARCHAR(50),
    bands TEXT[],
    cloud_coverage DECIMAL(5,2),
    image_data BYTEA,  -- Or reference to S3
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(plot_id, acquisition_date, satellite_source)
);
```

---

### 2.4 Agent 4: DocumentVerificationAgent

**Purpose**: AI-powered document verification using LLM and RAG

**Technology Stack**:
- Runtime: Python 3.11
- LLM: OpenAI GPT-4, Anthropic Claude
- RAG: LangChain, ChromaDB
- OCR: Tesseract, Azure Form Recognizer
- Document Storage: S3/Azure Blob

**Implementation**:
```python
# document_verification_agent.py
from langchain import OpenAI, PromptTemplate
from langchain.vectorstores import Chroma
from langchain.document_loaders import PDFLoader
import pytesseract

class DocumentVerificationAgent:
    """
    LLM-based document verification for compliance
    """

    def __init__(self):
        self.llm = OpenAI(model="gpt-4", temperature=0)
        self.vector_store = Chroma(
            embedding_function=OpenAIEmbeddings(),
            persist_directory="./chroma_db"
        )
        self.load_compliance_knowledge()

    def verify_document(
        self,
        document: Document,
        commodity: str,
        supplier_id: str
    ) -> DocumentVerification:
        """
        Verify compliance documents using RAG + LLM
        """
        # Step 1: OCR if needed
        if document.type == 'image' or document.type == 'pdf':
            text = self._extract_text(document)
        else:
            text = document.content

        # Step 2: Classify document type
        doc_type = self._classify_document(text)

        # Step 3: Extract key information
        extracted_info = self._extract_information(text, doc_type)

        # Step 4: Verify against compliance requirements
        verification = self._verify_compliance(
            extracted_info,
            doc_type,
            commodity
        )

        # Step 5: Check authenticity markers
        authenticity = self._check_authenticity(document, extracted_info)

        return DocumentVerification(
            document_id=document.id,
            document_type=doc_type,
            extracted_data=extracted_info,
            compliance_status=verification.status,
            compliance_issues=verification.issues,
            authenticity_score=authenticity.score,
            confidence=verification.confidence,
            verified_at=datetime.now()
        )

    def _classify_document(self, text: str) -> str:
        """
        Classify document type using LLM
        """
        prompt = PromptTemplate(
            template="""
            Classify this document into one of these EUDR compliance categories:
            - FOREST_CERTIFICATE
            - LAND_OWNERSHIP
            - HARVEST_PERMIT
            - ORIGIN_DECLARATION
            - SUSTAINABILITY_CERT
            - LEGAL_COMPLIANCE
            - OTHER

            Document text:
            {text}

            Classification (respond with only the category name):
            """,
            input_variables=["text"]
        )

        response = self.llm(prompt.format(text=text[:3000]))
        return response.strip()

    def _extract_information(self, text: str, doc_type: str) -> Dict:
        """
        Extract structured information from document
        """
        extraction_prompts = {
            'FOREST_CERTIFICATE': """
                Extract from this forest certificate:
                - Certificate number
                - Issue date
                - Expiry date
                - Forest management unit
                - Certified area (hectares)
                - Certification body
                - GPS coordinates (if present)
            """,
            'HARVEST_PERMIT': """
                Extract from this harvest permit:
                - Permit number
                - Issuing authority
                - Valid from/to dates
                - Permitted species
                - Permitted volume
                - Location/plot identifier
                - GPS coordinates (if present)
            """
        }

        prompt = PromptTemplate(
            template=extraction_prompts.get(doc_type, "Extract all relevant information") + """

            Document text:
            {text}

            Return as JSON:
            """,
            input_variables=["text"]
        )

        response = self.llm(prompt.format(text=text))
        return json.loads(response)

    def _verify_compliance(
        self,
        extracted_info: Dict,
        doc_type: str,
        commodity: str
    ) -> ComplianceVerification:
        """
        Verify document meets EUDR requirements using RAG
        """
        # Query vector store for relevant compliance rules
        relevant_rules = self.vector_store.similarity_search(
            f"EUDR requirements for {commodity} {doc_type}",
            k=5
        )

        # Build verification prompt
        prompt = PromptTemplate(
            template="""
            Verify if this document meets EUDR compliance requirements.

            Document Type: {doc_type}
            Commodity: {commodity}
            Extracted Information: {info}

            Relevant EUDR Rules:
            {rules}

            Provide:
            1. Compliance status (COMPLIANT/NON_COMPLIANT/INSUFFICIENT_INFO)
            2. List any compliance issues
            3. Confidence score (0-1)
            4. Missing information needed

            Response as JSON:
            """,
            input_variables=["doc_type", "commodity", "info", "rules"]
        )

        response = self.llm(prompt.format(
            doc_type=doc_type,
            commodity=commodity,
            info=json.dumps(extracted_info),
            rules="\n".join([doc.page_content for doc in relevant_rules])
        ))

        return ComplianceVerification(**json.loads(response))

    def _check_authenticity(
        self,
        document: Document,
        extracted_info: Dict
    ) -> AuthenticityCheck:
        """
        Check document authenticity markers
        """
        checks = {
            'has_signature': self._detect_signature(document),
            'has_stamp': self._detect_official_stamp(document),
            'has_watermark': self._detect_watermark(document),
            'valid_format': self._check_format_validity(extracted_info),
            'issuer_verified': self._verify_issuer(extracted_info)
        }

        score = sum(checks.values()) / len(checks)

        return AuthenticityCheck(
            score=score,
            checks=checks,
            risk_level='LOW' if score > 0.7 else 'HIGH'
        )

    def load_compliance_knowledge(self):
        """
        Load EUDR compliance rules into vector store
        """
        documents = [
            "EUDR Article 3: Products shall be deforestation-free",
            "EUDR Article 4: Products shall be produced in accordance with relevant legislation",
            "EUDR Article 9: Operators shall exercise due diligence",
            "Forest certificates must be issued by FSC or PEFC certified bodies",
            "Harvest permits must include GPS coordinates with 6 decimal precision",
            # ... more compliance rules
        ]

        for doc in documents:
            self.vector_store.add_texts([doc])
```

---

### 2.5 Agent 5: DDSReportingAgent

**Purpose**: Generate and submit Due Diligence Statements to EU portal

**Technology Stack**:
- Runtime: Node.js 20
- Framework: Express.js
- Template Engine: Handlebars
- EU Integration: SOAP/REST
- Queue: Bull (Redis-based)

**Implementation**:
```javascript
// dds_reporting_agent.js
const express = require('express');
const Bull = require('bull');
const soap = require('soap');
const { generateDDS } = require('./dds-generator');

class DDSReportingAgent {
    constructor() {
        this.ddsQueue = new Bull('dds-submission');
        this.euPortalClient = null;
        this.initializeEUPortal();
    }

    async initializeEUPortal() {
        // Initialize EU Portal SOAP client
        const wsdlUrl = process.env.EU_PORTAL_WSDL;
        this.euPortalClient = await soap.createClientAsync(wsdlUrl);
    }

    async generateDueDiligenceStatement(complianceData) {
        /**
         * Generate DDS from aggregated compliance data
         */
        const dds = {
            referenceNumber: this.generateReferenceNumber(),
            submissionDate: new Date().toISOString(),
            operator: {
                name: complianceData.company.name,
                taxId: complianceData.company.taxId,
                address: complianceData.company.address,
                economicOperatorId: complianceData.company.eoriNumber
            },
            commodity: {
                type: complianceData.commodity.type,
                cnCode: complianceData.commodity.cnCode,
                quantity: complianceData.commodity.quantity,
                unit: complianceData.commodity.unit
            },
            suppliers: complianceData.suppliers.map(supplier => ({
                name: supplier.name,
                country: supplier.country,
                plots: supplier.plots.map(plot => ({
                    coordinates: this.formatCoordinates(plot.geometry),
                    areaHectares: plot.areaHectares,
                    commodityType: plot.commodityType
                }))
            })),
            riskAssessment: {
                overallRisk: complianceData.riskLevel,
                deforestationRisk: complianceData.deforestationRisk,
                legalityRisk: complianceData.legalityRisk,
                mitigationMeasures: complianceData.mitigationMeasures
            },
            conclusions: {
                compliant: complianceData.isCompliant,
                negligibleRisk: complianceData.riskLevel === 'LOW',
                statement: this.generateComplianceStatement(complianceData)
            },
            attachments: complianceData.documents.map(doc => ({
                type: doc.type,
                name: doc.name,
                hash: doc.sha256Hash,
                url: doc.storageUrl
            }))
        };

        // Generate PDF version
        const ddsPdf = await this.generatePDF(dds);

        // Store in database
        await this.saveDDS(dds, ddsPdf);

        return dds;
    }

    async submitToEUPortal(dds) {
        /**
         * Submit DDS to EU Information System
         */
        try {
            // Prepare SOAP request
            const request = {
                DueDiligenceStatement: {
                    ReferenceNumber: dds.referenceNumber,
                    OperatorDetails: this.mapOperatorToEU(dds.operator),
                    CommodityDetails: this.mapCommodityToEU(dds.commodity),
                    GeolocationData: this.mapGeolocationToEU(dds.suppliers),
                    RiskAssessment: this.mapRiskToEU(dds.riskAssessment),
                    Attachments: dds.attachments
                }
            };

            // Submit via SOAP
            const response = await this.euPortalClient.SubmitDDSAsync(request);

            if (response.Success) {
                return {
                    success: true,
                    euReferenceNumber: response.EUReferenceNumber,
                    submissionId: response.SubmissionId,
                    timestamp: response.Timestamp
                };
            } else {
                throw new Error(`EU Portal rejection: ${response.ErrorMessage}`);
            }

        } catch (error) {
            // Fallback to REST API if available
            if (process.env.EU_REST_API_ENABLED === 'true') {
                return this.submitViaREST(dds);
            }

            // Queue for retry
            await this.ddsQueue.add('retry-submission', {
                dds,
                attempt: 1,
                error: error.message
            });

            throw error;
        }
    }

    formatCoordinates(geometry) {
        /**
         * Format coordinates per EU requirements
         */
        if (geometry.type === 'Point') {
            const [lon, lat] = geometry.coordinates;
            return {
                type: 'POINT',
                coordinates: `${lat.toFixed(6)}, ${lon.toFixed(6)}`
            };
        } else if (geometry.type === 'Polygon') {
            const coords = geometry.coordinates[0].map(([lon, lat]) =>
                `${lat.toFixed(6)}, ${lon.toFixed(6)}`
            ).join('; ');
            return {
                type: 'POLYGON',
                coordinates: coords
            };
        }
    }

    generateComplianceStatement(data) {
        /**
         * Generate human-readable compliance statement
         */
        const templates = {
            compliant: `Based on our due diligence assessment, the ${data.commodity.type} products covered by this statement are compliant with EU Regulation 2023/1115. No deforestation has been detected in the production areas since December 31, 2020, and all applicable laws in the country of production have been followed.`,

            nonCompliant: `Our due diligence assessment has identified compliance issues with the ${data.commodity.type} products. Deforestation risk level: ${data.deforestationRisk}. The following issues were found: ${data.issues.join(', ')}. These products should not be placed on the EU market without additional mitigation measures.`,

            conditional: `The ${data.commodity.type} products show ${data.riskLevel} risk. While no direct deforestation was detected, the following risk factors require attention: ${data.riskFactors.join(', ')}. Enhanced monitoring and additional documentation are recommended.`
        };

        if (data.isCompliant && data.riskLevel === 'LOW') {
            return templates.compliant;
        } else if (!data.isCompliant) {
            return templates.nonCompliant;
        } else {
            return templates.conditional;
        }
    }

    async monitorSubmissionStatus(submissionId) {
        /**
         * Monitor DDS submission status
         */
        const checkStatus = async () => {
            const response = await this.euPortalClient.GetSubmissionStatusAsync({
                SubmissionId: submissionId
            });

            return {
                status: response.Status,
                euReference: response.EUReferenceNumber,
                validUntil: response.ValidityDate,
                comments: response.Comments
            };
        };

        // Poll every 30 seconds for up to 5 minutes
        for (let i = 0; i < 10; i++) {
            const status = await checkStatus();

            if (status.status === 'ACCEPTED' || status.status === 'REJECTED') {
                return status;
            }

            await new Promise(resolve => setTimeout(resolve, 30000));
        }

        throw new Error('Submission status check timeout');
    }
}

// Express API endpoints
const app = express();
const ddsAgent = new DDSReportingAgent();

app.post('/api/v1/dds/generate', async (req, res) => {
    try {
        const dds = await ddsAgent.generateDueDiligenceStatement(req.body);
        res.json({ success: true, dds });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/api/v1/dds/submit', async (req, res) => {
    try {
        const result = await ddsAgent.submitToEUPortal(req.body.dds);
        res.json({ success: true, submission: result });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/api/v1/dds/status/:submissionId', async (req, res) => {
    try {
        const status = await ddsAgent.monitorSubmissionStatus(req.params.submissionId);
        res.json({ success: true, status });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});
```

---

## 3. INFRASTRUCTURE ARCHITECTURE

### 3.1 Cloud Infrastructure (AWS)

```yaml
# infrastructure.yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: GL-EUDR-APP Infrastructure

Resources:
  # VPC Configuration
  EUDRVPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true

  # EKS Cluster for Microservices
  EUDRCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: eudr-production
      Version: '1.27'
      RoleArn: !GetAtt EKSServiceRole.Arn
      ResourcesVpcConfig:
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2
        SecurityGroupIds:
          - !Ref ClusterSecurityGroup

  # RDS PostgreSQL with PostGIS
  EUDRDatabase:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: eudr-postgres
      Engine: postgres
      EngineVersion: '15.3'
      DBInstanceClass: db.r6g.4xlarge
      AllocatedStorage: 1000
      StorageType: gp3
      StorageEncrypted: true
      MasterUsername: eudradmin
      MasterUserPassword: !Ref DBPassword
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      BackupRetentionPeriod: 30
      PreferredBackupWindow: "03:00-04:00"
      MultiAZ: true

  # ElastiCache Redis
  EUDRCache:
    Type: AWS::ElastiCache::CacheCluster
    Properties:
      CacheNodeType: cache.r6g.xlarge
      Engine: redis
      NumCacheNodes: 3
      VpcSecurityGroupIds:
        - !Ref CacheSecurityGroup

  # S3 Buckets
  DocumentBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: eudr-documents
      VersioningConfiguration:
        Status: Enabled
      LifecycleConfiguration:
        Rules:
          - Id: ArchiveOldDocuments
            Status: Enabled
            Transitions:
              - TransitionInDays: 90
                StorageClass: GLACIER
      ServerSideEncryptionConfiguration:
        Rules:
          - ApplyServerSideEncryptionByDefault:
              SSEAlgorithm: AES256

  SatelliteImageryBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: eudr-satellite-imagery
      LifecycleConfiguration:
        Rules:
          - Id: DeleteOldImagery
            Status: Enabled
            ExpirationInDays: 180

  # Auto Scaling Configuration
  AgentAutoScaling:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      MinSize: 2
      MaxSize: 20
      DesiredCapacity: 4
      TargetGroupARNs:
        - !Ref ALBTargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300

  # CloudWatch Alarms
  HighErrorRateAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmDescription: Alert when error rate is too high
      MetricName: 5XXError
      Namespace: AWS/ApplicationELB
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 2
      Threshold: 100
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref SNSAlertTopic
```

### 3.2 Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: eudr-platform

---
# Agent 1: Supplier Intake
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supplier-intake-agent
  namespace: eudr-platform
spec:
  replicas: 3
  selector:
    matchLabels:
      app: supplier-intake
  template:
    metadata:
      labels:
        app: supplier-intake
    spec:
      containers:
      - name: agent
        image: eudr/supplier-intake:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: KAFKA_BROKERS
          value: kafka-broker.eudr-platform:9092
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
# Agent 2: Geo Validation
apiVersion: apps/v1
kind: Deployment
metadata:
  name: geo-validation-agent
  namespace: eudr-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: geo-validation
  template:
    metadata:
      labels:
        app: geo-validation
    spec:
      containers:
      - name: agent
        image: eudr/geo-validation:1.0.0
        ports:
        - containerPort: 8001
        env:
        - name: POSTGIS_URL
          valueFrom:
            secretKeyRef:
              name: postgis-secret
              key: url
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"

---
# Agent 3: Deforestation Risk (GPU enabled)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deforestation-risk-agent
  namespace: eudr-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deforestation-risk
  template:
    metadata:
      labels:
        app: deforestation-risk
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: p3.2xlarge
      containers:
      - name: agent
        image: eudr/deforestation-risk:1.0.0
        ports:
        - containerPort: 8002
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: SENTINEL_API_KEY
          valueFrom:
            secretKeyRef:
              name: satellite-secret
              key: sentinel-key
        - name: MODEL_PATH
          value: /models/deforestation-v2.h5

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: supplier-intake-hpa
  namespace: eudr-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: supplier-intake-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# Service Mesh (Istio)
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: eudr-routing
  namespace: eudr-platform
spec:
  hosts:
  - eudr-platform.internal
  http:
  - match:
    - uri:
        prefix: /api/v1/intake
    route:
    - destination:
        host: supplier-intake-service
        port:
          number: 8000
  - match:
    - uri:
        prefix: /api/v1/geo
    route:
    - destination:
        host: geo-validation-service
        port:
          number: 8001
```

---

## 4. SECURITY ARCHITECTURE

### 4.1 Security Layers

```yaml
# security-config.yaml
security:
  network:
    waf:
      provider: AWS WAF
      rules:
        - sql_injection_protection
        - xss_protection
        - rate_limiting

    vpc:
      private_subnets: true
      nat_gateway: true
      flow_logs: enabled

    ssl:
      version: TLS 1.3
      certificates: AWS Certificate Manager

  authentication:
    provider: Auth0 / AWS Cognito
    mfa: required_for_admin
    password_policy:
      min_length: 12
      require_special: true
      require_numbers: true
      rotation_days: 90

  authorization:
    model: RBAC
    roles:
      - super_admin
      - compliance_manager
      - operator
      - auditor
      - readonly

  encryption:
    at_rest:
      algorithm: AES-256-GCM
      key_management: AWS KMS
      rotation: 90_days

    in_transit:
      protocol: TLS 1.3
      cipher_suites:
        - TLS_AES_256_GCM_SHA384
        - TLS_CHACHA20_POLY1305_SHA256

  secrets:
    manager: HashiCorp Vault / AWS Secrets Manager
    rotation: automatic

  monitoring:
    siem: Splunk / ELK Stack
    ids: Snort / Suricata
    vulnerability_scanning: weekly
    penetration_testing: quarterly
```

---

## 5. MONITORING & OBSERVABILITY

### 5.1 Metrics & Dashboards

```yaml
# monitoring-stack.yaml
monitoring:
  metrics:
    provider: Prometheus
    scrape_interval: 15s
    retention: 30d

    key_metrics:
      - dds_processing_rate
      - supplier_compliance_percentage
      - deforestation_detections_per_day
      - api_response_time_p95
      - error_rate
      - system_availability

  visualization:
    provider: Grafana
    dashboards:
      - operational_overview
      - compliance_status
      - deforestation_monitoring
      - system_performance
      - security_events

  tracing:
    provider: Jaeger / AWS X-Ray
    sampling_rate: 0.1

  logging:
    aggregator: ELK Stack / AWS CloudWatch
    retention: 90_days
    log_levels:
      production: INFO
      staging: DEBUG

  alerting:
    channels:
      - pagerduty
      - slack
      - email
    escalation:
      p1: immediate
      p2: 15_minutes
      p3: 1_hour
```

---

## 6. DISASTER RECOVERY

### 6.1 Backup Strategy

```yaml
# disaster-recovery.yaml
backup:
  database:
    frequency: hourly
    retention:
      hourly: 24
      daily: 30
      monthly: 12
    location: cross-region S3

  documents:
    strategy: continuous replication
    regions:
      - us-east-1
      - eu-west-1
      - ap-southeast-1

recovery:
  rto: 15 minutes
  rpo: 5 minutes

  procedures:
    - automated_failover
    - dns_switching
    - cache_warming
    - health_checks

testing:
  frequency: monthly
  scenarios:
    - region_failure
    - database_corruption
    - ddos_attack
    - data_breach
```

---

*Architecture Version: 1.0*
*Last Updated: November 2024*
*Architect: GL-EUDR Technical Team*
*Status: APPROVED FOR IMPLEMENTATION*