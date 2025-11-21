// ==============================================
// GreenLang MongoDB NoSQL Architecture
// Version: 1.0.0
// Collections: 100+ across multiple databases
// ==============================================

// MongoDB Connection and Configuration
const mongoConfig = {
    replicaSet: 'greenlang-rs',
    writeConcern: { w: 'majority', j: true, wtimeout: 5000 },
    readConcern: { level: 'majority' },
    readPreference: 'secondaryPreferred',
    connectionPoolSize: 100,
    serverSelectionTimeoutMS: 30000,
    socketTimeoutMS: 45000,
    compression: ['snappy', 'zlib']
};

// ==============================================
// DATABASE: greenlang_documents
// ==============================================

db = db.getSiblingDB('greenlang_documents');

// CSRD Reports Collection (10M+ documents)
db.createCollection('csrd_reports', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['organization_id', 'report_type', 'reporting_period', 'created_at'],
            properties: {
                organization_id: {
                    bsonType: 'string',
                    description: 'UUID of organization'
                },
                report_type: {
                    enum: ['annual', 'quarterly', 'monthly', 'ad-hoc'],
                    description: 'Type of CSRD report'
                },
                reporting_period: {
                    bsonType: 'object',
                    required: ['start_date', 'end_date'],
                    properties: {
                        start_date: { bsonType: 'date' },
                        end_date: { bsonType: 'date' }
                    }
                },
                esrs_standards: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            standard_code: { bsonType: 'string' },
                            disclosure_items: { bsonType: 'array' },
                            compliance_status: { bsonType: 'string' }
                        }
                    }
                },
                sections: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            title: { bsonType: 'string' },
                            content: { bsonType: 'string' },
                            data_points: { bsonType: 'array' },
                            charts: { bsonType: 'array' },
                            tables: { bsonType: 'array' },
                            footnotes: { bsonType: 'array' }
                        }
                    }
                },
                materiality_assessment: {
                    bsonType: 'object',
                    properties: {
                        material_topics: { bsonType: 'array' },
                        impact_assessment: { bsonType: 'object' },
                        stakeholder_engagement: { bsonType: 'object' }
                    }
                },
                verification: {
                    bsonType: 'object',
                    properties: {
                        status: { enum: ['draft', 'in_review', 'verified', 'published'] },
                        auditor: { bsonType: 'string' },
                        audit_date: { bsonType: 'date' },
                        audit_opinion: { bsonType: 'string' },
                        findings: { bsonType: 'array' }
                    }
                },
                attachments: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            file_name: { bsonType: 'string' },
                            file_type: { bsonType: 'string' },
                            file_size: { bsonType: 'int' },
                            storage_path: { bsonType: 'string' },
                            checksum: { bsonType: 'string' }
                        }
                    }
                },
                created_at: { bsonType: 'date' },
                updated_at: { bsonType: 'date' },
                published_at: { bsonType: 'date' },
                version: { bsonType: 'int' },
                metadata: { bsonType: 'object' }
            }
        }
    }
});

// Indexes for CSRD Reports
db.csrd_reports.createIndex({ organization_id: 1, reporting_period: -1 });
db.csrd_reports.createIndex({ 'reporting_period.start_date': -1 });
db.csrd_reports.createIndex({ 'verification.status': 1 });
db.csrd_reports.createIndex({ report_type: 1, created_at: -1 });
db.csrd_reports.createIndex({ 'esrs_standards.standard_code': 1 });
// Text search index
db.csrd_reports.createIndex({
    'sections.title': 'text',
    'sections.content': 'text'
});

// Sustainability Documents Collection
db.createCollection('sustainability_documents', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['organization_id', 'document_type', 'title'],
            properties: {
                organization_id: { bsonType: 'string' },
                document_type: {
                    enum: ['policy', 'procedure', 'assessment', 'certificate', 'audit_report', 'action_plan']
                },
                title: { bsonType: 'string' },
                content: { bsonType: 'string' },
                tags: { bsonType: 'array', items: { bsonType: 'string' } },
                related_standards: { bsonType: 'array' },
                approval_workflow: {
                    bsonType: 'object',
                    properties: {
                        status: { bsonType: 'string' },
                        approvers: { bsonType: 'array' },
                        approval_history: { bsonType: 'array' }
                    }
                },
                version_control: {
                    bsonType: 'object',
                    properties: {
                        version: { bsonType: 'string' },
                        previous_versions: { bsonType: 'array' },
                        change_log: { bsonType: 'array' }
                    }
                },
                access_control: {
                    bsonType: 'object',
                    properties: {
                        visibility: { enum: ['public', 'internal', 'confidential'] },
                        authorized_users: { bsonType: 'array' },
                        authorized_roles: { bsonType: 'array' }
                    }
                }
            }
        }
    }
});

// ==============================================
// DATABASE: greenlang_analytics
// ==============================================

db = db.getSiblingDB('greenlang_analytics');

// Aggregated Emissions Analytics
db.createCollection('emissions_analytics', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['organization_id', 'period', 'metrics'],
            properties: {
                organization_id: { bsonType: 'string' },
                period: {
                    bsonType: 'object',
                    required: ['year', 'month'],
                    properties: {
                        year: { bsonType: 'int' },
                        month: { bsonType: 'int' },
                        quarter: { bsonType: 'int' }
                    }
                },
                metrics: {
                    bsonType: 'object',
                    properties: {
                        total_emissions: {
                            bsonType: 'object',
                            properties: {
                                scope1: { bsonType: 'double' },
                                scope2: { bsonType: 'double' },
                                scope3: { bsonType: 'object' },
                                total: { bsonType: 'double' }
                            }
                        },
                        emissions_by_source: { bsonType: 'array' },
                        emissions_by_facility: { bsonType: 'array' },
                        emissions_by_category: { bsonType: 'array' },
                        intensity_metrics: {
                            bsonType: 'object',
                            properties: {
                                per_revenue: { bsonType: 'double' },
                                per_employee: { bsonType: 'double' },
                                per_unit_produced: { bsonType: 'double' }
                            }
                        },
                        year_over_year: {
                            bsonType: 'object',
                            properties: {
                                absolute_change: { bsonType: 'double' },
                                percentage_change: { bsonType: 'double' },
                                trend: { enum: ['increasing', 'decreasing', 'stable'] }
                            }
                        }
                    }
                },
                targets: {
                    bsonType: 'object',
                    properties: {
                        sbti_aligned: { bsonType: 'bool' },
                        net_zero_target: { bsonType: 'date' },
                        interim_targets: { bsonType: 'array' },
                        progress_percentage: { bsonType: 'double' }
                    }
                },
                benchmarks: {
                    bsonType: 'object',
                    properties: {
                        industry_average: { bsonType: 'double' },
                        percentile_rank: { bsonType: 'int' },
                        peer_comparison: { bsonType: 'array' }
                    }
                }
            }
        }
    }
});

// Compound index for time-series queries
db.emissions_analytics.createIndex({
    organization_id: 1,
    'period.year': -1,
    'period.month': -1
});

// Supply Chain Analytics
db.createCollection('supply_chain_analytics', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['organization_id', 'supplier_id', 'assessment_date'],
            properties: {
                organization_id: { bsonType: 'string' },
                supplier_id: { bsonType: 'string' },
                assessment_date: { bsonType: 'date' },
                risk_assessment: {
                    bsonType: 'object',
                    properties: {
                        overall_risk_score: { bsonType: 'int', minimum: 0, maximum: 100 },
                        environmental_risk: { bsonType: 'int' },
                        social_risk: { bsonType: 'int' },
                        governance_risk: { bsonType: 'int' },
                        operational_risk: { bsonType: 'int' },
                        risk_factors: { bsonType: 'array' },
                        mitigation_actions: { bsonType: 'array' }
                    }
                },
                sustainability_performance: {
                    bsonType: 'object',
                    properties: {
                        carbon_footprint: { bsonType: 'double' },
                        water_usage: { bsonType: 'double' },
                        waste_generation: { bsonType: 'double' },
                        renewable_energy_percentage: { bsonType: 'double' },
                        certifications: { bsonType: 'array' }
                    }
                },
                procurement_metrics: {
                    bsonType: 'object',
                    properties: {
                        total_spend: { bsonType: 'double' },
                        order_count: { bsonType: 'int' },
                        on_time_delivery_rate: { bsonType: 'double' },
                        quality_score: { bsonType: 'double' },
                        cost_savings: { bsonType: 'double' }
                    }
                },
                collaboration_metrics: {
                    bsonType: 'object',
                    properties: {
                        innovation_projects: { bsonType: 'array' },
                        improvement_initiatives: { bsonType: 'array' },
                        training_programs: { bsonType: 'array' }
                    }
                }
            }
        }
    }
});

// ==============================================
// DATABASE: greenlang_iot
// ==============================================

db = db.getSiblingDB('greenlang_iot');

// IoT Device Telemetry (High-volume time-series)
db.createCollection('device_telemetry', {
    timeseries: {
        timeField: 'timestamp',
        metaField: 'metadata',
        granularity: 'minutes'
    },
    expireAfterSeconds: 31536000 // 1 year retention
});

// Sample document structure for device_telemetry
/*
{
    timestamp: ISODate("2024-01-01T12:00:00Z"),
    metadata: {
        device_id: "sensor-001",
        device_type: "air_quality",
        location: {
            facility_id: "facility-001",
            building: "A",
            floor: 3,
            coordinates: [longitude, latitude]
        }
    },
    measurements: {
        temperature: 23.5,
        humidity: 45.2,
        co2_level: 420,
        pm2_5: 12.3,
        pm10: 18.7,
        voc: 0.3
    },
    quality_indicators: {
        signal_strength: -45,
        battery_level: 87,
        calibration_status: "valid"
    }
}
*/

// IoT Alert Events
db.createCollection('alert_events', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['device_id', 'alert_type', 'severity', 'timestamp'],
            properties: {
                device_id: { bsonType: 'string' },
                alert_type: {
                    enum: ['threshold_exceeded', 'device_offline', 'anomaly_detected', 'maintenance_required']
                },
                severity: {
                    enum: ['low', 'medium', 'high', 'critical']
                },
                timestamp: { bsonType: 'date' },
                measurements: { bsonType: 'object' },
                threshold_details: {
                    bsonType: 'object',
                    properties: {
                        metric: { bsonType: 'string' },
                        threshold_value: { bsonType: 'double' },
                        actual_value: { bsonType: 'double' },
                        duration: { bsonType: 'int' }
                    }
                },
                actions_taken: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            action_type: { bsonType: 'string' },
                            timestamp: { bsonType: 'date' },
                            performed_by: { bsonType: 'string' },
                            result: { bsonType: 'string' }
                        }
                    }
                },
                resolution: {
                    bsonType: 'object',
                    properties: {
                        status: { enum: ['open', 'acknowledged', 'in_progress', 'resolved'] },
                        resolved_at: { bsonType: 'date' },
                        resolved_by: { bsonType: 'string' },
                        resolution_notes: { bsonType: 'string' }
                    }
                }
            }
        }
    }
});

db.alert_events.createIndex({ device_id: 1, timestamp: -1 });
db.alert_events.createIndex({ severity: 1, 'resolution.status': 1 });
db.alert_events.createIndex({ alert_type: 1 });

// ==============================================
// DATABASE: greenlang_workflows
// ==============================================

db = db.getSiblingDB('greenlang_workflows');

// Workflow Definitions
db.createCollection('workflow_definitions', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['workflow_name', 'workflow_type', 'steps'],
            properties: {
                workflow_name: { bsonType: 'string' },
                workflow_type: {
                    enum: ['data_collection', 'approval', 'verification', 'reporting', 'audit']
                },
                description: { bsonType: 'string' },
                trigger_conditions: {
                    bsonType: 'object',
                    properties: {
                        event_type: { bsonType: 'string' },
                        schedule: { bsonType: 'string' }, // Cron expression
                        manual: { bsonType: 'bool' }
                    }
                },
                steps: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        required: ['step_id', 'step_name', 'action_type'],
                        properties: {
                            step_id: { bsonType: 'int' },
                            step_name: { bsonType: 'string' },
                            action_type: { bsonType: 'string' },
                            assignee_role: { bsonType: 'string' },
                            sla_hours: { bsonType: 'int' },
                            conditions: { bsonType: 'object' },
                            validations: { bsonType: 'array' },
                            notifications: { bsonType: 'array' }
                        }
                    }
                },
                escalation_rules: { bsonType: 'array' },
                is_active: { bsonType: 'bool' }
            }
        }
    }
});

// Workflow Instances (Execution tracking)
db.createCollection('workflow_instances', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['workflow_id', 'organization_id', 'status', 'started_at'],
            properties: {
                workflow_id: { bsonType: 'objectId' },
                organization_id: { bsonType: 'string' },
                status: {
                    enum: ['pending', 'in_progress', 'completed', 'failed', 'cancelled']
                },
                current_step: { bsonType: 'int' },
                started_at: { bsonType: 'date' },
                completed_at: { bsonType: 'date' },
                context_data: { bsonType: 'object' },
                step_history: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            step_id: { bsonType: 'int' },
                            status: { bsonType: 'string' },
                            started_at: { bsonType: 'date' },
                            completed_at: { bsonType: 'date' },
                            assigned_to: { bsonType: 'string' },
                            action_taken: { bsonType: 'string' },
                            comments: { bsonType: 'string' },
                            data_snapshot: { bsonType: 'object' }
                        }
                    }
                },
                error_details: { bsonType: 'object' }
            }
        }
    }
});

// ==============================================
// DATABASE: greenlang_ml
// ==============================================

db = db.getSiblingDB('greenlang_ml');

// ML Model Registry
db.createCollection('model_registry', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['model_name', 'model_type', 'version'],
            properties: {
                model_name: { bsonType: 'string' },
                model_type: {
                    enum: ['regression', 'classification', 'time_series', 'anomaly_detection', 'nlp']
                },
                version: { bsonType: 'string' },
                description: { bsonType: 'string' },
                algorithm: { bsonType: 'string' },
                training_metadata: {
                    bsonType: 'object',
                    properties: {
                        training_date: { bsonType: 'date' },
                        dataset_version: { bsonType: 'string' },
                        features: { bsonType: 'array' },
                        hyperparameters: { bsonType: 'object' },
                        training_metrics: { bsonType: 'object' }
                    }
                },
                performance_metrics: {
                    bsonType: 'object',
                    properties: {
                        accuracy: { bsonType: 'double' },
                        precision: { bsonType: 'double' },
                        recall: { bsonType: 'double' },
                        f1_score: { bsonType: 'double' },
                        rmse: { bsonType: 'double' },
                        mae: { bsonType: 'double' }
                    }
                },
                deployment_info: {
                    bsonType: 'object',
                    properties: {
                        status: { enum: ['development', 'staging', 'production', 'retired'] },
                        endpoint_url: { bsonType: 'string' },
                        deployed_at: { bsonType: 'date' },
                        serving_framework: { bsonType: 'string' }
                    }
                },
                model_artifacts: {
                    bsonType: 'object',
                    properties: {
                        model_path: { bsonType: 'string' },
                        model_size_mb: { bsonType: 'double' },
                        checksum: { bsonType: 'string' }
                    }
                }
            }
        }
    }
});

// ML Predictions Log
db.createCollection('predictions_log', {
    timeseries: {
        timeField: 'timestamp',
        metaField: 'metadata',
        granularity: 'hours'
    },
    expireAfterSeconds: 7776000 // 90 days retention
});

// ==============================================
// DATABASE: greenlang_cache
// ==============================================

db = db.getSiblingDB('greenlang_cache');

// API Response Cache
db.createCollection('api_cache', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['cache_key', 'response_data', 'expires_at'],
            properties: {
                cache_key: { bsonType: 'string' },
                endpoint: { bsonType: 'string' },
                request_params: { bsonType: 'object' },
                response_data: { bsonType: 'object' },
                response_headers: { bsonType: 'object' },
                cache_metadata: {
                    bsonType: 'object',
                    properties: {
                        hit_count: { bsonType: 'int' },
                        last_accessed: { bsonType: 'date' },
                        cache_size_bytes: { bsonType: 'int' }
                    }
                },
                expires_at: { bsonType: 'date' }
            }
        }
    }
});

// TTL index for automatic cache expiration
db.api_cache.createIndex({ expires_at: 1 }, { expireAfterSeconds: 0 });
db.api_cache.createIndex({ cache_key: 1 }, { unique: true });

// ==============================================
// SHARDING CONFIGURATION
// ==============================================

// Enable sharding for high-volume collections
sh.enableSharding('greenlang_documents');
sh.enableSharding('greenlang_analytics');
sh.enableSharding('greenlang_iot');

// Shard CSRD Reports by organization_id
sh.shardCollection(
    'greenlang_documents.csrd_reports',
    { organization_id: 'hashed' },
    false,
    { numInitialChunks: 64 }
);

// Shard emissions analytics by organization and time
sh.shardCollection(
    'greenlang_analytics.emissions_analytics',
    { organization_id: 1, 'period.year': 1 },
    false,
    { numInitialChunks: 128 }
);

// Shard IoT telemetry by device and time
sh.shardCollection(
    'greenlang_iot.device_telemetry',
    { 'metadata.device_id': 1, timestamp: 1 },
    false,
    { numInitialChunks: 256 }
);

// ==============================================
// AGGREGATION PIPELINES
// ==============================================

// Create stored aggregation pipeline for emissions reporting
db.getSiblingDB('greenlang_analytics').system.js.save({
    _id: 'generateEmissionsReport',
    value: function(orgId, startDate, endDate) {
        return db.emissions_analytics.aggregate([
            {
                $match: {
                    organization_id: orgId,
                    'period.year': {
                        $gte: startDate.getFullYear(),
                        $lte: endDate.getFullYear()
                    }
                }
            },
            {
                $group: {
                    _id: {
                        year: '$period.year',
                        quarter: '$period.quarter'
                    },
                    total_scope1: { $sum: '$metrics.total_emissions.scope1' },
                    total_scope2: { $sum: '$metrics.total_emissions.scope2' },
                    total_scope3: { $sum: '$metrics.total_emissions.scope3.total' },
                    avg_intensity_per_revenue: { $avg: '$metrics.intensity_metrics.per_revenue' }
                }
            },
            {
                $sort: {
                    '_id.year': 1,
                    '_id.quarter': 1
                }
            },
            {
                $facet: {
                    quarterly_data: [{ $match: {} }],
                    yearly_totals: [
                        {
                            $group: {
                                _id: '$_id.year',
                                annual_total: {
                                    $sum: {
                                        $add: ['$total_scope1', '$total_scope2', '$total_scope3']
                                    }
                                }
                            }
                        }
                    ],
                    summary_stats: [
                        {
                            $group: {
                                _id: null,
                                total_emissions: {
                                    $sum: {
                                        $add: ['$total_scope1', '$total_scope2', '$total_scope3']
                                    }
                                },
                                avg_quarterly: {
                                    $avg: {
                                        $add: ['$total_scope1', '$total_scope2', '$total_scope3']
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        ]).toArray();
    }
});

// ==============================================
// CHANGE STREAMS FOR REAL-TIME SYNC
// ==============================================

// Example change stream setup for real-time notifications
/*
const changeStream = db.getSiblingDB('greenlang_documents').csrd_reports.watch(
    [
        {
            $match: {
                $or: [
                    { operationType: 'insert' },
                    { operationType: 'update' }
                ]
            }
        }
    ],
    {
        fullDocument: 'updateLookup',
        resumeAfter: resumeToken // For resumable processing
    }
);

changeStream.on('change', (change) => {
    // Process change event
    // Send notifications, update cache, trigger workflows
});
*/

// ==============================================
// BACKUP AND RECOVERY CONFIGURATION
// ==============================================

// Backup configuration (to be used with mongodump)
const backupConfig = {
    schedule: '0 2 * * *', // Daily at 2 AM
    retention_days: 30,
    backup_location: 's3://greenlang-backups/mongodb/',
    databases_to_backup: [
        'greenlang_documents',
        'greenlang_analytics',
        'greenlang_workflows',
        'greenlang_ml'
    ],
    exclude_collections: [
        'greenlang_cache.api_cache',
        'greenlang_iot.device_telemetry' // Backed up separately due to size
    ],
    compression: 'gzip',
    encryption: {
        enabled: true,
        algorithm: 'AES256',
        key_management: 'AWS_KMS'
    }
};

// ==============================================
// MONITORING AND PERFORMANCE VIEWS
// ==============================================

// Create view for monitoring collection sizes
db.getSiblingDB('admin').createView(
    'collection_stats',
    'system.namespaces',
    [
        {
            $match: {
                name: { $regex: '^greenlang' }
            }
        },
        {
            $lookup: {
                from: 'system.indexes',
                localField: 'name',
                foreignField: 'ns',
                as: 'indexes'
            }
        },
        {
            $project: {
                database: { $arrayElemAt: [{ $split: ['$name', '.'] }, 0] },
                collection: { $arrayElemAt: [{ $split: ['$name', '.'] }, 1] },
                index_count: { $size: '$indexes' }
            }
        }
    ]
);

// ==============================================
// SECURITY AND ACCESS CONTROL
// ==============================================

// Create application users with specific roles
db.getSiblingDB('admin').createUser({
    user: 'greenlang_app',
    pwd: 'secure_password_from_vault',
    roles: [
        { role: 'readWrite', db: 'greenlang_documents' },
        { role: 'readWrite', db: 'greenlang_analytics' },
        { role: 'readWrite', db: 'greenlang_workflows' },
        { role: 'readWrite', db: 'greenlang_cache' },
        { role: 'read', db: 'greenlang_ml' }
    ]
});

db.getSiblingDB('admin').createUser({
    user: 'greenlang_analytics',
    pwd: 'secure_password_from_vault',
    roles: [
        { role: 'read', db: 'greenlang_documents' },
        { role: 'read', db: 'greenlang_analytics' },
        { role: 'read', db: 'greenlang_iot' }
    ]
});

// Enable authentication and encryption
/*
mongod --auth --tlsMode requireTLS \
       --tlsCertificateKeyFile /path/to/mongodb.pem \
       --tlsCAFile /path/to/ca.pem \
       --enableEncryption \
       --encryptionKeyFile /path/to/keyfile
*/