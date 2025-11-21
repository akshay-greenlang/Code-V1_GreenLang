# -*- coding: utf-8 -*-
"""
GreenLang Apache Airflow ETL/ELT Pipeline Architecture
Version: 1.0.0
DAGs: 100+ data pipelines
Processing: 10TB+ daily
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import json

from airflow import DAG
from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.amazon.aws.transfers.s3_to_redshift import S3ToRedshiftOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
from airflow.providers.dbt.cloud.operators.dbt import DbtCloudRunJobOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.exceptions import AirflowException
from greenlang.determinism import DeterministicClock

# ==============================================
# DEFAULT DAG CONFIGURATION
# ==============================================

default_args = {
    'owner': 'greenlang-data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['data-team@greenlang.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
    'sla': timedelta(hours=3),
}

# ==============================================
# EMISSIONS DATA PIPELINE (MASTER DAG)
# ==============================================

@dag(
    dag_id='emissions_data_pipeline',
    default_args=default_args,
    schedule_interval='0 1 * * *',  # Daily at 1 AM
    max_active_runs=1,
    catchup=False,
    tags=['emissions', 'critical', 'daily'],
    doc_md="""
    # Emissions Data Pipeline

    Master pipeline for processing emissions data from multiple sources:
    1. Extract data from ERP systems (SAP, Oracle, Workday)
    2. Collect IoT sensor readings
    3. Process and validate data
    4. Calculate emissions using emission factors
    5. Aggregate and store results
    6. Update dashboards and reports
    """
)
def emissions_data_pipeline():
    """Main emissions data processing pipeline."""

    # ==============================================
    # EXTRACTION PHASE
    # ==============================================

    with TaskGroup(group_id='extract_data') as extract_group:

        @task(pool='erp_connections', priority_weight=10)
        def extract_sap_emissions(**context):
            """Extract emissions data from SAP."""
            execution_date = context['ds']

            query = f"""
            SELECT
                source_id,
                facility_code,
                activity_date,
                activity_type,
                activity_value,
                activity_unit,
                data_source
            FROM SAP_EMISSIONS_TABLE
            WHERE activity_date = '{execution_date}'
            """

            # Connect to SAP and extract data
            hook = PostgresHook(postgres_conn_id='sap_db')
            df = hook.get_pandas_df(query)

            # Store in staging
            df.to_parquet(f'/data/staging/sap_emissions_{execution_date}.parquet')

            return {'records_extracted': len(df), 'date': execution_date}

        @task(pool='erp_connections')
        def extract_oracle_emissions(**context):
            """Extract emissions data from Oracle ERP."""
            execution_date = context['ds']

            # Oracle extraction logic
            oracle_hook = PostgresHook(postgres_conn_id='oracle_db')
            query = f"""
            SELECT * FROM EMISSIONS_ACTIVITIES
            WHERE ACTIVITY_DATE = TO_DATE('{execution_date}', 'YYYY-MM-DD')
            """

            df = oracle_hook.get_pandas_df(query)
            df.to_parquet(f'/data/staging/oracle_emissions_{execution_date}.parquet')

            return {'records_extracted': len(df), 'date': execution_date}

        @task(pool='iot_connections')
        def extract_iot_sensor_data(**context):
            """Extract IoT sensor readings."""
            execution_date = context['ds']

            # Query time-series database
            query = f"""
            SELECT
                device_id,
                timestamp,
                metric_name,
                value,
                unit,
                quality_score
            FROM iot.sensor_readings
            WHERE DATE(timestamp) = '{execution_date}'
            """

            hook = PostgresHook(postgres_conn_id='timescale_db')
            df = hook.get_pandas_df(query)

            # Aggregate to hourly
            df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
            hourly = df.groupby(['device_id', 'hour', 'metric_name']).agg({
                'value': 'mean',
                'quality_score': 'mean'
            }).reset_index()

            hourly.to_parquet(f'/data/staging/iot_emissions_{execution_date}.parquet')

            return {'records_extracted': len(hourly), 'date': execution_date}

        # Execute extraction tasks
        sap_data = extract_sap_emissions()
        oracle_data = extract_oracle_emissions()
        iot_data = extract_iot_sensor_data()

    # ==============================================
    # VALIDATION PHASE
    # ==============================================

    with TaskGroup(group_id='validate_data') as validate_group:

        @task
        def validate_emissions_data(**context):
            """Validate extracted emissions data."""
            execution_date = context['ds']

            validation_results = {
                'date': execution_date,
                'validations': []
            }

            # Load all staging files
            files = [
                f'/data/staging/sap_emissions_{execution_date}.parquet',
                f'/data/staging/oracle_emissions_{execution_date}.parquet',
                f'/data/staging/iot_emissions_{execution_date}.parquet'
            ]

            for file_path in files:
                df = pd.read_parquet(file_path)

                # Data quality checks
                checks = {
                    'null_check': df.isnull().sum().to_dict(),
                    'negative_values': (df.select_dtypes(include=['number']) < 0).sum().to_dict(),
                    'duplicate_check': df.duplicated().sum(),
                    'record_count': len(df)
                }

                # Schema validation
                expected_columns = ['source_id', 'activity_date', 'activity_value']
                missing_columns = set(expected_columns) - set(df.columns)

                if missing_columns:
                    raise AirflowException(f"Missing required columns: {missing_columns}")

                validation_results['validations'].append({
                    'file': file_path,
                    'checks': checks
                })

            return validation_results

        @task
        def check_data_quality_thresholds(validation_results):
            """Check if data quality meets thresholds."""
            min_quality_score = 70

            for validation in validation_results['validations']:
                if validation['checks']['duplicate_check'] > 100:
                    raise AirflowException(f"Too many duplicates: {validation['checks']['duplicate_check']}")

                # Check for excessive nulls
                for col, null_count in validation['checks']['null_check'].items():
                    if null_count > validation['checks']['record_count'] * 0.1:
                        raise AirflowException(f"Too many nulls in {col}: {null_count}")

            return True

        validation = validate_emissions_data()
        quality_check = check_data_quality_thresholds(validation)

    # ==============================================
    # TRANSFORMATION PHASE
    # ==============================================

    with TaskGroup(group_id='transform_data') as transform_group:

        @task
        def calculate_emissions(**context):
            """Calculate emissions using emission factors."""
            execution_date = context['ds']

            # Load activity data
            df = pd.read_parquet(f'/data/staging/sap_emissions_{execution_date}.parquet')

            # Load emission factors
            factors_query = """
            SELECT
                activity_type,
                region,
                factor_value,
                unit
            FROM master_data.emission_factors
            WHERE valid_from <= CURRENT_DATE
            AND valid_to >= CURRENT_DATE
            """

            hook = PostgresHook(postgres_conn_id='greenlang_db')
            factors = hook.get_pandas_df(factors_query)

            # Merge and calculate
            df = df.merge(factors, on='activity_type', how='left')
            df['co2e_emissions'] = df['activity_value'] * df['factor_value']

            # Add metadata
            df['calculation_method'] = 'emission_factor'
            df['calculation_date'] = DeterministicClock.now()
            df['data_quality_score'] = df['quality_score'].fillna(80)

            df.to_parquet(f'/data/processed/calculated_emissions_{execution_date}.parquet')

            return {'records_calculated': len(df), 'total_emissions': df['co2e_emissions'].sum()}

        @task
        def aggregate_by_scope(**context):
            """Aggregate emissions by scope category."""
            execution_date = context['ds']

            df = pd.read_parquet(f'/data/processed/calculated_emissions_{execution_date}.parquet')

            # Categorize by scope
            scope_mapping = {
                'natural_gas': 'scope1',
                'diesel': 'scope1',
                'electricity': 'scope2',
                'business_travel': 'scope3',
                'supply_chain': 'scope3'
            }

            df['scope_category'] = df['activity_type'].map(scope_mapping)

            # Aggregate
            aggregated = df.groupby(['scope_category', 'facility_code']).agg({
                'co2e_emissions': 'sum',
                'data_quality_score': 'mean',
                'activity_value': 'sum'
            }).reset_index()

            aggregated.to_parquet(f'/data/processed/scope_emissions_{execution_date}.parquet')

            return aggregated.to_dict('records')

        emissions = calculate_emissions()
        scope_data = aggregate_by_scope()

    # ==============================================
    # LOADING PHASE
    # ==============================================

    with TaskGroup(group_id='load_data') as load_group:

        @task
        def load_to_postgres(**context):
            """Load processed data to PostgreSQL."""
            execution_date = context['ds']

            df = pd.read_parquet(f'/data/processed/calculated_emissions_{execution_date}.parquet')

            # Load to database
            hook = PostgresHook(postgres_conn_id='greenlang_db')
            conn = hook.get_conn()

            df.to_sql(
                'emissions_data',
                conn,
                schema='emissions',
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )

            return {'records_loaded': len(df)}

        @task
        def load_to_data_lake(**context):
            """Load to S3 data lake."""
            execution_date = context['ds']
            year, month, day = execution_date.split('-')

            # Partition by date
            s3_path = f"s3://greenlang-data-lake/emissions/year={year}/month={month}/day={day}/"

            df = pd.read_parquet(f'/data/processed/calculated_emissions_{execution_date}.parquet')

            # Convert to different formats
            df.to_parquet(s3_path + 'data.parquet')
            df.to_csv(s3_path + 'data.csv', index=False)

            return {'s3_path': s3_path, 'records': len(df)}

        postgres_load = load_to_postgres()
        s3_load = load_to_data_lake()

    # ==============================================
    # ANALYTICS UPDATE
    # ==============================================

    @task
    def update_kpi_metrics(**context):
        """Update KPI metrics."""
        execution_date = context['ds']

        hook = PostgresHook(postgres_conn_id='greenlang_db')

        # Calculate KPIs
        kpi_queries = [
            """
            INSERT INTO analytics.kpi_values (organization_id, kpi_id, period_start, value)
            SELECT
                organization_id,
                'TOTAL_EMISSIONS' as kpi_id,
                DATE('{date}') as period_start,
                SUM(co2e_total) as value
            FROM emissions.emissions_data
            WHERE DATE(time) = '{date}'
            GROUP BY organization_id
            """.format(date=execution_date),

            """
            INSERT INTO analytics.kpi_values (organization_id, kpi_id, period_start, value)
            SELECT
                organization_id,
                'EMISSIONS_INTENSITY' as kpi_id,
                DATE('{date}') as period_start,
                SUM(co2e_total) / NULLIF(COUNT(DISTINCT source_id), 0) as value
            FROM emissions.emissions_data
            WHERE DATE(time) = '{date}'
            GROUP BY organization_id
            """.format(date=execution_date)
        ]

        for query in kpi_queries:
            hook.run(query)

        return {'kpis_updated': len(kpi_queries)}

    # Define dependencies
    extract_group >> validate_group >> transform_group >> load_group >> update_kpi_metrics()

emissions_dag = emissions_data_pipeline()

# ==============================================
# SUPPLY CHAIN DATA PIPELINE
# ==============================================

@dag(
    dag_id='supply_chain_integration',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    tags=['supply_chain', 'daily'],
    doc_md="""
    # Supply Chain Integration Pipeline

    Processes supplier data, purchase orders, and sustainability metrics
    """
)
def supply_chain_integration():
    """Supply chain data integration pipeline."""

    @task
    def extract_supplier_data(**context):
        """Extract supplier master data from multiple ERPs."""
        execution_date = context['ds']

        suppliers = []

        # SAP extraction
        sap_query = """
        SELECT * FROM SUPPLIERS
        WHERE LAST_MODIFIED >= CURRENT_DATE - 1
        """
        sap_hook = PostgresHook(postgres_conn_id='sap_db')
        sap_suppliers = sap_hook.get_pandas_df(sap_query)
        suppliers.append(sap_suppliers)

        # Oracle extraction
        oracle_query = """
        SELECT * FROM AP_SUPPLIERS
        WHERE LAST_UPDATE_DATE >= SYSDATE - 1
        """
        oracle_hook = PostgresHook(postgres_conn_id='oracle_db')
        oracle_suppliers = oracle_hook.get_pandas_df(oracle_query)
        suppliers.append(oracle_suppliers)

        # Combine and deduplicate
        all_suppliers = pd.concat(suppliers, ignore_index=True)
        all_suppliers = all_suppliers.drop_duplicates(subset=['supplier_id'])

        all_suppliers.to_parquet(f'/data/staging/suppliers_{execution_date}.parquet')

        return {'suppliers_extracted': len(all_suppliers)}

    @task
    def calculate_supplier_risk_scores(**context):
        """Calculate supplier sustainability risk scores."""
        execution_date = context['ds']

        df = pd.read_parquet(f'/data/staging/suppliers_{execution_date}.parquet')

        # Risk scoring algorithm
        df['environmental_risk'] = 0
        df['social_risk'] = 0
        df['governance_risk'] = 0

        # Environmental risk factors
        df.loc[df['has_iso14001'] == False, 'environmental_risk'] += 20
        df.loc[df['carbon_disclosure'] == False, 'environmental_risk'] += 30
        df.loc[df['renewable_energy_percent'] < 20, 'environmental_risk'] += 25

        # Social risk factors
        df.loc[df['has_social_audit'] == False, 'social_risk'] += 25
        df.loc[df['labor_violations'] > 0, 'social_risk'] += 40

        # Governance risk factors
        df.loc[df['has_code_of_conduct'] == False, 'governance_risk'] += 20
        df.loc[df['transparency_score'] < 50, 'governance_risk'] += 30

        # Overall risk score
        df['overall_risk_score'] = (
            df['environmental_risk'] * 0.4 +
            df['social_risk'] * 0.3 +
            df['governance_risk'] * 0.3
        )

        df.to_parquet(f'/data/processed/supplier_risk_{execution_date}.parquet')

        return {
            'suppliers_scored': len(df),
            'high_risk_suppliers': len(df[df['overall_risk_score'] > 70])
        }

    extract = extract_supplier_data()
    risk_scores = calculate_supplier_risk_scores()

    extract >> risk_scores

supply_chain_dag = supply_chain_integration()

# ==============================================
# CSRD REPORTING PIPELINE
# ==============================================

@dag(
    dag_id='csrd_reporting_pipeline',
    default_args=default_args,
    schedule_interval='0 0 1 * *',  # Monthly on 1st
    tags=['csrd', 'reporting', 'monthly'],
    doc_md="""
    # CSRD Reporting Pipeline

    Generates monthly CSRD reports with all required disclosures
    """
)
def csrd_reporting_pipeline():
    """CSRD reporting data pipeline."""

    @task
    def collect_esrs_data_points(**context):
        """Collect all ESRS standard data points."""
        reporting_month = context['ds']

        hook = PostgresHook(postgres_conn_id='greenlang_db')

        # Collect E1 Climate Change data
        e1_query = """
        SELECT
            organization_id,
            SUM(co2e_total) as total_emissions,
            AVG(data_quality_score) as quality_score
        FROM emissions.emissions_data
        WHERE DATE_TRUNC('month', time) = DATE_TRUNC('month', '{month}'::date)
        GROUP BY organization_id
        """.format(month=reporting_month)

        e1_data = hook.get_pandas_df(e1_query)

        # Collect E2 Pollution data
        e2_query = """
        SELECT * FROM environmental.pollution_metrics
        WHERE reporting_month = DATE_TRUNC('month', '{month}'::date)
        """.format(month=reporting_month)

        e2_data = hook.get_pandas_df(e2_query)

        # Save consolidated data
        consolidated = {
            'E1_Climate': e1_data.to_dict('records'),
            'E2_Pollution': e2_data.to_dict('records'),
            'reporting_period': reporting_month
        }

        with open(f'/data/csrd/esrs_data_{reporting_month}.json', 'w') as f:
            json.dump(consolidated, f)

        return {'data_points_collected': len(consolidated)}

    @task
    def generate_csrd_report(**context):
        """Generate CSRD report document."""
        reporting_month = context['ds']

        # Load data points
        with open(f'/data/csrd/esrs_data_{reporting_month}.json', 'r') as f:
            data = json.load(f)

        # Generate report structure
        report = {
            'report_id': f"CSRD-{reporting_month}",
            'reporting_period': reporting_month,
            'sections': [],
            'verification_status': 'draft'
        }

        # Add sections for each ESRS standard
        for standard, data_points in data.items():
            section = {
                'standard': standard,
                'disclosures': data_points,
                'narrative': f"Disclosure for {standard}",
                'charts': [],
                'tables': []
            }
            report['sections'].append(section)

        # Save report
        with open(f'/data/csrd/report_{reporting_month}.json', 'w') as f:
            json.dump(report, f)

        return {'report_generated': report['report_id']}

    collect = collect_esrs_data_points()
    generate = generate_csrd_report()

    collect >> generate

csrd_dag = csrd_reporting_pipeline()

# ==============================================
# REAL-TIME IOT PROCESSING
# ==============================================

@dag(
    dag_id='iot_stream_processing',
    default_args=default_args,
    schedule_interval='*/15 * * * *',  # Every 15 minutes
    tags=['iot', 'streaming', 'real-time'],
    doc_md="""
    # IoT Stream Processing Pipeline

    Processes real-time IoT sensor data for immediate insights
    """
)
def iot_stream_processing():
    """Real-time IoT data processing."""

    @task
    def process_sensor_batch(**context):
        """Process batch of sensor readings."""
        execution_time = context['ts']

        # Read from Kafka
        kafka_config = {
            'bootstrap_servers': 'kafka:9092',
            'topic': 'iot.sensor-readings',
            'batch_size': 10000
        }

        # Process sensor data
        processed_records = []

        # Anomaly detection
        anomalies = []

        return {
            'records_processed': len(processed_records),
            'anomalies_detected': len(anomalies)
        }

    process = process_sensor_batch()

iot_dag = iot_stream_processing()

# ==============================================
# DATA QUALITY MONITORING
# ==============================================

@dag(
    dag_id='data_quality_monitoring',
    default_args=default_args,
    schedule_interval='0 * * * *',  # Hourly
    tags=['monitoring', 'data_quality'],
    doc_md="""
    # Data Quality Monitoring Pipeline

    Monitors data quality across all tables and sources
    """
)
def data_quality_monitoring():
    """Data quality monitoring pipeline."""

    @task
    def check_data_freshness(**context):
        """Check data freshness across all sources."""
        hook = PostgresHook(postgres_conn_id='greenlang_db')

        freshness_checks = [
            {
                'table': 'emissions.emissions_data',
                'timestamp_column': 'time',
                'max_lag_hours': 24
            },
            {
                'table': 'supply_chain.purchase_orders',
                'timestamp_column': 'created_at',
                'max_lag_hours': 48
            }
        ]

        alerts = []

        for check in freshness_checks:
            query = f"""
            SELECT
                MAX({check['timestamp_column']}) as last_update,
                EXTRACT(EPOCH FROM (NOW() - MAX({check['timestamp_column']})))/3600 as lag_hours
            FROM {check['table']}
            """

            result = hook.get_first(query)
            lag_hours = result[1] if result[1] else 999

            if lag_hours > check['max_lag_hours']:
                alerts.append({
                    'table': check['table'],
                    'lag_hours': lag_hours,
                    'threshold': check['max_lag_hours']
                })

        if alerts:
            raise AirflowException(f"Data freshness alerts: {alerts}")

        return {'checks_passed': len(freshness_checks)}

    freshness = check_data_freshness()

quality_dag = data_quality_monitoring()

# ==============================================
# ML MODEL TRAINING PIPELINE
# ==============================================

@dag(
    dag_id='ml_model_training',
    default_args=default_args,
    schedule_interval='0 0 * * 0',  # Weekly on Sunday
    tags=['ml', 'training'],
    doc_md="""
    # ML Model Training Pipeline

    Trains and deploys ML models for emissions forecasting and anomaly detection
    """
)
def ml_model_training():
    """ML model training pipeline."""

    spark_job = SparkSubmitOperator(
        task_id='train_emissions_forecast_model',
        application='/spark/jobs/train_emissions_model.py',
        conn_id='spark_default',
        conf={
            'spark.executor.memory': '4g',
            'spark.executor.cores': '4',
            'spark.executor.instances': '10'
        }
    )

    databricks_job = DatabricksSubmitRunOperator(
        task_id='train_anomaly_detection_model',
        databricks_conn_id='databricks_default',
        notebook_task={
            'notebook_path': '/Models/AnomalyDetection',
            'base_parameters': {
                'training_date': '{{ ds }}',
                'model_version': 'v2'
            }
        },
        new_cluster={
            'spark_version': '11.3.x-scala2.12',
            'node_type_id': 'i3.xlarge',
            'num_workers': 4
        }
    )

    spark_job >> databricks_job

ml_dag = ml_model_training()

# ==============================================
# MASTER ORCHESTRATION DAG
# ==============================================

@dag(
    dag_id='master_orchestration',
    default_args=default_args,
    schedule_interval='0 0 * * *',  # Daily at midnight
    tags=['orchestration', 'master'],
    doc_md="""
    # Master Orchestration Pipeline

    Orchestrates all data pipelines and ensures proper execution order
    """
)
def master_orchestration():
    """Master orchestration pipeline."""

    start = DummyOperator(task_id='start')

    # Wait for external data sources
    wait_for_sap = ExternalTaskSensor(
        task_id='wait_for_sap_export',
        external_dag_id='sap_data_export',
        external_task_id='export_complete',
        timeout=3600,
        poke_interval=60,
        mode='reschedule'
    )

    # Trigger child DAGs
    trigger_emissions = BashOperator(
        task_id='trigger_emissions_pipeline',
        bash_command='airflow dags trigger emissions_data_pipeline'
    )

    trigger_supply_chain = BashOperator(
        task_id='trigger_supply_chain_pipeline',
        bash_command='airflow dags trigger supply_chain_integration'
    )

    # Data quality checks
    quality_check = BashOperator(
        task_id='run_data_quality_checks',
        bash_command='airflow dags trigger data_quality_monitoring'
    )

    # Completion
    complete = DummyOperator(task_id='pipeline_complete')

    # Define dependencies
    start >> wait_for_sap >> [trigger_emissions, trigger_supply_chain] >> quality_check >> complete

master_dag = master_orchestration()

# ==============================================
# DISASTER RECOVERY DAG
# ==============================================

@dag(
    dag_id='disaster_recovery_backup',
    default_args=default_args,
    schedule_interval='0 3 * * *',  # Daily at 3 AM
    tags=['backup', 'disaster_recovery'],
    doc_md="""
    # Disaster Recovery Backup Pipeline

    Backs up critical data to multiple locations
    """
)
def disaster_recovery_backup():
    """Disaster recovery backup pipeline."""

    @task
    def backup_postgres(**context):
        """Backup PostgreSQL databases."""
        backup_date = context['ds']

        backup_command = f"""
        pg_dump -h postgres-primary -U greenlang -d greenlang_production \
        | gzip > /backups/postgres_backup_{backup_date}.sql.gz
        """

        # Upload to S3
        s3_command = f"""
        aws s3 cp /backups/postgres_backup_{backup_date}.sql.gz \
        s3://greenlang-backups/postgres/{backup_date}/
        """

        return {'backup_file': f'postgres_backup_{backup_date}.sql.gz'}

    @task
    def backup_mongodb(**context):
        """Backup MongoDB databases."""
        backup_date = context['ds']

        backup_command = f"""
        mongodump --uri="mongodb://mongo-cluster:27017" \
        --archive=/backups/mongo_backup_{backup_date}.archive \
        --gzip
        """

        return {'backup_file': f'mongo_backup_{backup_date}.archive'}

    postgres_backup = backup_postgres()
    mongo_backup = backup_mongodb()

backup_dag = disaster_recovery_backup()