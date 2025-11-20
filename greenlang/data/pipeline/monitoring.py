"""
Data Quality Monitoring System

Comprehensive monitoring for emission factors data quality:
- Coverage gap analysis
- Source diversity tracking
- Geographic coverage heatmap
- Data freshness tracking
- Quality metrics dashboard
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging
from collections import defaultdict

from .models import DataQualityMetrics

logger = logging.getLogger(__name__)


class CoverageAnalyzer:
    """
    Analyze data coverage gaps.

    Identifies which sectors, regions, and categories need more factors.
    """

    def __init__(self, db_path: str):
        """
        Initialize coverage analyzer.

        Args:
            db_path: Path to database
        """
        self.db_path = db_path

    def analyze_category_coverage(self) -> Dict[str, Any]:
        """
        Analyze coverage by category.

        Returns:
            Dictionary with coverage statistics by category
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get factor counts by category
        cursor.execute("""
            SELECT category, subcategory, COUNT(*) as count
            FROM emission_factors
            GROUP BY category, subcategory
            ORDER BY category, count DESC
        """)

        coverage = defaultdict(lambda: {'total': 0, 'subcategories': {}})

        for category, subcategory, count in cursor.fetchall():
            coverage[category]['total'] += count
            coverage[category]['subcategories'][subcategory or 'general'] = count

        # Identify gaps (categories with <10 factors)
        gaps = {}
        for category, data in coverage.items():
            if data['total'] < 10:
                gaps[category] = {
                    'current_count': data['total'],
                    'recommended_min': 10,
                    'gap': 10 - data['total']
                }

        # Identify subcategories needing more factors
        subcategory_gaps = []
        for category, data in coverage.items():
            for subcat, count in data['subcategories'].items():
                if count < 3:
                    subcategory_gaps.append({
                        'category': category,
                        'subcategory': subcat,
                        'current_count': count,
                        'recommended_min': 3
                    })

        conn.close()

        return {
            'coverage_by_category': dict(coverage),
            'category_gaps': gaps,
            'subcategory_gaps': subcategory_gaps,
            'total_categories': len(coverage),
            'categories_below_threshold': len(gaps)
        }

    def analyze_geographic_coverage(self) -> Dict[str, Any]:
        """
        Analyze geographic coverage.

        Returns:
            Dictionary with geographic coverage statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Coverage by geography level
        cursor.execute("""
            SELECT
                geography_level,
                COUNT(*) as count,
                COUNT(DISTINCT category) as categories
            FROM emission_factors
            WHERE geography_level IS NOT NULL
            GROUP BY geography_level
            ORDER BY count DESC
        """)

        by_level = {}
        for level, count, categories in cursor.fetchall():
            by_level[level] = {
                'factor_count': count,
                'category_count': categories
            }

        # Coverage by country
        cursor.execute("""
            SELECT
                country_code,
                geographic_scope,
                COUNT(*) as count,
                COUNT(DISTINCT category) as categories
            FROM emission_factors
            WHERE country_code IS NOT NULL
            GROUP BY country_code, geographic_scope
            ORDER BY count DESC
        """)

        by_country = {}
        for country, scope, count, categories in cursor.fetchall():
            if country not in by_country:
                by_country[country] = []

            by_country[country].append({
                'scope': scope,
                'factor_count': count,
                'category_count': categories
            })

        # Global vs regional factors
        cursor.execute("""
            SELECT
                CASE
                    WHEN geography_level = 'Global' THEN 'Global'
                    ELSE 'Regional/Local'
                END as level_type,
                COUNT(*) as count
            FROM emission_factors
            GROUP BY level_type
        """)

        global_vs_regional = dict(cursor.fetchall())

        # Geographic coverage gaps
        expected_countries = ['US', 'UK', 'EU', 'CN', 'IN', 'JP', 'AU', 'CA']
        missing_countries = [
            country for country in expected_countries
            if country not in by_country
        ]

        conn.close()

        return {
            'by_geography_level': by_level,
            'by_country': by_country,
            'global_vs_regional': global_vs_regional,
            'unique_countries': len(by_country),
            'missing_key_countries': missing_countries,
            'total_country_specific_factors': sum(
                sum(item['factor_count'] for item in items)
                for items in by_country.values()
            )
        }

    def analyze_scope_coverage(self) -> Dict[str, Any]:
        """
        Analyze coverage by GHG scope.

        Returns:
            Dictionary with scope coverage statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Coverage by scope
        cursor.execute("""
            SELECT
                scope,
                COUNT(*) as count,
                COUNT(DISTINCT category) as categories
            FROM emission_factors
            WHERE scope IS NOT NULL
            GROUP BY scope
            ORDER BY count DESC
        """)

        by_scope = {}
        for scope, count, categories in cursor.fetchall():
            by_scope[scope] = {
                'factor_count': count,
                'category_count': categories
            }

        # Scope distribution
        total_factors = sum(data['factor_count'] for data in by_scope.values())
        for scope, data in by_scope.items():
            data['percentage'] = (data['factor_count'] / total_factors * 100) if total_factors > 0 else 0

        conn.close()

        return {
            'by_scope': by_scope,
            'total_factors': total_factors
        }

    def generate_coverage_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive coverage report.

        Returns:
            Complete coverage analysis report
        """
        logger.info("Generating coverage report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'category_coverage': self.analyze_category_coverage(),
            'geographic_coverage': self.analyze_geographic_coverage(),
            'scope_coverage': self.analyze_scope_coverage()
        }

        # Generate recommendations
        recommendations = []

        # Category recommendations
        cat_gaps = report['category_coverage']['category_gaps']
        if cat_gaps:
            recommendations.append(
                f"Add more factors for {len(cat_gaps)} categories with insufficient coverage"
            )

        # Geographic recommendations
        missing_countries = report['geographic_coverage']['missing_key_countries']
        if missing_countries:
            recommendations.append(
                f"Add country-specific factors for: {', '.join(missing_countries)}"
            )

        # Scope recommendations
        scope_coverage = report['scope_coverage']['by_scope']
        if 'Scope 3' in scope_coverage:
            scope3_pct = scope_coverage['Scope 3']['percentage']
            if scope3_pct < 40:
                recommendations.append(
                    f"Increase Scope 3 factor coverage (currently {scope3_pct:.1f}%)"
                )

        report['recommendations'] = recommendations

        logger.info(f"Coverage report generated: {len(recommendations)} recommendations")

        return report


class SourceDiversityAnalyzer:
    """
    Analyze source diversity.

    Tracks diversity of data sources to avoid over-reliance on single sources.
    """

    def __init__(self, db_path: str):
        """
        Initialize source diversity analyzer.

        Args:
            db_path: Path to database
        """
        self.db_path = db_path

    def analyze_source_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of emission factors by source.

        Returns:
            Source distribution statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Factors per source
        cursor.execute("""
            SELECT
                source_org,
                COUNT(*) as count,
                COUNT(DISTINCT category) as categories,
                MIN(last_updated) as oldest,
                MAX(last_updated) as newest
            FROM emission_factors
            WHERE source_org IS NOT NULL
            GROUP BY source_org
            ORDER BY count DESC
        """)

        sources = {}
        total_factors = 0

        for source, count, categories, oldest, newest in cursor.fetchall():
            sources[source] = {
                'factor_count': count,
                'category_count': categories,
                'oldest_factor': oldest,
                'newest_factor': newest
            }
            total_factors += count

        # Calculate diversity metrics
        unique_sources = len(sources)

        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        # HHI = sum of squared market shares (0 = perfect diversity, 10000 = monopoly)
        hhi = sum(
            ((count / total_factors) * 100) ** 2
            for count in (s['factor_count'] for s in sources.values())
        ) if total_factors > 0 else 0

        # Diversity score (0-100, higher is better)
        # Inverse of HHI normalized
        diversity_score = max(0, 100 - (hhi / 100))

        # Top 3 sources concentration
        top_3_count = sum(
            sorted([s['factor_count'] for s in sources.values()], reverse=True)[:3]
        )
        top_3_concentration = (top_3_count / total_factors * 100) if total_factors > 0 else 0

        conn.close()

        return {
            'sources': sources,
            'unique_sources': unique_sources,
            'total_factors': total_factors,
            'diversity_score': round(diversity_score, 2),
            'hhi': round(hhi, 2),
            'top_3_concentration_pct': round(top_3_concentration, 2)
        }

    def identify_source_gaps(self) -> List[str]:
        """
        Identify recommended sources not yet included.

        Returns:
            List of recommended sources
        """
        # Authoritative emission factor sources
        recommended_sources = {
            'EPA', 'IPCC', 'DEFRA', 'IEA', 'Ecoinvent',
            'GHG Protocol', 'ISO', 'NREL', 'Carbon Trust',
            'UK Government', 'European Commission'
        }

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT source_org FROM emission_factors")
        current_sources = {row[0] for row in cursor.fetchall()}

        conn.close()

        # Find missing sources (fuzzy match)
        missing_sources = []
        for recommended in recommended_sources:
            found = any(
                recommended.lower() in current.lower()
                for current in current_sources
            )
            if not found:
                missing_sources.append(recommended)

        return missing_sources


class FreshnessTracker:
    """
    Track data freshness.

    Monitors age of emission factors and identifies stale data.
    """

    def __init__(self, db_path: str, stale_threshold_years: int = 3):
        """
        Initialize freshness tracker.

        Args:
            db_path: Path to database
            stale_threshold_years: Years after which data is considered stale
        """
        self.db_path = db_path
        self.stale_threshold_years = stale_threshold_years

    def analyze_freshness(self) -> Dict[str, Any]:
        """
        Analyze data freshness.

        Returns:
            Freshness statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Age distribution
        cursor.execute("""
            SELECT
                factor_id,
                name,
                category,
                last_updated,
                CAST((julianday('now') - julianday(last_updated)) / 365.25 AS INTEGER) as age_years
            FROM emission_factors
            ORDER BY last_updated DESC
        """)

        age_distribution = {
            'fresh': [],  # <1 year
            'recent': [],  # 1-2 years
            'aging': [],  # 2-3 years
            'stale': []  # >3 years
        }

        for factor_id, name, category, last_updated, age_years in cursor.fetchall():
            factor_info = {
                'factor_id': factor_id,
                'name': name,
                'category': category,
                'last_updated': last_updated,
                'age_years': age_years
            }

            if age_years < 1:
                age_distribution['fresh'].append(factor_info)
            elif age_years < 2:
                age_distribution['recent'].append(factor_info)
            elif age_years < 3:
                age_distribution['aging'].append(factor_info)
            else:
                age_distribution['stale'].append(factor_info)

        # Calculate freshness score (0-100)
        total = sum(len(factors) for factors in age_distribution.values())

        if total > 0:
            freshness_score = (
                (len(age_distribution['fresh']) * 1.0 +
                 len(age_distribution['recent']) * 0.8 +
                 len(age_distribution['aging']) * 0.5 +
                 len(age_distribution['stale']) * 0.0) / total * 100
            )
        else:
            freshness_score = 0.0

        # Average age
        cursor.execute("""
            SELECT AVG(julianday('now') - julianday(last_updated)) / 365.25
            FROM emission_factors
        """)
        avg_age_years = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            'freshness_score': round(freshness_score, 2),
            'average_age_years': round(avg_age_years, 2),
            'distribution': {
                'fresh_count': len(age_distribution['fresh']),
                'recent_count': len(age_distribution['recent']),
                'aging_count': len(age_distribution['aging']),
                'stale_count': len(age_distribution['stale'])
            },
            'stale_factors': age_distribution['stale'][:10],  # Top 10 oldest
            'total_factors': total
        }


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring.

    Combines all monitoring components into unified quality metrics.
    """

    def __init__(self, db_path: str):
        """
        Initialize data quality monitor.

        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.coverage_analyzer = CoverageAnalyzer(db_path)
        self.source_analyzer = SourceDiversityAnalyzer(db_path)
        self.freshness_tracker = FreshnessTracker(db_path)

    def calculate_quality_metrics(self) -> DataQualityMetrics:
        """
        Calculate comprehensive data quality metrics.

        Returns:
            DataQualityMetrics object with all metrics
        """
        logger.info("Calculating data quality metrics...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total factors
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        total_factors = cursor.fetchone()[0]

        # Get coverage analysis
        coverage = self.coverage_analyzer.analyze_category_coverage()
        geo_coverage = self.coverage_analyzer.analyze_geographic_coverage()

        # Get source diversity
        source_diversity = self.source_analyzer.analyze_source_distribution()

        # Get freshness
        freshness = self.freshness_tracker.analyze_freshness()

        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(cursor)

        # Calculate accuracy score
        accuracy_score = self._calculate_accuracy_score(cursor)

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(cursor)

        # Calculate overall quality score (weighted average)
        overall_quality_score = (
            completeness_score * 0.25 +
            accuracy_score * 0.25 +
            freshness['freshness_score'] * 0.25 +
            consistency_score * 0.15 +
            source_diversity['diversity_score'] * 0.10
        )

        # Quality tier distribution
        cursor.execute("""
            SELECT data_quality_tier, COUNT(*)
            FROM emission_factors
            WHERE data_quality_tier IS NOT NULL
            GROUP BY data_quality_tier
        """)
        tier_distribution = dict(cursor.fetchall())

        conn.close()

        metrics = DataQualityMetrics(
            metric_id=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_factors=total_factors,
            overall_quality_score=round(overall_quality_score, 2),
            completeness_score=round(completeness_score, 2),
            accuracy_score=round(accuracy_score, 2),
            freshness_score=round(freshness['freshness_score'], 2),
            consistency_score=round(consistency_score, 2),
            stale_factors_count=freshness['distribution']['stale_count'],
            avg_age_days=freshness['average_age_years'] * 365,
            oldest_factor_days=int(freshness['average_age_years'] * 365),
            unique_sources=source_diversity['unique_sources'],
            source_distribution={
                source: data['factor_count']
                for source, data in source_diversity['sources'].items()
            },
            geographic_coverage=geo_coverage['by_country'],
            category_coverage=coverage['coverage_by_category'],
            tier_distribution=tier_distribution,
            coverage_metrics={
                'category_gaps': len(coverage['category_gaps']),
                'subcategory_gaps': len(coverage['subcategory_gaps']),
                'missing_countries': len(geo_coverage['missing_key_countries'])
            }
        )

        logger.info(f"Quality metrics calculated: Overall score {metrics.overall_quality_score}/100")

        return metrics

    def _calculate_completeness_score(self, cursor) -> float:
        """Calculate completeness score (0-100)."""
        # Check for missing optional but important fields
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN standard IS NOT NULL THEN 1 ELSE 0 END) as has_standard,
                SUM(CASE WHEN data_quality_tier IS NOT NULL THEN 1 ELSE 0 END) as has_quality,
                SUM(CASE WHEN uncertainty_percent IS NOT NULL THEN 1 ELSE 0 END) as has_uncertainty,
                SUM(CASE WHEN geographic_scope IS NOT NULL THEN 1 ELSE 0 END) as has_geography
            FROM emission_factors
        """)

        total, has_standard, has_quality, has_uncertainty, has_geography = cursor.fetchone()

        if total == 0:
            return 0.0

        # Calculate completeness as average of field coverage
        completeness = (
            (has_standard / total) * 0.25 +
            (has_quality / total) * 0.25 +
            (has_uncertainty / total) * 0.25 +
            (has_geography / total) * 0.25
        ) * 100

        return completeness

    def _calculate_accuracy_score(self, cursor) -> float:
        """Calculate accuracy score (0-100)."""
        # Check for invalid values
        cursor.execute("""
            SELECT COUNT(*) FROM emission_factors
            WHERE emission_factor_value <= 0
        """)
        invalid_values = cursor.fetchone()[0]

        # Check for extreme outliers (>10000)
        cursor.execute("""
            SELECT COUNT(*) FROM emission_factors
            WHERE emission_factor_value > 10000
        """)
        extreme_values = cursor.fetchone()[0]

        # Get total
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        total = cursor.fetchone()[0]

        if total == 0:
            return 0.0

        # Accuracy is based on valid values
        accuracy = ((total - invalid_values - extreme_values) / total) * 100

        return accuracy

    def _calculate_consistency_score(self, cursor) -> float:
        """Calculate consistency score (0-100)."""
        # Check for duplicate factor IDs (should be 0 due to PRIMARY KEY)
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT factor_id, COUNT(*) as count
                FROM emission_factors
                GROUP BY factor_id
                HAVING count > 1
            )
        """)
        duplicates = cursor.fetchone()[0]

        # Check for inconsistent units within categories
        # (This is complex, simplified version)
        cursor.execute("SELECT COUNT(*) FROM emission_factors")
        total = cursor.fetchone()[0]

        if total == 0:
            return 0.0

        # Consistency is high if no duplicates
        consistency = ((total - duplicates) / total) * 100

        return consistency

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Returns:
            Complete monitoring report with all metrics
        """
        logger.info("Generating monitoring report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'quality_metrics': self.calculate_quality_metrics().dict(),
            'coverage_report': self.coverage_analyzer.generate_coverage_report(),
            'source_analysis': self.source_analyzer.analyze_source_distribution(),
            'freshness_analysis': self.freshness_tracker.analyze_freshness(),
            'recommended_actions': self._generate_recommendations()
        }

        logger.info("Monitoring report generated")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Get metrics
        metrics = self.calculate_quality_metrics()

        # Quality score recommendations
        if metrics.overall_quality_score < 70:
            recommendations.append(
                f"Overall quality score is {metrics.overall_quality_score:.1f}/100. "
                "Review data quality issues."
            )

        # Completeness recommendations
        if metrics.completeness_score < 60:
            recommendations.append(
                "Improve data completeness by adding missing standard and quality tier fields"
            )

        # Freshness recommendations
        if metrics.stale_factors_count > metrics.total_factors * 0.2:
            recommendations.append(
                f"Update {metrics.stale_factors_count} stale factors (>3 years old)"
            )

        # Source diversity recommendations
        if metrics.unique_sources < 5:
            recommendations.append(
                f"Increase source diversity (currently {metrics.unique_sources} sources)"
            )

        return recommendations
