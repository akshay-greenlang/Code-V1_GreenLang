# -*- coding: utf-8 -*-
"""
Training Metrics Tracking System

Tracks and reports on GreenLang training effectiveness:
- Workshop completion rates
- Certification pass rates
- Time to first contribution
- Developer satisfaction
- Code review feedback trends
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from greenlang.determinism import DeterministicClock


class TrainingMetricsTracker:
    """Track training metrics for GreenLang enablement."""

    def __init__(self, db_path: str = "training_metrics.db"):
        """Initialize metrics tracker."""
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Developers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS developers (
                developer_id TEXT PRIMARY KEY,
                name TEXT,
                start_date TEXT,
                team TEXT,
                manager TEXT
            )
        """)

        # Workshop completions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workshop_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id TEXT,
                workshop_number INTEGER,
                completion_date TEXT,
                score INTEGER,
                time_spent_minutes INTEGER,
                FOREIGN KEY (developer_id) REFERENCES developers(developer_id)
            )
        """)

        # Certification attempts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS certification_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id TEXT,
                level INTEGER,
                attempt_number INTEGER,
                attempt_date TEXT,
                score INTEGER,
                passed INTEGER,
                time_spent_minutes INTEGER,
                FOREIGN KEY (developer_id) REFERENCES developers(developer_id)
            )
        """)

        # Code contributions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS code_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id TEXT,
                pr_number INTEGER,
                submission_date TEXT,
                approval_date TEXT,
                uses_infrastructure INTEGER,
                violations_count INTEGER,
                review_feedback TEXT,
                FOREIGN KEY (developer_id) REFERENCES developers(developer_id)
            )
        """)

        # Satisfaction surveys
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS satisfaction_surveys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                developer_id TEXT,
                survey_date TEXT,
                overall_rating INTEGER,
                workshop_rating INTEGER,
                materials_rating INTEGER,
                support_rating INTEGER,
                comments TEXT,
                FOREIGN KEY (developer_id) REFERENCES developers(developer_id)
            )
        """)

        conn.commit()
        conn.close()

    def add_developer(self, developer_id: str, name: str, start_date: str,
                     team: str, manager: str):
        """Register a new developer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO developers (developer_id, name, start_date, team, manager)
            VALUES (?, ?, ?, ?, ?)
        """, (developer_id, name, start_date, team, manager))

        conn.commit()
        conn.close()

    def record_workshop_completion(self, developer_id: str, workshop_number: int,
                                   score: int, time_spent_minutes: int):
        """Record workshop completion."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO workshop_completions
            (developer_id, workshop_number, completion_date, score, time_spent_minutes)
            VALUES (?, ?, ?, ?, ?)
        """, (developer_id, workshop_number, DeterministicClock.now().isoformat(),
              score, time_spent_minutes))

        conn.commit()
        conn.close()

    def record_certification_attempt(self, developer_id: str, level: int,
                                     attempt_number: int, score: int,
                                     time_spent_minutes: int):
        """Record certification attempt."""
        passed = 1 if score >= 80 else 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO certification_attempts
            (developer_id, level, attempt_number, attempt_date, score, passed, time_spent_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (developer_id, level, attempt_number, DeterministicClock.now().isoformat(),
              score, passed, time_spent_minutes))

        conn.commit()
        conn.close()

    def record_code_contribution(self, developer_id: str, pr_number: int,
                                 submission_date: str, approval_date: Optional[str],
                                 uses_infrastructure: bool, violations_count: int,
                                 review_feedback: str):
        """Record code contribution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO code_contributions
            (developer_id, pr_number, submission_date, approval_date,
             uses_infrastructure, violations_count, review_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (developer_id, pr_number, submission_date, approval_date,
              1 if uses_infrastructure else 0, violations_count, review_feedback))

        conn.commit()
        conn.close()

    def record_satisfaction_survey(self, developer_id: str, overall_rating: int,
                                   workshop_rating: int, materials_rating: int,
                                   support_rating: int, comments: str):
        """Record satisfaction survey."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO satisfaction_surveys
            (developer_id, survey_date, overall_rating, workshop_rating,
             materials_rating, support_rating, comments)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (developer_id, DeterministicClock.now().isoformat(), overall_rating,
              workshop_rating, materials_rating, support_rating, comments))

        conn.commit()
        conn.close()

    def get_workshop_completion_rate(self) -> Dict[int, float]:
        """Get completion rate for each workshop."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total developers
        cursor.execute("SELECT COUNT(*) FROM developers")
        total_devs = cursor.fetchone()[0]

        if total_devs == 0:
            return {}

        # Completions per workshop
        completion_rates = {}
        for workshop_num in range(1, 7):
            cursor.execute("""
                SELECT COUNT(DISTINCT developer_id)
                FROM workshop_completions
                WHERE workshop_number = ?
            """, (workshop_num,))

            completed = cursor.fetchone()[0]
            completion_rates[workshop_num] = (completed / total_devs) * 100

        conn.close()
        return completion_rates

    def get_certification_pass_rates(self) -> Dict[int, Dict]:
        """Get pass rates for each certification level."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pass_rates = {}
        for level in range(1, 4):
            cursor.execute("""
                SELECT
                    COUNT(*) as total_attempts,
                    SUM(passed) as total_passed,
                    AVG(score) as avg_score
                FROM certification_attempts
                WHERE level = ?
            """, (level,))

            result = cursor.fetchone()
            total_attempts = result[0]
            total_passed = result[1] or 0
            avg_score = result[2] or 0

            pass_rates[level] = {
                "total_attempts": total_attempts,
                "total_passed": total_passed,
                "pass_rate": (total_passed / total_attempts * 100) if total_attempts > 0 else 0,
                "avg_score": round(avg_score, 1)
            }

        conn.close()
        return pass_rates

    def get_time_to_first_contribution(self) -> Dict:
        """Get average time from start to first contribution."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                d.developer_id,
                d.start_date,
                MIN(c.submission_date) as first_contribution
            FROM developers d
            LEFT JOIN code_contributions c ON d.developer_id = c.developer_id
            WHERE c.uses_infrastructure = 1
            GROUP BY d.developer_id, d.start_date
        """)

        results = cursor.fetchall()
        days_to_contribution = []

        for row in results:
            start = datetime.fromisoformat(row[1])
            first_contrib = datetime.fromisoformat(row[2]) if row[2] else None

            if first_contrib:
                days = (first_contrib - start).days
                days_to_contribution.append(days)

        conn.close()

        if not days_to_contribution:
            return {
                "average_days": 0,
                "median_days": 0,
                "min_days": 0,
                "max_days": 0
            }

        return {
            "average_days": round(sum(days_to_contribution) / len(days_to_contribution), 1),
            "median_days": sorted(days_to_contribution)[len(days_to_contribution) // 2],
            "min_days": min(days_to_contribution),
            "max_days": max(days_to_contribution),
            "sample_size": len(days_to_contribution)
        }

    def get_code_review_feedback_trends(self) -> Dict:
        """Analyze code review feedback trends."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                AVG(uses_infrastructure) * 100 as infrastructure_usage_rate,
                AVG(violations_count) as avg_violations
            FROM code_contributions
        """)

        result = cursor.fetchone()

        conn.close()

        return {
            "infrastructure_usage_rate": round(result[0] or 0, 1),
            "avg_violations_per_pr": round(result[1] or 0, 2)
        }

    def get_satisfaction_scores(self) -> Dict:
        """Get developer satisfaction scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                AVG(overall_rating) as avg_overall,
                AVG(workshop_rating) as avg_workshop,
                AVG(materials_rating) as avg_materials,
                AVG(support_rating) as avg_support,
                COUNT(*) as response_count
            FROM satisfaction_surveys
        """)

        result = cursor.fetchone()

        conn.close()

        return {
            "overall_rating": round(result[0] or 0, 1),
            "workshop_rating": round(result[1] or 0, 1),
            "materials_rating": round(result[2] or 0, 1),
            "support_rating": round(result[3] or 0, 1),
            "response_count": result[4]
        }

    def generate_report(self, output_file: str = "training_report.md"):
        """Generate comprehensive training report."""

        report = []
        report.append("# GreenLang Training Metrics Report")
        report.append(f"\n**Generated:** {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n---\n")

        # Workshop Completion Rates
        report.append("## Workshop Completion Rates\n")
        completion_rates = self.get_workshop_completion_rate()
        for workshop, rate in completion_rates.items():
            report.append(f"- Workshop {workshop}: {rate:.1f}%")

        # Certification Pass Rates
        report.append("\n## Certification Pass Rates\n")
        pass_rates = self.get_certification_pass_rates()
        for level, stats in pass_rates.items():
            report.append(f"\n### Level {level}")
            report.append(f"- Total Attempts: {stats['total_attempts']}")
            report.append(f"- Total Passed: {stats['total_passed']}")
            report.append(f"- Pass Rate: {stats['pass_rate']:.1f}%")
            report.append(f"- Average Score: {stats['avg_score']}")

        # Time to First Contribution
        report.append("\n## Time to First Contribution\n")
        time_stats = self.get_time_to_first_contribution()
        report.append(f"- Average: {time_stats['average_days']} days")
        report.append(f"- Median: {time_stats['median_days']} days")
        report.append(f"- Range: {time_stats['min_days']}-{time_stats['max_days']} days")
        report.append(f"- Sample Size: {time_stats['sample_size']} developers")

        # Code Review Trends
        report.append("\n## Code Review Feedback Trends\n")
        review_stats = self.get_code_review_feedback_trends()
        report.append(f"- Infrastructure Usage Rate: {review_stats['infrastructure_usage_rate']:.1f}%")
        report.append(f"- Average Violations per PR: {review_stats['avg_violations_per_pr']}")

        # Satisfaction Scores
        report.append("\n## Developer Satisfaction\n")
        satisfaction = self.get_satisfaction_scores()
        report.append(f"- Overall Rating: {satisfaction['overall_rating']}/10")
        report.append(f"- Workshop Rating: {satisfaction['workshop_rating']}/10")
        report.append(f"- Materials Rating: {satisfaction['materials_rating']}/10")
        report.append(f"- Support Rating: {satisfaction['support_rating']}/10")
        report.append(f"- Response Count: {satisfaction['response_count']}")

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))

        print(f"Report generated: {output_file}")

        return '\n'.join(report)


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = TrainingMetricsTracker()

    # Add sample data
    tracker.add_developer("dev001", "Alice Smith", "2024-01-15", "CSRD", "Manager1")
    tracker.add_developer("dev002", "Bob Jones", "2024-02-01", "VCCI", "Manager2")

    # Record workshop completions
    tracker.record_workshop_completion("dev001", 1, 95, 120)
    tracker.record_workshop_completion("dev001", 2, 88, 180)
    tracker.record_workshop_completion("dev002", 1, 92, 110)

    # Record certifications
    tracker.record_certification_attempt("dev001", 1, 1, 85, 90)
    tracker.record_certification_attempt("dev002", 1, 1, 78, 95)  # Failed
    tracker.record_certification_attempt("dev002", 1, 2, 82, 85)  # Passed

    # Record contributions
    tracker.record_code_contribution(
        "dev001", 123, "2024-01-20", "2024-01-21",
        True, 0, "Great use of infrastructure!"
    )

    # Record satisfaction
    tracker.record_satisfaction_survey("dev001", 9, 9, 8, 9, "Excellent training!")

    # Generate report
    report = tracker.generate_report()
    print("\n" + report)
