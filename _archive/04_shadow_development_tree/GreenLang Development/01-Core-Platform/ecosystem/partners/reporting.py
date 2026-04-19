# -*- coding: utf-8 -*-
"""
Reporting System for GreenLang Partners

This module provides automated reporting capabilities including:
- Daily usage reports
- Monthly summary reports
- Custom date range reports
- PDF generation with charts
- CSV export for raw data
- Email delivery
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, BinaryIO
from enum import Enum
import logging
import io
import csv

from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import pandas as pd
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Report types"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report formats"""
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    HTML = "html"


class DeliveryMethod(str, Enum):
    """Report delivery methods"""
    EMAIL = "email"
    DOWNLOAD = "download"
    API = "api"


# Pydantic Models
class ReportRequest(BaseModel):
    """Report generation request"""
    partner_id: str
    report_type: ReportType
    start_date: datetime
    end_date: datetime
    format: ReportFormat = ReportFormat.PDF
    delivery_method: DeliveryMethod = DeliveryMethod.EMAIL
    email_to: Optional[EmailStr] = None
    include_charts: bool = True
    include_raw_data: bool = False


class ReportMetadata(BaseModel):
    """Report metadata"""
    report_id: str
    partner_id: str
    report_type: ReportType
    format: ReportFormat
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    file_url: Optional[str]
    file_size_bytes: int


@dataclass
class ReportData:
    """Report data structure"""
    partner_id: str
    partner_name: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time_ms: float
    total_data_mb: float
    top_agents: List[Dict[str, Any]]
    top_endpoints: List[Dict[str, Any]]
    daily_usage: List[Dict[str, Any]]
    error_breakdown: Dict[str, int]
    cost_summary: Dict[str, float]


class ReportGenerator:
    """
    Generates reports in various formats
    """

    def __init__(self, db: Session):
        self.db = db

    def generate_report(
        self,
        partner_id: str,
        start_date: datetime,
        end_date: datetime,
        report_type: ReportType = ReportType.CUSTOM
    ) -> ReportData:
        """
        Generate report data

        Args:
            partner_id: Partner ID
            start_date: Start date
            end_date: End date
            report_type: Report type

        Returns:
            ReportData
        """
        from .api import PartnerModel, UsageRecordModel
        from .analytics import AnalyticsEngine

        # Get partner
        partner = self.db.query(PartnerModel).filter(
            PartnerModel.id == partner_id
        ).first()

        if not partner:
            raise ValueError(f"Partner {partner_id} not found")

        # Get usage records
        usage_records = self.db.query(UsageRecordModel).filter(
            UsageRecordModel.partner_id == partner_id,
            UsageRecordModel.timestamp >= start_date,
            UsageRecordModel.timestamp <= end_date
        ).all()

        # Calculate metrics
        total_requests = len(usage_records)
        successful_requests = sum(1 for r in usage_records if 200 <= r.status_code < 300)
        failed_requests = total_requests - successful_requests
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        avg_response_time = sum(r.response_time_ms for r in usage_records) / total_requests if total_requests > 0 else 0

        total_data_bytes = sum(
            r.request_size_bytes + r.response_size_bytes for r in usage_records
        )
        total_data_mb = total_data_bytes / (1024 * 1024)

        # Top agents
        agent_usage = {}
        for record in usage_records:
            agent_id = record.metadata.get('agent_id')
            if agent_id:
                agent_usage[agent_id] = agent_usage.get(agent_id, 0) + 1

        top_agents = [
            {"agent_id": agent_id, "count": count}
            for agent_id, count in sorted(agent_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Top endpoints
        endpoint_usage = {}
        for record in usage_records:
            endpoint_usage[record.endpoint] = endpoint_usage.get(record.endpoint, 0) + 1

        top_endpoints = [
            {"endpoint": endpoint, "count": count}
            for endpoint, count in sorted(endpoint_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Daily usage
        daily_usage_dict = {}
        for record in usage_records:
            day = record.timestamp.date()
            if day not in daily_usage_dict:
                daily_usage_dict[day] = {'date': day, 'requests': 0, 'errors': 0}
            daily_usage_dict[day]['requests'] += 1
            if record.status_code >= 400:
                daily_usage_dict[day]['errors'] += 1

        daily_usage = sorted(daily_usage_dict.values(), key=lambda x: x['date'])

        # Error breakdown
        error_breakdown = {}
        for record in usage_records:
            if record.status_code >= 400:
                error_breakdown[str(record.status_code)] = error_breakdown.get(str(record.status_code), 0) + 1

        # Cost summary (example pricing)
        cost_per_request = 0.001  # $0.001 per request
        total_cost = total_requests * cost_per_request

        cost_summary = {
            'total_requests': total_requests,
            'cost_per_request': cost_per_request,
            'total_cost': total_cost,
            'currency': 'USD'
        }

        return ReportData(
            partner_id=partner_id,
            partner_name=partner.name,
            period_start=start_date,
            period_end=end_date,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            total_data_mb=total_data_mb,
            top_agents=top_agents,
            top_endpoints=top_endpoints,
            daily_usage=daily_usage,
            error_breakdown=error_breakdown,
            cost_summary=cost_summary
        )

    def generate_pdf_report(
        self,
        report_data: ReportData,
        include_charts: bool = True
    ) -> bytes:
        """
        Generate PDF report

        Args:
            report_data: Report data
            include_charts: Whether to include charts

        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()

        # Create document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E40AF'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        # Add title
        title = Paragraph(f"GreenLang Usage Report", title_style)
        story.append(title)

        # Add partner info
        partner_info = Paragraph(
            f"<b>Partner:</b> {report_data.partner_name}<br/>"
            f"<b>Period:</b> {report_data.period_start.strftime('%Y-%m-%d')} to {report_data.period_end.strftime('%Y-%m-%d')}<br/>"
            f"<b>Generated:</b> {DeterministicClock.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            styles['Normal']
        )
        story.append(partner_info)
        story.append(Spacer(1, 20))

        # Summary section
        story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
        story.append(Spacer(1, 10))

        summary_data = [
            ['Metric', 'Value'],
            ['Total Requests', f"{report_data.total_requests:,}"],
            ['Successful Requests', f"{report_data.successful_requests:,}"],
            ['Failed Requests', f"{report_data.failed_requests:,}"],
            ['Success Rate', f"{report_data.success_rate:.2f}%"],
            ['Avg Response Time', f"{report_data.avg_response_time_ms:.0f} ms"],
            ['Total Data Transfer', f"{report_data.total_data_mb:.2f} MB"],
            ['Estimated Cost', f"${report_data.cost_summary['total_cost']:.2f}"],
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E40AF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 20))

        # Top Agents
        if report_data.top_agents:
            story.append(Paragraph("<b>Top Agents</b>", styles['Heading2']))
            story.append(Spacer(1, 10))

            agent_data = [['Agent ID', 'Usage Count']]
            for agent in report_data.top_agents[:5]:
                agent_data.append([agent['agent_id'], str(agent['count'])])

            agent_table = Table(agent_data, colWidths=[3*inch, 2*inch])
            agent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10B981')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(agent_table)
            story.append(Spacer(1, 20))

        # Charts
        if include_charts and report_data.daily_usage:
            story.append(PageBreak())
            story.append(Paragraph("<b>Usage Trends</b>", styles['Heading2']))
            story.append(Spacer(1, 10))

            # Create daily usage chart
            chart_buffer = self._create_usage_chart(report_data.daily_usage)
            if chart_buffer:
                img = Image(chart_buffer, width=6*inch, height=3*inch)
                story.append(img)

        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def _create_usage_chart(self, daily_usage: List[Dict[str, Any]]) -> Optional[io.BytesIO]:
        """Create usage trend chart"""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 5))

            dates = [item['date'] for item in daily_usage]
            requests = [item['requests'] for item in daily_usage]
            errors = [item['errors'] for item in daily_usage]

            # Plot data
            ax.plot(dates, requests, marker='o', label='Total Requests', color='#1E40AF', linewidth=2)
            ax.plot(dates, errors, marker='s', label='Errors', color='#EF4444', linewidth=2)

            # Formatting
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Daily Usage Trend', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)

            plt.tight_layout()

            # Save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150)
            buffer.seek(0)
            plt.close()

            return buffer
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return None

    def generate_csv_report(self, report_data: ReportData) -> bytes:
        """
        Generate CSV report

        Args:
            report_data: Report data

        Returns:
            CSV bytes
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)

        # Write summary
        writer.writerow(['GreenLang Usage Report'])
        writer.writerow(['Partner', report_data.partner_name])
        writer.writerow(['Period', f"{report_data.period_start.date()} to {report_data.period_end.date()}"])
        writer.writerow([])

        # Write metrics
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Requests', report_data.total_requests])
        writer.writerow(['Successful Requests', report_data.successful_requests])
        writer.writerow(['Failed Requests', report_data.failed_requests])
        writer.writerow(['Success Rate (%)', f"{report_data.success_rate:.2f}"])
        writer.writerow(['Avg Response Time (ms)', f"{report_data.avg_response_time_ms:.0f}"])
        writer.writerow(['Total Data (MB)', f"{report_data.total_data_mb:.2f}"])
        writer.writerow([])

        # Write daily usage
        writer.writerow(['Daily Usage'])
        writer.writerow(['Date', 'Requests', 'Errors'])
        for item in report_data.daily_usage:
            writer.writerow([item['date'], item['requests'], item['errors']])
        writer.writerow([])

        # Write top agents
        writer.writerow(['Top Agents'])
        writer.writerow(['Agent ID', 'Count'])
        for agent in report_data.top_agents:
            writer.writerow([agent['agent_id'], agent['count']])

        csv_bytes = buffer.getvalue().encode('utf-8')
        buffer.close()

        return csv_bytes

    def send_email_report(
        self,
        report_data: ReportData,
        email_to: str,
        report_bytes: bytes,
        format: ReportFormat
    ):
        """
        Send report via email

        Args:
            report_data: Report data
            email_to: Recipient email
            report_bytes: Report file bytes
            format: Report format
        """
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders

        # Email configuration (should be in config)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "reports@greenlang.com"
        sender_password = "your-password"

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email_to
        msg['Subject'] = f"GreenLang Usage Report - {report_data.period_start.date()} to {report_data.period_end.date()}"

        # Email body
        body = f"""
        Dear {report_data.partner_name},

        Please find attached your GreenLang usage report for the period:
        {report_data.period_start.date()} to {report_data.period_end.date()}

        Summary:
        - Total Requests: {report_data.total_requests:,}
        - Success Rate: {report_data.success_rate:.2f}%
        - Avg Response Time: {report_data.avg_response_time_ms:.0f} ms

        Best regards,
        GreenLang Team
        """

        msg.attach(MIMEText(body, 'plain'))

        # Attach report
        filename = f"greenlang_report_{report_data.period_start.date()}.{format.value}"
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(report_bytes)
        encoders.encode_base64(attachment)
        attachment.add_header('Content-Disposition', f'attachment; filename={filename}')
        msg.attach(attachment)

        try:
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Report sent to {email_to}")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise


class ReportScheduler:
    """
    Schedules automatic report generation and delivery
    """

    def __init__(self, db: Session):
        self.db = db
        self.generator = ReportGenerator(db)

    async def send_daily_reports(self):
        """Send daily reports to all partners"""
        from .api import PartnerModel

        partners = self.db.query(PartnerModel).filter(
            PartnerModel.status == "ACTIVE"
        ).all()

        yesterday = DeterministicClock.utcnow().date() - timedelta(days=1)
        start_date = datetime.combine(yesterday, datetime.min.time())
        end_date = datetime.combine(yesterday, datetime.max.time())

        for partner in partners:
            try:
                # Generate report
                report_data = self.generator.generate_report(
                    partner.id,
                    start_date,
                    end_date,
                    ReportType.DAILY
                )

                # Generate PDF
                pdf_bytes = self.generator.generate_pdf_report(report_data)

                # Send email
                if partner.email:
                    self.generator.send_email_report(
                        report_data,
                        partner.email,
                        pdf_bytes,
                        ReportFormat.PDF
                    )

                logger.info(f"Sent daily report to partner {partner.id}")
            except Exception as e:
                logger.error(f"Error sending daily report to {partner.id}: {e}")

    async def send_monthly_reports(self):
        """Send monthly reports to all partners"""
        from .api import PartnerModel

        partners = self.db.query(PartnerModel).filter(
            PartnerModel.status == "ACTIVE"
        ).all()

        # Last month
        today = DeterministicClock.utcnow().date()
        first_of_month = today.replace(day=1)
        last_month_end = first_of_month - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)

        start_date = datetime.combine(last_month_start, datetime.min.time())
        end_date = datetime.combine(last_month_end, datetime.max.time())

        for partner in partners:
            try:
                # Generate report
                report_data = self.generator.generate_report(
                    partner.id,
                    start_date,
                    end_date,
                    ReportType.MONTHLY
                )

                # Generate PDF
                pdf_bytes = self.generator.generate_pdf_report(
                    report_data,
                    include_charts=True
                )

                # Send email
                if partner.email:
                    self.generator.send_email_report(
                        report_data,
                        partner.email,
                        pdf_bytes,
                        ReportFormat.PDF
                    )

                logger.info(f"Sent monthly report to partner {partner.id}")
            except Exception as e:
                logger.error(f"Error sending monthly report to {partner.id}: {e}")


if __name__ == "__main__":
    # Example usage
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("postgresql://localhost/greenlang_partners")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    generator = ReportGenerator(db)

    # Generate report
    report_data = generator.generate_report(
        "partner_123",
        DeterministicClock.utcnow() - timedelta(days=30),
        DeterministicClock.utcnow(),
        ReportType.MONTHLY
    )

    # Generate PDF
    pdf_bytes = generator.generate_pdf_report(report_data, include_charts=True)

    # Save to file
    with open("report.pdf", "wb") as f:
        f.write(pdf_bytes)

    print("Report generated successfully")
