# -*- coding: utf-8 -*-
"""
Create a sample PDF for demo corpus.

This script creates a simple PDF document about carbon offsetting
to serve as a demo file for INTL-104 RAG testing.
"""

import sys
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
except ImportError:
    print("reportlab not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

def create_carbon_offset_pdf(output_path):
    """Create a sample PDF about carbon offsetting."""

    # Create document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for flowables
    story = []

    # Get styles
    styles = getSampleStyleSheet()

    # Create custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor='#000000',
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor='#000000',
        spaceAfter=12,
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12,
    )

    # Title
    story.append(Paragraph("Carbon Offset Standards and Best Practices", title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Document info
    story.append(Paragraph("Technical Report v2.1", styles['Normal']))
    story.append(Paragraph("Published: September 2025", styles['Normal']))
    story.append(Paragraph("Status: Final", styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # Section 1
    story.append(Paragraph("1. Introduction to Carbon Offsetting", heading_style))
    story.append(Paragraph(
        "Carbon offsetting is a mechanism that allows organizations and individuals to compensate "
        "for their greenhouse gas emissions by funding projects that reduce or remove an equivalent "
        "amount of carbon dioxide from the atmosphere. This approach enables entities to achieve "
        "carbon neutrality or net-zero emissions when direct emission reductions are not immediately "
        "feasible or economically viable.",
        body_style
    ))

    story.append(Paragraph(
        "The carbon offset market operates on the principle that emissions reductions can occur "
        "anywhere in the world with the same net benefit to the global atmosphere. This geographic "
        "flexibility allows for cost-effective climate mitigation, as projects can be implemented "
        "where reduction costs are lowest.",
        body_style
    ))

    # Section 2
    story.append(Paragraph("2. Types of Carbon Offset Projects", heading_style))

    story.append(Paragraph("2.1 Renewable Energy Projects", ParagraphStyle(
        'Subsection',
        parent=heading_style,
        fontSize=13,
    )))
    story.append(Paragraph(
        "Renewable energy projects displace fossil fuel-based electricity generation. Common types include:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Wind farms</b>: Large-scale wind turbine installations that generate clean electricity<br/>"
        "• <b>Solar photovoltaic</b>: Solar panel arrays for utility-scale or distributed generation<br/>"
        "• <b>Hydroelectric</b>: Run-of-river hydro projects with minimal environmental impact<br/>"
        "• <b>Biomass energy</b>: Energy from sustainable biomass feedstocks",
        body_style
    ))

    story.append(Paragraph("2.2 Forestry and Land Use Projects", ParagraphStyle(
        'Subsection',
        parent=heading_style,
        fontSize=13,
    )))
    story.append(Paragraph(
        "Forestry projects sequester carbon through photosynthesis and protect existing carbon stocks:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Afforestation/Reforestation (A/R)</b>: Planting trees on previously unforested land<br/>"
        "• <b>REDD+</b>: Reducing Emissions from Deforestation and forest Degradation<br/>"
        "• <b>Improved Forest Management (IFM)</b>: Sustainable forestry practices that increase carbon storage<br/>"
        "• <b>Agricultural soil carbon sequestration</b>: Practices like no-till farming and cover cropping",
        body_style
    ))

    # Section 3
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("3. Verification Standards", heading_style))

    story.append(Paragraph("3.1 Verified Carbon Standard (VCS)", ParagraphStyle(
        'Subsection',
        parent=heading_style,
        fontSize=13,
    )))
    story.append(Paragraph(
        "The Verified Carbon Standard (VCS), administered by Verra, is the world's most widely used "
        "voluntary GHG program. Key requirements include:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Additionality</b>: Projects must demonstrate that emission reductions would not have occurred "
        "without the carbon finance incentive<br/>"
        "• <b>Permanence</b>: Emission reductions must be permanent (or risks addressed through buffer pools)<br/>"
        "• <b>No leakage</b>: Projects must not cause emissions to increase elsewhere<br/>"
        "• <b>Third-party validation and verification</b>: Independent auditors must assess projects",
        body_style
    ))

    story.append(Paragraph("3.2 Gold Standard", ParagraphStyle(
        'Subsection',
        parent=heading_style,
        fontSize=13,
    )))
    story.append(Paragraph(
        "Gold Standard certification ensures projects deliver measurable emission reductions while "
        "contributing to sustainable development. Projects must demonstrate co-benefits such as:",
        body_style
    ))
    story.append(Paragraph(
        "• Improved health outcomes (e.g., clean cookstoves reducing indoor air pollution)<br/>"
        "• Poverty alleviation and livelihood improvements<br/>"
        "• Biodiversity conservation<br/>"
        "• Community empowerment and stakeholder engagement",
        body_style
    ))

    # Section 4
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("4. Quality Criteria for Carbon Offsets", heading_style))

    story.append(Paragraph(
        "High-quality carbon offsets must meet rigorous criteria to ensure environmental integrity:",
        body_style
    ))

    story.append(Paragraph(
        "<b>Real</b>: Emission reductions must be quantified using conservative methodologies and "
        "verified by independent third parties.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Additional</b>: The project would not have happened without carbon finance. Common tests "
        "include investment analysis, barrier analysis, and common practice analysis.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Permanent</b>: Reductions must not be reversed. For forestry projects, this requires "
        "long-term monitoring and buffer pools to account for reversals from fire, disease, or illegal logging.",
        body_style
    ))

    story.append(Paragraph(
        "<b>Verified</b>: Independent auditors must assess projects against approved methodologies "
        "at regular intervals (typically annually).",
        body_style
    ))

    story.append(Paragraph(
        "<b>Unique</b>: Each offset must be uniquely serialized and retired once used, preventing "
        "double counting through robust registry systems.",
        body_style
    ))

    # Section 5
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("5. Best Practices for Corporate Buyers", heading_style))

    story.append(Paragraph(
        "Organizations purchasing carbon offsets should follow these guidelines to ensure quality "
        "and maximize climate impact:",
        body_style
    ))

    story.append(Paragraph(
        "<b>1. Prioritize emission reductions first</b>: Offsets should supplement, not replace, "
        "direct emission reduction efforts. Follow the mitigation hierarchy: avoid, reduce, offset.",
        body_style
    ))

    story.append(Paragraph(
        "<b>2. Choose credible standards</b>: Purchase offsets from recognized programs like VCS, "
        "Gold Standard, Climate Action Reserve, or American Carbon Registry.",
        body_style
    ))

    story.append(Paragraph(
        "<b>3. Diversify project types</b>: Balance portfolio across project types and geographies "
        "to manage risk. Consider both removal projects (forestry) and avoidance projects (renewables).",
        body_style
    ))

    story.append(Paragraph(
        "<b>4. Assess co-benefits</b>: Select projects that deliver sustainable development benefits "
        "aligned with Sustainable Development Goals (SDGs).",
        body_style
    ))

    story.append(Paragraph(
        "<b>5. Ensure transparency</b>: Publicly disclose offset purchases, retirement certificates, "
        "and climate commitments to build stakeholder trust.",
        body_style
    ))

    # Section 6
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("6. Conclusion", heading_style))

    story.append(Paragraph(
        "Carbon offsetting is a valuable tool in the climate mitigation toolkit, but it must be "
        "implemented with rigor and integrity. High-quality offsets that meet stringent criteria "
        "can contribute meaningfully to global emission reduction efforts while supporting "
        "sustainable development in project communities.",
        body_style
    ))

    story.append(Paragraph(
        "However, offsets are not a substitute for deep decarbonization within organizations. "
        "Companies should prioritize reducing their own emissions and use offsets strategically "
        "for residual emissions that cannot yet be eliminated. As technology advances and costs "
        "decline, the reliance on offsets should decrease over time.",
        body_style
    ))

    # Build PDF
    doc.build(story)
    print(f"[OK] Created PDF: {output_path}")

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_file = output_dir / "carbon_offset_standards.pdf"
    create_carbon_offset_pdf(output_file)
