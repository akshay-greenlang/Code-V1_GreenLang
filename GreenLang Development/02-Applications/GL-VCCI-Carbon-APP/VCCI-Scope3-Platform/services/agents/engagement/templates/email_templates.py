# -*- coding: utf-8 -*-
"""
Email templates for supplier engagement campaigns.

Includes multi-touch sequences with personalization and compliance elements.
"""
from typing import Dict, Any, List
from string import Template

from ..models import EmailTemplate


# Template metadata
TEMPLATE_METADATA = {
    "touch_1_introduction": {
        "touch_number": 1,
        "day_offset": 0,
        "purpose": "Initial introduction and value proposition",
        "typical_subject": "Partner with us on carbon transparency"
    },
    "touch_2_reminder": {
        "touch_number": 2,
        "day_offset": 14,
        "purpose": "Gentle reminder with benefits",
        "typical_subject": "Your action needed: Carbon data request"
    },
    "touch_3_final_reminder": {
        "touch_number": 3,
        "day_offset": 35,
        "purpose": "Final call with urgency",
        "typical_subject": "Final reminder: Carbon transparency program"
    },
    "touch_4_thank_you": {
        "touch_number": 4,
        "day_offset": 42,
        "purpose": "Thank you or alternative next steps",
        "typical_subject": "Thank you or next steps"
    }
}


# Touch 1: Introduction
TOUCH_1_TEMPLATE = EmailTemplate(
    template_id="touch_1_introduction",
    subject="Partner with ${company_name} on carbon transparency",
    body_html="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px;">
        <h2 style="color: #2c5f2d; margin-top: 0;">Partner with us on carbon transparency</h2>
    </div>

    <p>Dear ${contact_name},</p>

    <p>As a valued supplier to <strong>${company_name}</strong>, we're inviting you to participate in our carbon transparency program. By sharing your product carbon footprint (PCF) data, you'll help us:</p>

    <ul style="margin: 20px 0;">
        <li>Meet our science-based climate targets</li>
        <li>Improve supply chain transparency</li>
        <li>Identify mutual decarbonization opportunities</li>
        <li>Support industry-wide sustainability initiatives</li>
    </ul>

    <div style="background-color: #e8f5e9; padding: 15px; border-left: 4px solid #2c5f2d; margin: 20px 0;">
        <h3 style="margin-top: 0; color: #2c5f2d;">What's in it for you?</h3>
        <ul style="margin-bottom: 0;">
            <li>Demonstrate leadership in sustainability</li>
            <li>Access to aggregated industry benchmarks</li>
            <li>Recognition in our supplier awards program</li>
            <li>Strengthen business relationship with ${company_name}</li>
        </ul>
    </div>

    <div style="text-align: center; margin: 30px 0;">
        <a href="${portal_url}" style="background-color: #2c5f2d; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Access Supplier Portal</a>
    </div>

    <p><strong>Next Steps:</strong></p>
    <ol>
        <li>Click the link above to access our secure supplier portal</li>
        <li>Review the data requirements (we've made it simple!)</li>
        <li>Upload your PCF data or use our guided form</li>
        <li>Track your progress and earn recognition badges</li>
    </ol>

    <p>Questions? Reply to this email or contact our sustainability team at ${support_email}.</p>

    <p>Best regards,<br>
    <strong>${sender_name}</strong><br>
    Sustainability Team, ${company_name}</p>

    <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">

    <div style="font-size: 12px; color: #666;">
        <p><a href="${privacy_policy_url}" style="color: #2c5f2d;">Privacy Policy</a> |
        <a href="${unsubscribe_url}" style="color: #2c5f2d;">Unsubscribe</a></p>

        <p style="margin-top: 10px;">${company_address}</p>

        <p style="margin-top: 10px; font-style: italic;">This email was sent to ${email_address} regarding your business relationship with ${company_name}.</p>
    </div>
</body>
</html>
    """,
    body_text="""
Dear ${contact_name},

As a valued supplier to ${company_name}, we're inviting you to participate in our carbon transparency program. By sharing your product carbon footprint (PCF) data, you'll help us:

- Meet our science-based climate targets
- Improve supply chain transparency
- Identify mutual decarbonization opportunities
- Support industry-wide sustainability initiatives

WHAT'S IN IT FOR YOU?

- Demonstrate leadership in sustainability
- Access to aggregated industry benchmarks
- Recognition in our supplier awards program
- Strengthen business relationship with ${company_name}

ACCESS SUPPLIER PORTAL: ${portal_url}

NEXT STEPS:

1. Click the link above to access our secure supplier portal
2. Review the data requirements (we've made it simple!)
3. Upload your PCF data or use our guided form
4. Track your progress and earn recognition badges

Questions? Reply to this email or contact our sustainability team at ${support_email}.

Best regards,
${sender_name}
Sustainability Team, ${company_name}

---

Privacy Policy: ${privacy_policy_url}
Unsubscribe: ${unsubscribe_url}

${company_address}

This email was sent to ${email_address} regarding your business relationship with ${company_name}.
    """,
    language="en",
    personalization_fields=[
        "company_name", "contact_name", "portal_url", "sender_name",
        "support_email", "email_address", "privacy_policy_url",
        "unsubscribe_url", "company_address"
    ]
)


# Touch 2: Reminder
TOUCH_2_TEMPLATE = EmailTemplate(
    template_id="touch_2_reminder",
    subject="Action needed: Carbon data request from ${company_name}",
    body_html="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background-color: #fff3cd; padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #ffc107;">
        <h2 style="color: #856404; margin-top: 0;">Reminder: Carbon data submission</h2>
    </div>

    <p>Dear ${contact_name},</p>

    <p>We reached out two weeks ago inviting you to participate in our carbon transparency program. We wanted to follow up and see if you need any assistance getting started.</p>

    <p><strong>Why this matters:</strong></p>
    <ul>
        <li>Scope 3 emissions represent <strong>${scope3_percentage}%</strong> of our total footprint</li>
        <li>Your participation helps us both achieve climate goals</li>
        <li>Early participants receive special recognition</li>
    </ul>

    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0; color: #1565c0;">We've made it easy!</h3>
        <p>✓ Simple upload process (CSV, Excel, or JSON)<br>
        ✓ Guided data entry forms<br>
        ✓ Real-time validation and feedback<br>
        ✓ Progress tracking dashboard</p>
    </div>

    <div style="text-align: center; margin: 30px 0;">
        <a href="${portal_url}" style="background-color: #ffc107; color: #000; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Upload Your Data Now</a>
    </div>

    <p><strong>Need help?</strong> We're here to support you:</p>
    <ul>
        <li>📧 Email: ${support_email}</li>
        <li>📚 Data submission guide: ${guide_url}</li>
        <li>💬 FAQ: ${faq_url}</li>
    </ul>

    <p>Thank you for being a valued partner in our sustainability journey.</p>

    <p>Best regards,<br>
    <strong>${sender_name}</strong><br>
    Sustainability Team, ${company_name}</p>

    <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">

    <div style="font-size: 12px; color: #666;">
        <p><a href="${privacy_policy_url}" style="color: #2c5f2d;">Privacy Policy</a> |
        <a href="${unsubscribe_url}" style="color: #2c5f2d;">Unsubscribe</a></p>
        <p>${company_address}</p>
    </div>
</body>
</html>
    """,
    body_text="""
Dear ${contact_name},

REMINDER: Carbon data submission

We reached out two weeks ago inviting you to participate in our carbon transparency program. We wanted to follow up and see if you need any assistance getting started.

WHY THIS MATTERS:

- Scope 3 emissions represent ${scope3_percentage}% of our total footprint
- Your participation helps us both achieve climate goals
- Early participants receive special recognition

WE'VE MADE IT EASY!

✓ Simple upload process (CSV, Excel, or JSON)
✓ Guided data entry forms
✓ Real-time validation and feedback
✓ Progress tracking dashboard

UPLOAD YOUR DATA: ${portal_url}

NEED HELP? We're here to support you:

- Email: ${support_email}
- Data submission guide: ${guide_url}
- FAQ: ${faq_url}

Thank you for being a valued partner in our sustainability journey.

Best regards,
${sender_name}
Sustainability Team, ${company_name}

---

Privacy Policy: ${privacy_policy_url}
Unsubscribe: ${unsubscribe_url}
${company_address}
    """,
    language="en",
    personalization_fields=[
        "company_name", "contact_name", "portal_url", "sender_name",
        "support_email", "scope3_percentage", "guide_url", "faq_url",
        "privacy_policy_url", "unsubscribe_url", "company_address", "email_address"
    ]
)


# Touch 3: Final reminder
TOUCH_3_TEMPLATE = EmailTemplate(
    template_id="touch_3_final_reminder",
    subject="Final reminder: Carbon transparency program deadline approaching",
    body_html="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background-color: #f8d7da; padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #dc3545;">
        <h2 style="color: #721c24; margin-top: 0;">⏰ Final Reminder: Deadline ${deadline_date}</h2>
    </div>

    <p>Dear ${contact_name},</p>

    <p>This is our final reminder about the carbon data submission program. The deadline is <strong>${deadline_date}</strong>, which is less than one week away.</p>

    <div style="background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <p style="margin: 0; font-size: 18px;"><strong>⚠️ Don't miss out on:</strong></p>
        <ul style="margin-bottom: 0;">
            <li>Being recognized as a sustainability leader</li>
            <li>Early adopter badge and certificates</li>
            <li>Industry benchmark insights</li>
            <li>Strengthening partnership with ${company_name}</li>
        </ul>
    </div>

    <div style="text-align: center; margin: 30px 0;">
        <a href="${portal_url}" style="background-color: #dc3545; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Submit Data Before ${deadline_date}</a>
    </div>

    <p><strong>If you're facing challenges:</strong></p>
    <ul>
        <li>We accept partial submissions (you can complete later)</li>
        <li>Our team can help with data formatting</li>
        <li>We provide estimation methodologies if you lack primary data</li>
    </ul>

    <p><strong>Quick action steps:</strong></p>
    <ol>
        <li>Access portal: ${portal_url}</li>
        <li>Upload data or use guided form (15-30 minutes)</li>
        <li>Submit and track your progress</li>
    </ol>

    <p>Last chance to participate! We value your partnership and hope to include your contribution.</p>

    <p>Urgent questions? Contact us immediately at ${support_email} or call ${support_phone}.</p>

    <p>Best regards,<br>
    <strong>${sender_name}</strong><br>
    Sustainability Team, ${company_name}</p>

    <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">

    <div style="font-size: 12px; color: #666;">
        <p><a href="${privacy_policy_url}" style="color: #2c5f2d;">Privacy Policy</a> |
        <a href="${unsubscribe_url}" style="color: #2c5f2d;">Unsubscribe</a></p>
        <p>${company_address}</p>
    </div>
</body>
</html>
    """,
    body_text="""
Dear ${contact_name},

FINAL REMINDER: Deadline ${deadline_date}

This is our final reminder about the carbon data submission program. The deadline is ${deadline_date}, which is less than one week away.

DON'T MISS OUT ON:

- Being recognized as a sustainability leader
- Early adopter badge and certificates
- Industry benchmark insights
- Strengthening partnership with ${company_name}

SUBMIT DATA BEFORE ${deadline_date}: ${portal_url}

IF YOU'RE FACING CHALLENGES:

- We accept partial submissions (you can complete later)
- Our team can help with data formatting
- We provide estimation methodologies if you lack primary data

QUICK ACTION STEPS:

1. Access portal: ${portal_url}
2. Upload data or use guided form (15-30 minutes)
3. Submit and track your progress

Last chance to participate! We value your partnership and hope to include your contribution.

Urgent questions? Contact us immediately at ${support_email} or call ${support_phone}.

Best regards,
${sender_name}
Sustainability Team, ${company_name}

---

Privacy Policy: ${privacy_policy_url}
Unsubscribe: ${unsubscribe_url}
${company_address}
    """,
    language="en",
    personalization_fields=[
        "company_name", "contact_name", "portal_url", "sender_name",
        "support_email", "support_phone", "deadline_date",
        "privacy_policy_url", "unsubscribe_url", "company_address", "email_address"
    ]
)


# Touch 4: Thank you or next steps
TOUCH_4_TEMPLATE = EmailTemplate(
    template_id="touch_4_thank_you",
    subject="Thank you from ${company_name} sustainability team",
    body_html="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background-color: #d4edda; padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 4px solid #28a745;">
        <h2 style="color: #155724; margin-top: 0;">🎉 Thank you for your participation!</h2>
    </div>

    <p>Dear ${contact_name},</p>

    <p>Thank you for participating in our carbon transparency program! Your contribution is invaluable to our shared sustainability goals.</p>

    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h3 style="margin-top: 0; color: #2c5f2d;">What happens next?</h3>
        <ol style="margin-bottom: 0;">
            <li>We're validating your submission (you'll receive confirmation within 3 business days)</li>
            <li>You'll receive your sustainability badge and certificate</li>
            <li>Access to benchmark insights (coming soon)</li>
            <li>Invitation to supplier sustainability forum</li>
        </ol>
    </div>

    <p><strong>Your Impact:</strong></p>
    <ul>
        <li>You're among the <strong>${participant_percentage}%</strong> of suppliers who participated</li>
        <li>Your data helps ${company_name} track <strong>${coverage_percentage}%</strong> of Scope 3 emissions</li>
        <li>Together, we're driving industry transformation</li>
    </ul>

    <div style="text-align: center; margin: 30px 0;">
        <a href="${portal_url}" style="background-color: #28a745; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">View Your Dashboard</a>
    </div>

    <p><strong>Continue your journey:</strong></p>
    <ul>
        <li>Update your data anytime through the portal</li>
        <li>Track your data quality score improvements</li>
        <li>Climb the supplier leaderboard</li>
        <li>Earn additional recognition badges</li>
    </ul>

    <p>We're grateful for your partnership and commitment to sustainability. Together, we're making a real difference.</p>

    <p>Warm regards,<br>
    <strong>${sender_name}</strong><br>
    Sustainability Team, ${company_name}</p>

    <hr style="border: none; border-top: 1px solid #ddd; margin: 30px 0;">

    <div style="font-size: 12px; color: #666;">
        <p><a href="${privacy_policy_url}" style="color: #2c5f2d;">Privacy Policy</a> |
        <a href="${unsubscribe_url}" style="color: #2c5f2d;">Unsubscribe</a></p>
        <p>${company_address}</p>
    </div>
</body>
</html>
    """,
    body_text="""
Dear ${contact_name},

THANK YOU FOR YOUR PARTICIPATION!

Thank you for participating in our carbon transparency program! Your contribution is invaluable to our shared sustainability goals.

WHAT HAPPENS NEXT?

1. We're validating your submission (you'll receive confirmation within 3 business days)
2. You'll receive your sustainability badge and certificate
3. Access to benchmark insights (coming soon)
4. Invitation to supplier sustainability forum

YOUR IMPACT:

- You're among the ${participant_percentage}% of suppliers who participated
- Your data helps ${company_name} track ${coverage_percentage}% of Scope 3 emissions
- Together, we're driving industry transformation

VIEW YOUR DASHBOARD: ${portal_url}

CONTINUE YOUR JOURNEY:

- Update your data anytime through the portal
- Track your data quality score improvements
- Climb the supplier leaderboard
- Earn additional recognition badges

We're grateful for your partnership and commitment to sustainability. Together, we're making a real difference.

Warm regards,
${sender_name}
Sustainability Team, ${company_name}

---

Privacy Policy: ${privacy_policy_url}
Unsubscribe: ${unsubscribe_url}
${company_address}
    """,
    language="en",
    personalization_fields=[
        "company_name", "contact_name", "portal_url", "sender_name",
        "participant_percentage", "coverage_percentage",
        "privacy_policy_url", "unsubscribe_url", "company_address", "email_address"
    ]
)


# Template registry
EMAIL_TEMPLATES = {
    "touch_1_introduction": TOUCH_1_TEMPLATE,
    "touch_2_reminder": TOUCH_2_TEMPLATE,
    "touch_3_final_reminder": TOUCH_3_TEMPLATE,
    "touch_4_thank_you": TOUCH_4_TEMPLATE,
}


def get_template(template_id: str) -> EmailTemplate:
    """
    Get email template by ID.

    Args:
        template_id: Template identifier

    Returns:
        Email template

    Raises:
        KeyError: If template not found
    """
    if template_id not in EMAIL_TEMPLATES:
        raise KeyError(f"Template {template_id} not found")
    return EMAIL_TEMPLATES[template_id]


def render_template(
    template: EmailTemplate,
    personalization_data: Dict[str, Any]
) -> Dict[str, str]:
    """
    Render email template with personalization data.

    Args:
        template: Email template
        personalization_data: Dictionary of personalization values

    Returns:
        Dictionary with rendered subject, body_html, body_text
    """
    # Use safe_substitute to avoid KeyError for missing variables
    subject_template = Template(template.subject)
    html_template = Template(template.body_html)
    text_template = Template(template.body_text)

    rendered = {
        "subject": subject_template.safe_substitute(personalization_data),
        "body_html": html_template.safe_substitute(personalization_data),
        "body_text": text_template.safe_substitute(personalization_data),
    }

    return rendered


def list_templates() -> List[str]:
    """
    List available template IDs.

    Returns:
        List of template IDs
    """
    return list(EMAIL_TEMPLATES.keys())


def get_template_metadata(template_id: str) -> Dict[str, Any]:
    """
    Get metadata for template.

    Args:
        template_id: Template identifier

    Returns:
        Template metadata
    """
    return TEMPLATE_METADATA.get(template_id, {})
