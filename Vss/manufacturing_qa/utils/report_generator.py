import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from markdown import markdown
from bs4 import BeautifulSoup
import logging

import re
from clients.llm_client import LLMClient

logger = logging.getLogger(__name__)

def clean_markdown_for_reportlab(text: str) -> str:
    """
    Converts basic Markdown (bold, italic, bullets) to ReportLab XML-like tags.
    """
    if not text:
        return ""
    
    # 0. Remove <think>...</think> blocks (Chain of Thought)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 1. Handle Bold (**text** -> <b>text</b>)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    
    # 2. Handle Italics (*text* -> <i>text</i>) - be careful not to conflict with bullets
    # text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # 3. Handle Lists (* Item or - Item)
    # Convert "* " at start of line to a bullet character
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('* ') or stripped.startswith('- '):
            # Use a bullet character and some padding
            processed_lines.append(f"• {stripped[2:]}")
        elif re.match(r'^\d+\.', stripped):
            # numbered lists
            processed_lines.append(stripped)
        else:
            processed_lines.append(line)
    
    text = '\n'.join(processed_lines)
    
    # 4. Handle Line Breaks (essential for ReportLab Paragraph)
    text = text.replace('\n', '<br/>')
    
    # 5. Clean up any remaining artifacts
    text = text.replace('_', '') # Sometimes _ is used for italics
    
    return text

def generate_pdf_report(data: dict, output_path: str) -> str:
    """
    Generates a premium PDF report for the manufacturing QA inspection.
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter, 
                            leftMargin=0.5*inch, rightMargin=0.5*inch, 
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(
        name='MainTitle',
        parent=styles['Heading1'],
        fontSize=26,
        textColor=colors.HexColor('#005073'), # Cisco Blue
        alignment=1, # Center
        spaceAfter=20
    ))
    
    styles.add(ParagraphStyle(
        name='PremiumHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#005073'),
        borderPadding=5,
        spaceBefore=15,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name='AuditText',
        parent=styles['BodyText'],
        fontSize=10,
        leading=12
    ))

    story = []
    
    # 1. Header
    story.append(Paragraph("MANUFACTURING QUALITY ASSURANCE REPORT", styles['MainTitle']))
    # story.append(Paragraph(f"Session Identifier: {data.get('video_source', 'N/A')}", styles['AuditText'])) # Removed
    story.append(Spacer(1, 10))
    
    # 2. Executive Summary Block
    story.append(Paragraph("EXECUTIVE SUMMARY", styles['PremiumHeader']))
    
    verdict = data.get("verdict", "REVIEW REQUIRED")
    verdict_color = colors.green if verdict == "PASS" else (colors.red if verdict == "FAIL" else colors.orange)
    
    summary_data = [
        ["INSPECTION DATE", data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
        ["VIDEO SOURCE", Paragraph(data.get("video_source", "N/A"), styles['AuditText'])],
        # ["RAG COLLECTION", data.get("rag_collection", "default")], # Removed
        ["OVERALL VERDICT", Paragraph(f"<b>{verdict}</b>", ParagraphStyle('v', textColor=verdict_color, fontSize=12))]
    ]
    
    t_summary = Table(summary_data, colWidths=[2*inch, 5*inch])
    t_summary.setStyle(TableStyle([
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F2F2F2')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # New Section: Executive Summary Text
    exec_summary = data.get("executive_summary", "")
    if exec_summary:
        story.append(Paragraph("<b>SUMMARY:</b> " + clean_markdown_for_reportlab(exec_summary), styles['AuditText']))
        story.append(Spacer(1, 15))

    # New Section: Improvements & Highlights (Side by Side potentially, or stacked)
    improvements = data.get("improvements", [])
    highlights = data.get("positive_highlights", [])
    
    if improvements:
        story.append(Paragraph("PROJECTED IMPROVEMENTS", styles['Heading3']))
        for imp in improvements:
            story.append(Paragraph(f"• {clean_markdown_for_reportlab(imp)}", styles['AuditText']))
        story.append(Spacer(1, 10))
            
    if highlights:
        story.append(Paragraph("POSITIVE HIGHLIGHTS", styles['Heading3']))
        for high in highlights:
            story.append(Paragraph(f"• {clean_markdown_for_reportlab(high)}", styles['AuditText']))
        story.append(Spacer(1, 15))
    
    # Narrative Context
    story.append(Paragraph("SOP AUDIT CONTEXT", styles['PremiumHeader']))
    formatted_summary = clean_markdown_for_reportlab(data.get("sop_summary", "No SOP analysis available."))
    story.append(Paragraph(formatted_summary, styles['AuditText']))
    story.append(Spacer(1, 20))
    
    # 3. Detailed Findings
    story.append(Paragraph("DETAILED FINDINGS & VISUAL EVIDENCE", styles['PremiumHeader']))
    story.append(Spacer(1, 10))
    
    events = data.get("events", [])
    if not events:
        story.append(Paragraph("No anomalies or safety violations detected during this session.", styles['AuditText']))
    else:
        for i, event in enumerate(events):
            # Event Card
            event_title = f"Finding {i+1}: {event.get('type', 'Detection')}"
            story.append(Paragraph(event_title, styles['Heading3']))
            
            # Score Calculation / Color
            score = event.get("compliance_score", 0)
            score_color = colors.green if score > 80 else (colors.orange if score > 50 else colors.red)
            
            # Table Data - Keep only concise metadata
            event_details = [
                ["TIMESTAMP", event.get("timestamp", "N/A")],
                ["SOP SECTION", event.get("sop_section", "N/A")],
                ["COMPLIANCE SCORE", Paragraph(f"<b>{score}%</b>", ParagraphStyle('s', textColor=score_color))],
            ]
            
            # Split into details and image if image exists
            screenshot_path = event.get("screenshot_path")
            has_image = screenshot_path and os.path.exists(screenshot_path)
            
            if has_image:
                img = Image(screenshot_path, width=3.5*inch, height=2*inch)
                # Left: Metadata Table, Right: Image
                main_table_data = [[Table(event_details, colWidths=[1.5*inch, 2.5*inch]), img]]
                col_widths = [4.2*inch, 3.8*inch]
            else:
                main_table_data = [[Table(event_details, colWidths=[1.5*inch, 6*inch])]]
                col_widths = [7.5*inch]
            
            t_main = Table(main_table_data, colWidths=col_widths)
            t_main.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(t_main)
            
            # Add flowing text outside the table (Allows splitting across pages)
            story.append(Paragraph("<b>AUDIT ANALYSIS:</b>", styles['AuditText']))
            story.append(Paragraph(clean_markdown_for_reportlab(event.get("rag_verification", "")), styles['AuditText']))
            story.append(Spacer(1, 10))
            
            story.append(Paragraph("<b>CHRONOLOGICAL AUDIT TIMELINE:</b>", styles['AuditText']))
            story.append(Paragraph(clean_markdown_for_reportlab(event.get("dense_timeline", "N/A")), styles['AuditText']))
            story.append(Spacer(1, 20))
            
    try:
        doc.build(story)
        logger.info(f"Premium report generated at {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate premium PDF: {e}")
        return ""

if __name__ == "__main__":
    # Test
    test_data = {
        "timestamp": "2025-01-01 10:00:00",
        "video_source": "cam_01.mp4",
        "sop_summary": "All personnel must wear helmets.",
        "verdict": "FAIL",
        "events": [
            {
                "timestamp": "00:05:23",
                "type": "Safety Violation",
                "description": "Person detected without helmet.",
                "rag_verification": "Violation Confirmed: SOP Section 3.1 mandates helmets in this zone.",
                "screenshot_path": None
            }
        ]
    }
    generate_pdf_report(test_data, "test_report.pdf")
