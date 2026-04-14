"""
utils.py - PDF generation, graph generation, and session helpers.
All outputs returned as base64 strings — nothing written to disk.
"""

import base64
import io
import datetime
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ─────────────────────────────────────────────
# GRAPH GENERATION
# ─────────────────────────────────────────────

LEVEL_COLORS = {
    "Low":    "#22c55e",
    "Medium": "#f59e0b",
    "High":   "#ef4444",
}


def generate_single_chart(class_counts):
    """Pie chart of classification for single image. Returns base64 PNG."""
    labels = []
    sizes  = []
    pie_colors = {'Fiber': '#6366f1', 'Fragment': '#f59e0b', 'Pellet': '#22c55e', 'Unknown': '#64748b'}

    for cls, cnt in class_counts.items():
        if cnt > 0:
            labels.append(cls)
            sizes.append(cnt)

    if not sizes:
        return None

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')

    clrs = [pie_colors.get(l, '#94a3b8') for l in labels]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=clrs,
        autopct='%1.0f%%', startangle=140,
        textprops={'color': '#e2e8f0', 'fontsize': 9},
        wedgeprops={'edgecolor': '#0f172a', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_color('#0f172a')
        at.set_fontweight('bold')

    ax.set_title('Particle Classification', color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────

def _b64_to_rl_image(b64_str, width_cm, height_cm):
    """Convert base64 PNG to a ReportLab Image object."""
    data = base64.b64decode(b64_str)
    buf = io.BytesIO(data)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


def generate_single_pdf(filename, result, original_b64, timestamp=None):
    """
    Generate a single-image PDF report.
    Returns base64-encoded PDF string.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title2', fontSize=18, alignment=TA_CENTER,
                                 textColor=colors.HexColor('#1e40af'),
                                 fontName='Helvetica-Bold', spaceAfter=6)
    sub_style = ParagraphStyle('Sub', fontSize=10, alignment=TA_CENTER,
                               textColor=colors.HexColor('#64748b'),
                               fontName='Helvetica')
    h2_style  = ParagraphStyle('H2', fontSize=13, fontName='Helvetica-Bold',
                                textColor=colors.HexColor('#1e293b'), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('Body', fontSize=9, fontName='Helvetica',
                                 textColor=colors.HexColor('#334155'))

    story = []

    # Header
    story.append(Paragraph("MicroScan — Microplastic Detection Report", title_style))
    story.append(Paragraph(f"Generated: {timestamp}  |  File: {filename}", sub_style))
    story.append(HRFlowable(width='100%', thickness=1.5, color=colors.HexColor('#3b82f6')))
    story.append(Spacer(1, 0.3*cm))

    # Summary stats
    level_color = {'Low': '#22c55e', 'Medium': '#f59e0b', 'High': '#ef4444'}.get(result['level'], '#64748b')
    summary_data = [
        ['Microplastic Count', str(result['count'])],
        ['Contamination Score', f"{result['score']}%"],
        ['Contamination Level', result['level']],
        ['Fibers',    str(result['class_counts'].get('Fiber', 0))],
        ['Fragments', str(result['class_counts'].get('Fragment', 0))],
        ['Pellets',   str(result['class_counts'].get('Pellet', 0))],
    ]
    t = Table(summary_data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eff6ff')),
        ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#f8fafc')),
        ('TEXTCOLOR',  (0, 0), (-1, -1), colors.HexColor('#1e293b')),
        ('FONTNAME',   (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 10),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1),
         [colors.HexColor('#f0f9ff'), colors.HexColor('#f8fafc')]),
        ('PADDING',    (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Images side by side
    story.append(Paragraph("Sample Images", h2_style))
    img_row = [_b64_to_rl_image(original_b64, 8, 8),
               _b64_to_rl_image(result['steps']['final'], 8, 8)]
    img_table = Table([img_row], colWidths=[8.5*cm, 8.5*cm])
    img_table.setStyle(TableStyle([
        ('ALIGN',   (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',  (0, 0), (-1, -1), 'MIDDLE'),
        ('INNERGRID', (0, 0), (-1, -1), 0, colors.white),
        ('BOX',     (0, 0), (-1, -1), 0, colors.white),
    ]))
    caption_table = Table([['Cropped ROI', 'Detection Result']],
                          colWidths=[8.5*cm, 8.5*cm])
    caption_table.setStyle(TableStyle([
        ('ALIGN',    (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR',(0, 0), (-1, -1), colors.HexColor('#64748b')),
    ]))
    story.append(img_table)
    story.append(caption_table)
    story.append(Spacer(1, 0.3*cm))

    # Classification table
    story.append(Paragraph("Detected Particles", h2_style))
    cls_header = ['#', 'Classification', 'Area (px²)', 'Circularity']
    cls_data   = [cls_header]
    for i, d in enumerate(result.get('detections', []), 1):
        cls_data.append([str(i), d['classification'],
                         str(d['area']), str(d['circularity'])])

    cls_table = Table(cls_data, colWidths=[1.5*cm, 5*cm, 4*cm, 4*cm])
    cls_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('GRID',       (0, 0), (-1, -1), 0.3, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor('#f8fafc'), colors.white]),
        ('PADDING',    (0, 0), (-1, -1), 5),
    ]))
    story.append(cls_table)
    story.append(Spacer(1, 0.3*cm))

    # Footer
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cbd5e1')))
    story.append(Paragraph("MicroScan © 2025 — Automated Microplastic Detection Platform",
                            ParagraphStyle('Footer', fontSize=7, alignment=TA_CENTER,
                                           textColor=colors.HexColor('#94a3b8'))))

    doc.build(story)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
# THUMBNAIL HELPER
# ─────────────────────────────────────────────

def make_thumbnail(b64_str, size=120):
    """Resize a base64 image to a thumbnail. Returns base64 PNG."""
    import cv2
    import numpy as np

    data = base64.b64decode(b64_str)
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return b64_str

    h, w = img.shape[:2]
    if max(h, w) == 0:
        return b64_str

    scale = size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    success, buf = cv2.imencode('.png', thumb)
    if not success:
        return b64_str
    return base64.b64encode(buf).decode('utf-8')
