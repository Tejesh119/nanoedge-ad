"""Convert Markdown to PDF for IEEE Conference Paper"""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib import colors
import re

def create_pdf():
    # Read markdown
    with open('outputs/IEEE_Conference_Paper.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create PDF
    pdf_file = 'outputs/IEEE_Conference_Paper.pdf'
    doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    # Container for flowables
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#000000'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    author_style = ParagraphStyle(
        'Author',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_CENTER,
        spaceAfter=6
    )
    
    abstract_style = ParagraphStyle(
        'Abstract',
        parent=styles['BodyText'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=12,
        fontName='Helvetica-Bold',
        spaceAfter=6,
        spaceBefore=12
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=10,
        fontName='Helvetica-Bold',
        spaceAfter=6,
        spaceBefore=10
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=9,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    # Parse markdown
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Title (first heading)
        if line.startswith('# ') and i < 5:
            title_text = line[2:].strip()
            story.append(Paragraph(title_text, title_style))
            story.append(Spacer(1, 0.2*inch))
            i += 1
            continue
        
        # Section headings (##)
        if line.startswith('## '):
            heading_text = line[3:].strip()
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(heading_text, heading1_style))
            i += 1
            continue
        
        # Subsection headings (###)
        if line.startswith('### '):
            heading_text = line[4:].strip()
            story.append(Paragraph(heading_text, heading2_style))
            i += 1
            continue
        
        # Bold author info
        if line.startswith('**') and '**' in line[2:]:
            author_text = line.replace('**', '<b>').replace('**', '</b>')
            story.append(Paragraph(author_text, author_style))
            i += 1
            continue
        
        # Italic text (conference info)
        if line.startswith('*') and not line.startswith('**'):
            text = line.replace('*', '<i>').replace('*', '</i>')
            story.append(Paragraph(text, author_style))
            i += 1
            continue
        
        # Horizontal rule
        if line.startswith('---'):
            story.append(Spacer(1, 0.1*inch))
            i += 1
            continue
        
        # Tables - simplified
        if line.startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            if len(table_lines) > 2:  # Has header and content
                # Parse table
                rows = []
                for tline in table_lines:
                    if '---' not in tline:  # Skip separator
                        cells = [cell.strip() for cell in tline.split('|')[1:-1]]
                        rows.append(cells)
                
                if rows:
                    # Create table
                    t = Table(rows, repeatRows=1)
                    t.setStyle(TableStyle([
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 7),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.15*inch))
            continue
        
        # Code blocks
        if line.startswith('```'):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # Skip closing ```
            
            code_text = '<font name="Courier" size="7">' + '<br/>'.join(code_lines[:10]) + '</font>'
            story.append(Paragraph(code_text, body_style))
            story.append(Spacer(1, 0.1*inch))
            continue
        
        # Math equations - simplified
        if '$$' in line:
            eq_text = line.replace('$$', '').strip()
            story.append(Paragraph(f'<i>{eq_text}</i>', body_style))
            i += 1
            continue
        
        # Bullet points
        if line.startswith('- ') or line.startswith('* '):
            bullet_text = '• ' + line[2:]
            story.append(Paragraph(bullet_text, body_style))
            i += 1
            continue
        
        # Numbered lists
        if re.match(r'^\d+[\.\)]\s', line):
            story.append(Paragraph(line, body_style))
            i += 1
            continue
        
        # References
        if line.startswith('[') and ']' in line:
            ref_text = line.replace('*', '<i>').replace('**', '<b>')
            story.append(Paragraph(ref_text, body_style))
            i += 1
            continue
        
        # Regular paragraphs
        # Clean up markdown formatting
        text = line
        text = re.sub(r'\*\*([^\*]+)\*\*', r'<b>\1</b>', text)  # Bold
        text = re.sub(r'\*([^\*]+)\*', r'<i>\1</i>', text)  # Italic
        text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)  # Code
        
        if text:
            story.append(Paragraph(text, body_style))
        
        i += 1
    
    # Build PDF
    doc.build(story)
    print(f'PDF created successfully: {pdf_file}')

if __name__ == '__main__':
    create_pdf()
