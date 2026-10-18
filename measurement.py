from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generate_homework_pdf(filename="i_did_my_homework.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # --- 1. FULLY SHADED PENCIL (Drawn first so it sits behind the text) ---
    c.saveState()
    c.setStrokeColor(colors.black)
    c.setLineWidth(2)
    
    # Position centered horizontally, shifted up
    c.translate(width / 2.0, height - 80)
    
    # Dimensions (extended length)
    body_h = 28    
    body_w = 390   # Extended length further to the right
    
    # Eraser (Far Left - Shaded gray fill)
    c.setFillColor(colors.HexColor('#d1d5db'))
    c.roundRect(-245, -body_h, 45, body_h * 2, 6, fill=1, stroke=1)
    
    # Main Shaft (Shaded Gray fill)
    c.setFillColor(colors.HexColor('#9ca3af'))  # Medium-light gray fill
    c.rect(-200, -body_h, body_w, body_h * 2, fill=1, stroke=1)
    
    # Wooden Tip (Pointing Right - Shaded Gray fill)
    tip_start_x = -200 + body_w
    c.setFillColor(colors.HexColor('#d1d5db'))
    p = c.beginPath()
    p.moveTo(tip_start_x, -body_h)
    p.lineTo(tip_start_x + 70, 0)
    p.lineTo(tip_start_x, body_h)
    p.close()
    c.drawPath(p, fill=1, stroke=1)
    
    # Lead Tip (Far Right - Solid Black fill)
    c.setFillColor(colors.black)  
    p_lead = c.beginPath()
    p_lead.moveTo(tip_start_x + 70, 0)
    p_lead.lineTo(tip_start_x + 45, -8)
    p_lead.lineTo(tip_start_x + 45, 8)
    p_lead.close()
    c.drawPath(p_lead, fill=1, stroke=1)
    
    c.restoreState()
    
    # --- 2. TITLE ("I Did My Homework!") - Drawn AFTER the pencil so it appears on top ---
    c.saveState()
    c.setFont("Helvetica-Bold", 38)
    c.setFillColor(colors.black)
    c.drawCentredString(width / 2.0, height - 95, "I Did My Homework!")
    c.restoreState()
    
    # --- 3. LARGE 4x4 GRID (Filling the rest of the page) ---
    margin = 36  # 0.5-inch margins
    grid_left = margin
    grid_right = width - margin
    grid_top = height - 150  # Clean separation below top elements
    grid_bottom = margin
    
    grid_width = grid_right - grid_left
    grid_height = grid_top - grid_bottom
    
    rows = 4
    cols = 4
    cell_w = grid_width / cols
    cell_h = grid_height / rows
    
    c.setStrokeColor(colors.black)
    c.setLineWidth(2)  # Bold grid lines
    c.setFillColor(colors.white)
    
    # Draw expanded grid cells
    for r in range(rows):
        for col in range(cols):
            x = grid_left + col * cell_w
            y = grid_bottom + r * cell_h
            c.rect(x, y, cell_w, cell_h, fill=0, stroke=1)
            
    # Save PDF
    c.save()
    print(f"PDF successfully generated: {filename}")

if __name__ == "__main__":
    generate_homework_pdf()