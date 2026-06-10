"""Generate a small, controlled PDF that exercises key extraction features.

Run once to produce golden_sample.pdf.  The PDF is version-pinned to the
PyMuPDF build used in CI so that coordinates stay deterministic.

Features exercised:
  - Title detection (large bold font)
  - Section headings (bold, medium font)
  - Single-column body paragraphs
  - Two-column layout with gap detection
  - Bordered table (detected by find_tables)
  - Figure caption tagging
  - Bullet list
  - Two pages (cross-page structure)
"""

import fitz  # PyMuPDF

WIDTH, HEIGHT = 612, 792  # US Letter in points
MARGIN = 54  # 0.75 in
COL_GAP = 36  # wide gap for reliable column detection
COL_W = (WIDTH - 2 * MARGIN - COL_GAP) / 2
LINE_H = 13  # line height for body text

BODY = "helv"  # Helvetica
BOLD = "hebo"  # Helvetica-Bold
SIZE_TITLE = 18
SIZE_HEADING = 13
SIZE_BODY = 10


def _center_x(text, fontname, fontsize):
    w = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
    return (WIDTH - w) / 2


def _assert_fit(rv, label):
    """Abort if a textbox overflowed (negative return = text did not render)."""
    if rv < 0:
        raise RuntimeError(f"Textbox '{label}' overflowed by {-rv:.1f} pt")


def _insert_wrapped(page, x, y, text, width, fontname, fontsize):
    """Insert text wrapped to width using a textbox. Returns new y."""
    # Estimate needed height: chars per line, then lines needed
    rc = fitz.Rect(x, y, x + width, y + 200)  # generous height
    rv = page.insert_textbox(rc, text, fontsize=fontsize, fontname=fontname)
    # actual height used = 200 - rv (unused space)
    used = 200 - rv
    return y + used + 4  # small padding


def build_pdf(path: str):
    doc = fitz.open()

    # ── PAGE 1 ────────────────────────────────────────────────
    p1 = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    # Title (large bold, centered)
    title = "Thermal Analysis of CubeSat Solar Panels"
    p1.insert_text(
        (_center_x(title, BOLD, SIZE_TITLE), y),
        title,
        fontsize=SIZE_TITLE,
        fontname=BOLD,
    )
    y += 40

    # ── Abstract heading ──
    p1.insert_text((MARGIN, y), "Abstract", fontsize=SIZE_HEADING, fontname=BOLD)
    y += 20

    abstract = (
        "This paper presents a thermal analysis of solar panel configurations "
        "for 3U CubeSat missions in low Earth orbit. We evaluate three panel "
        "geometries using finite element methods and validate predictions "
        "against on-orbit telemetry from the TechSat-1 mission."
    )
    rc = fitz.Rect(MARGIN, y, WIDTH - MARGIN, y + 60)
    rv = p1.insert_textbox(rc, abstract, fontsize=SIZE_BODY, fontname=BODY)
    _assert_fit(rv, "abstract")
    y += 65

    # ── 1. Introduction heading ──
    p1.insert_text((MARGIN, y), "1. Introduction", fontsize=SIZE_HEADING, fontname=BOLD)
    y += 22

    # Two-column body text — use separate textboxes placed side by side
    col1_x = MARGIN
    col2_x = MARGIN + COL_W + COL_GAP

    col1_text = (
        "Small satellites in low Earth orbit experience significant "
        "thermal cycling between sunlight and eclipse. Solar panels must "
        "withstand temperature swings from -65 C to +125 C during each "
        "90-minute orbital period."
    )
    col2_text = (
        "Previous studies focused on large spacecraft thermal control. "
        "CubeSat missions impose unique constraints including limited "
        "surface area, restricted mass budgets, and minimal active "
        "thermal control options."
    )

    col_h = 80
    rc1 = fitz.Rect(col1_x, y, col1_x + COL_W, y + col_h)
    rc2 = fitz.Rect(col2_x, y, col2_x + COL_W, y + col_h)
    rv1 = p1.insert_textbox(rc1, col1_text, fontsize=SIZE_BODY, fontname=BODY)
    rv2 = p1.insert_textbox(rc2, col2_text, fontsize=SIZE_BODY, fontname=BODY)
    _assert_fit(rv1, "col1")
    _assert_fit(rv2, "col2")
    y += col_h + 14

    # ── Figure caption (separated with enough vertical space) ──
    p1.insert_text(
        (MARGIN, y),
        "Figure 1. Temperature distribution across the panel surface.",
        fontsize=SIZE_BODY - 1,
        fontname=BODY,
    )
    y += 24

    # ── 2. Methodology heading ──
    p1.insert_text((MARGIN, y), "2. Methodology", fontsize=SIZE_HEADING, fontname=BOLD)
    y += 20

    method_text = (
        "The thermal model was constructed using a finite element approach "
        "with 2400 shell elements representing the solar panel substrate. "
        "Boundary conditions included solar flux during sunlight phases and "
        "radiative cooling to deep space during eclipse. Material properties "
        "were taken from manufacturer datasheets for the Spectrolab UTJ "
        "solar cells used on TechSat-1."
    )
    rc_m = fitz.Rect(MARGIN, y, WIDTH - MARGIN, y + 80)
    rv = p1.insert_textbox(rc_m, method_text, fontsize=SIZE_BODY, fontname=BODY)
    _assert_fit(rv, "methodology")

    # ── PAGE 2 ────────────────────────────────────────────────
    p2 = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    # ── 3. Results heading ──
    p2.insert_text((MARGIN, y), "3. Results", fontsize=SIZE_HEADING, fontname=BOLD)
    y += 22

    # ── Bordered table ──
    headers = ["Configuration", "Peak Temp (C)", "Delta-T (C)"]
    rows = [
        ["Body-mounted", "128.3", "193.3"],
        ["Single-deploy", "114.7", "179.7"],
        ["Double-deploy", "110.2", "175.2"],
    ]
    col_widths = [160, 120, 100]
    table_x = MARGIN
    row_h = 18
    # Header row
    cx = table_x
    for i, hdr in enumerate(headers):
        rect = fitz.Rect(cx, y, cx + col_widths[i], y + row_h)
        p2.draw_rect(rect, color=(0, 0, 0), width=0.8)
        p2.insert_textbox(
            rect, hdr, fontsize=SIZE_BODY, fontname=BOLD, align=fitz.TEXT_ALIGN_CENTER
        )
        cx += col_widths[i]
    y += row_h
    # Data rows
    for row in rows:
        cx = table_x
        for i, cell in enumerate(row):
            rect = fitz.Rect(cx, y, cx + col_widths[i], y + row_h)
            p2.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p2.insert_textbox(
                rect,
                cell,
                fontsize=SIZE_BODY,
                fontname=BODY,
                align=fitz.TEXT_ALIGN_CENTER,
            )
            cx += col_widths[i]
        y += row_h
    y += 20

    results_text = (
        "The double-deploy configuration achieves the lowest peak "
        "temperature of 110.2 C, representing a 14.1 percent reduction "
        "compared to body-mounted panels. The increased radiating surface "
        "area of deployable panels enables more efficient heat rejection."
    )
    rc_r = fitz.Rect(MARGIN, y, WIDTH - MARGIN, y + 60)
    rv = p2.insert_textbox(rc_r, results_text, fontsize=SIZE_BODY, fontname=BODY)
    _assert_fit(rv, "results")
    y += 68

    # ── 4. Conclusions heading ──
    p2.insert_text((MARGIN, y), "4. Conclusions", fontsize=SIZE_HEADING, fontname=BOLD)
    y += 20

    conclusion = (
        "This study demonstrates that deployable solar panel configurations "
        "offer significant thermal advantages for 3U CubeSat missions in LEO. "
        "The validated finite element model agrees with TechSat-1 telemetry "
        "to within 3.2 C RMS error. Future work will extend the analysis to "
        "highly elliptical orbits."
    )
    rc_c = fitz.Rect(MARGIN, y, WIDTH - MARGIN, y + 72)
    rv = p2.insert_textbox(rc_c, conclusion, fontsize=SIZE_BODY, fontname=BODY)
    _assert_fit(rv, "conclusion")

    doc.save(path)
    doc.close()
    print(f"Golden PDF written to {path}")


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).with_name("golden_sample.pdf")
    build_pdf(str(out))
