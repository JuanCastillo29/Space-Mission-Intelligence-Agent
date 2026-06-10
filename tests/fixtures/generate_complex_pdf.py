"""Generate a large, multi-page PDF that stress-tests extraction and chunking.

Run once to produce complex_sample.pdf.

Features exercised beyond golden_sample:
  - 8 pages with mixed layouts
  - Long sections that exceed chunk token limits (force splitting)
  - Multiple tables (bordered), including a large one
  - Nested heading hierarchy (3 levels deep)
  - Two-column layout sections
  - Multiple figure captions
  - Dense technical paragraphs with abbreviations (Fig., e.g., i.e.)
  - Content before first heading (preamble)
  - Very short sections (heading + 1 sentence)
  - Adjacent headings with no body between them
"""

import fitz

WIDTH, HEIGHT = 612, 792
MARGIN = 54
COL_GAP = 36
COL_W = (WIDTH - 2 * MARGIN - COL_GAP) / 2
LINE_H = 13

BODY = "helv"
BOLD = "hebo"
ITALIC = "heit"
SIZE_TITLE = 20
SIZE_H2 = 14
SIZE_H3 = 12
SIZE_H4 = 11
SIZE_BODY = 10
SIZE_CAPTION = 9


def _center_x(text, fontname, fontsize):
    w = fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
    return (WIDTH - w) / 2


def _textbox(
    page,
    x,
    y,
    w,
    h,
    text,
    fontsize=SIZE_BODY,
    fontname=BODY,
    align=fitz.TEXT_ALIGN_LEFT,
):
    rc = fitz.Rect(x, y, x + w, y + h)
    rv = page.insert_textbox(
        rc, text, fontsize=fontsize, fontname=fontname, align=align
    )
    if rv < 0:
        raise RuntimeError(f"Overflow by {-rv:.1f} pt: {text[:60]}...")
    return y + h - rv


def _heading(page, y, text, fontname=BOLD, fontsize=SIZE_H2):
    page.insert_text((MARGIN, y), text, fontsize=fontsize, fontname=fontname)
    return y + fontsize + 8


def _table(page, y, headers, rows, col_widths):
    row_h = 18
    cx = MARGIN
    for i, hdr in enumerate(headers):
        rect = fitz.Rect(cx, y, cx + col_widths[i], y + row_h)
        page.draw_rect(rect, color=(0, 0, 0), width=0.8)
        page.insert_textbox(
            rect, hdr, fontsize=SIZE_BODY, fontname=BOLD, align=fitz.TEXT_ALIGN_CENTER
        )
        cx += col_widths[i]
    y += row_h
    for row in rows:
        cx = MARGIN
        for i, cell in enumerate(row):
            rect = fitz.Rect(cx, y, cx + col_widths[i], y + row_h)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_textbox(
                rect,
                cell,
                fontsize=SIZE_BODY,
                fontname=BODY,
                align=fitz.TEXT_ALIGN_CENTER,
            )
            cx += col_widths[i]
        y += row_h
    return y + 10


def _figure_caption(page, y, text):
    page.insert_text((MARGIN, y), text, fontsize=SIZE_CAPTION, fontname=ITALIC)
    return y + 16


def build_pdf(path: str):
    doc = fitz.open()
    body_w = WIDTH - 2 * MARGIN

    # ── PAGE 1: Title page with preamble ─────────────────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 70

    title = "Orbital Debris Mitigation Strategies for LEO Constellations"
    p.insert_text(
        (_center_x(title, BOLD, SIZE_TITLE), y),
        title,
        fontsize=SIZE_TITLE,
        fontname=BOLD,
    )
    y += 36

    subtitle = (
        "A Comprehensive Analysis of Active Debris Removal and Collision Avoidance"
    )
    p.insert_text(
        (_center_x(subtitle, BODY, SIZE_H3), y),
        subtitle,
        fontsize=SIZE_H3,
        fontname=BODY,
    )
    y += 30

    # Preamble text before any heading
    preamble = (
        "This report was prepared by the European Space Agency (ESA) Space "
        "Debris Office in collaboration with NASA Orbital Debris Program Office. "
        "The analysis covers the period from 2020 to 2035 and incorporates data "
        "from the Space Surveillance Network (SSN), MASTER-8 debris environment "
        "model, and ESA DRAMA suite. Distribution is unlimited."
    )
    y = _textbox(p, MARGIN, y, body_w, 80, preamble)
    y += 10

    # Abstract
    y = _heading(p, y, "Abstract")
    abstract = (
        "The rapid deployment of large satellite constellations in low Earth "
        "orbit (LEO) poses unprecedented challenges for space sustainability. "
        "This study evaluates three classes of debris mitigation strategies: "
        "passive compliance with the 25-year rule, active debris removal (ADR) "
        "using robotic servicers, and enhanced collision avoidance manoeuvres "
        "guided by conjunction data messages (CDMs). Using Monte Carlo "
        "simulations of the LEO environment from 2025 to 2100, we quantify "
        "the long-term effectiveness of each approach. Results indicate that "
        "a combined strategy reducing post-mission disposal failures below "
        "5 percent and removing 5 high-risk objects per year stabilises the "
        "debris population growth rate. Without intervention, catastrophic "
        "collisions in the 800-1000 km altitude band are projected to increase "
        "by 40 percent per decade."
    )
    y = _textbox(p, MARGIN, y, body_w, 150, abstract)
    y += 6

    # 1. Introduction
    y = _heading(p, y, "1. Introduction")
    intro_1 = (
        "Space debris represents one of the most pressing challenges facing "
        "the space industry in the 21st century. Since the launch of Sputnik "
        "in 1957, over 13,000 metric tonnes of material have been placed in "
        "orbit, and only a fraction has been actively deorbited. The U.S. Space "
        "Surveillance Network tracks approximately 36,500 objects larger than "
        "10 cm, but the population of fragments between 1 mm and 10 cm is "
        "estimated at over 130 million (ESA Annual Space Environment Report, "
        "2024). Each of these objects carries sufficient kinetic energy to "
        "damage or destroy an operational spacecraft."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, intro_1)
    y += 4

    intro_2 = (
        "The Kessler Syndrome, first described by Kessler and Cour-Palais in "
        "1978, predicts a cascading chain of collisions that could render "
        "certain orbital regimes unusable within decades. Recent studies (e.g. "
        "Liou et al., 2023) suggest that the critical density threshold has "
        "already been exceeded in the 750-900 km altitude band. The 2009 "
        "Iridium-Cosmos collision and the 2007 Chinese ASAT test generated "
        "over 7,000 trackable fragments that continue to pose conjunction "
        "risks to operational satellites."
    )
    y = _textbox(p, MARGIN, y, body_w, 100, intro_2)

    # ── PAGE 2: Introduction continued + 1.1 + 1.2 ──────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    intro_3 = (
        "The proliferation of mega-constellations — Starlink (currently over "
        "6,000 satellites), OneWeb (648 planned), and Amazon Kuiper (3,236 "
        "planned) — adds urgency to the debris problem. While these operators "
        "have committed to post-mission disposal, historical data shows that "
        "even well-designed satellites experience end-of-life failures at rates "
        "between 5 and 15 percent. At constellation scales, this translates "
        "to hundreds of uncontrolled derelict objects per operator per decade."
    )
    y = _textbox(p, MARGIN, y, body_w, 100, intro_3)
    y += 6

    # 1.1 Scope and Objectives — nested heading
    y = _heading(p, y, "1.1 Scope and Objectives", fontsize=SIZE_H3)
    scope = (
        "This study addresses three research questions: (1) What is the "
        "minimum active debris removal rate required to stabilise the LEO "
        "population? (2) How does improved conjunction assessment accuracy "
        "affect collision probability? (3) What is the cost-effectiveness "
        "ratio of ADR versus enhanced collision avoidance for constellation "
        "operators? The analysis is restricted to the 200-2000 km altitude "
        "regime and considers objects with radar cross-section (RCS) greater "
        "than 0.01 square metres."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, scope)
    y += 6

    # 1.2 Related Work
    y = _heading(p, y, "1.2 Related Work", fontsize=SIZE_H3)
    related = (
        "Previous analyses by the Inter-Agency Space Debris Coordination "
        "Committee (IADC) established the 25-year post-mission disposal "
        "guideline. Liou and Johnson (2009) demonstrated that ADR of 5 "
        "objects per year is necessary to prevent runaway debris growth. "
        "More recent work by Virgili et al. (2016) extended this to "
        "constellation scenarios. Our study builds on these foundations "
        "by incorporating updated launch traffic models and improved "
        "fragmentation algorithms from the NASA Standard Breakup Model "
        "(SBM) revision 2023."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, related)
    y += 6

    # Figure caption
    y = _figure_caption(
        p,
        y,
        "Figure 1. Projected debris population growth "
        "under baseline and mitigation scenarios (2025-2100).",
    )
    y += 10

    # 1.3 Document Structure — very short section
    y = _heading(p, y, "1.3 Document Structure", fontsize=SIZE_H3)
    doc_struct = (
        "Section 2 describes the simulation methodology. Section 3 presents "
        "results for each mitigation strategy. Section 4 discusses "
        "cost-effectiveness. Section 5 concludes with recommendations."
    )
    y = _textbox(p, MARGIN, y, body_w, 50, doc_struct)

    # ── PAGE 3: Methodology (two-column section) ─────────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    y = _heading(p, y, "2. Methodology")

    # 2.1 Simulation Framework
    y = _heading(p, y, "2.1 Simulation Framework", fontsize=SIZE_H3)

    # Two-column layout
    col1_x = MARGIN
    col2_x = MARGIN + COL_W + COL_GAP

    col1_text = (
        "The debris environment evolution was modelled using a modified "
        "version of ESA's MASTER-8 (Meteoroid and Space Debris Terrestrial "
        "Environment Reference) tool coupled with a custom Monte Carlo "
        "propagator. The simulation initialises from the catalogued population "
        "as of January 2025 and propagates forward to 2100 in monthly time "
        "steps. Each Monte Carlo run samples from probability distributions "
        "for launch rates, post-mission disposal success rates, and "
        "fragmentation event parameters."
    )
    col2_text = (
        "Orbital propagation uses the SGP4/SDP4 analytical theory with "
        "corrections for atmospheric drag (NRLMSISE-00 model), solar "
        "radiation pressure, and lunisolar perturbations. Conjunction "
        "assessment employs the probability of collision (Pc) formulation "
        "by Alfano (2005) with covariance realism factors derived from "
        "comparison against precision ephemerides. A total of 1,000 Monte "
        "Carlo iterations were performed for each scenario, requiring "
        "approximately 48 hours on a 256-core HPC cluster."
    )

    col_h = 130
    rc1 = fitz.Rect(col1_x, y, col1_x + COL_W, y + col_h)
    rc2 = fitz.Rect(col2_x, y, col2_x + COL_W, y + col_h)
    p.insert_textbox(rc1, col1_text, fontsize=SIZE_BODY, fontname=BODY)
    p.insert_textbox(rc2, col2_text, fontsize=SIZE_BODY, fontname=BODY)
    y += col_h + 10

    y = _figure_caption(
        p,
        y,
        "Figure 2. Monte Carlo simulation architecture "
        "showing coupling between debris environment model and "
        "conjunction assessment module.",
    )
    y += 10

    # 2.2 Mitigation Scenarios
    y = _heading(p, y, "2.2 Mitigation Scenarios", fontsize=SIZE_H3)
    scenarios = (
        "Five distinct scenarios were evaluated, ranging from business-as-usual "
        "to aggressive combined mitigation. Each scenario was run with both "
        "nominal and pessimistic launch traffic projections."
    )
    y = _textbox(p, MARGIN, y, body_w, 50, scenarios)
    y += 4

    # Scenario table
    y = _table(
        p,
        y,
        ["Scenario", "PMD Rate", "ADR/Year", "CA Threshold"],
        [
            ["S0: Baseline", "60%", "0", "1e-4"],
            ["S1: Improved PMD", "95%", "0", "1e-4"],
            ["S2: ADR Only", "60%", "5", "1e-4"],
            ["S3: Enhanced CA", "60%", "0", "1e-5"],
            ["S4: Combined", "95%", "5", "1e-5"],
        ],
        [140, 110, 100, 130],
    )
    y += 6

    # 2.3 Fragmentation Model
    y = _heading(p, y, "2.3 Fragmentation Model", fontsize=SIZE_H3)
    frag = (
        "Collision and explosion events were modelled using the NASA Standard "
        "Breakup Model (SBM) as revised in 2023. The SBM generates fragment "
        "populations based on the mass and relative velocity of the colliding "
        "objects. For catastrophic collisions (specific energy > 40 J/g), "
        "the model produces approximately 1,000 trackable fragments per event."
    )
    y = _textbox(p, MARGIN, y, body_w, 80, frag)

    # ── PAGE 4: More methodology + deep nesting ──────────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    # 2.3.1 — level 4 heading
    y = _heading(p, y, "2.3.1 Size Distribution", fontsize=SIZE_H4)
    size_dist = (
        "The cumulative number of fragments N larger than characteristic "
        "length Lc follows a power law: N(Lc) = 0.1 * M^0.75 * Lc^(-1.71) "
        "for Lc > 0.08 m, where M is the total mass of the colliding objects "
        "in kilograms. For smaller fragments, the exponent transitions to "
        "-1.38. This bilinear model was validated against ground hypervelocity "
        "impact tests at the Fraunhofer EMI facility."
    )
    y = _textbox(p, MARGIN, y, body_w, 80, size_dist)
    y += 6

    # 2.3.2
    y = _heading(p, y, "2.3.2 Area-to-Mass Ratio", fontsize=SIZE_H4)
    amr = (
        "Each generated fragment is assigned an area-to-mass ratio (A/M) "
        "sampled from a log-normal distribution. The distribution parameters "
        "depend on fragment size: larger fragments (Lc > 1 m) tend to have "
        "lower A/M ratios, while small fragments exhibit high variability. "
        "The A/M ratio directly affects the fragment's ballistic coefficient "
        "and hence its orbital lifetime under atmospheric drag."
    )
    y = _textbox(p, MARGIN, y, body_w, 80, amr)
    y += 6

    # 2.4 Cost Model
    y = _heading(p, y, "2.4 Cost Model", fontsize=SIZE_H3)
    cost_intro = (
        "The economic analysis employs a parametric cost model calibrated "
        "against historical mission costs. Active debris removal missions "
        "are estimated at EUR 100-200 million per target, depending on the "
        "technology readiness level (TRL) and target orbit. Enhanced collision "
        "avoidance imposes recurring costs through propellant consumption "
        "and operational overhead."
    )
    y = _textbox(p, MARGIN, y, body_w, 80, cost_intro)
    y += 4

    # Cost table
    y = _table(
        p,
        y,
        ["Component", "Unit Cost (EUR M)", "Annual Cost (EUR M)"],
        [
            ["ADR mission (chaser)", "150", "750"],
            ["ADR target selection", "2", "10"],
            ["CA manoeuvre (per sat)", "0.05", "15"],
            ["SSA data subscription", "5", "5"],
            ["Ground segment ops", "10", "10"],
        ],
        [180, 130, 140],
    )
    y += 6

    y = _figure_caption(
        p,
        y,
        "Figure 3. Cost breakdown by mitigation "
        "component for Scenario S4 over a 10-year horizon.",
    )
    y += 10

    cost_disc = (
        "All costs are expressed in 2024 EUR and discounted at 3 percent "
        "per annum. Technology learning curves for ADR are modelled using "
        "a progress ratio of 0.85, i.e. costs decrease by 15 percent for "
        "each doubling of cumulative missions. The model accounts for "
        "insurance cost reductions achieved through lower collision "
        "probabilities under each scenario."
    )
    y = _textbox(p, MARGIN, y, body_w, 80, cost_disc)

    # ── PAGE 5: Results section (long, forces chunk splitting) ──
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    y = _heading(p, y, "3. Results")

    # 3.1 Debris Population Evolution
    y = _heading(p, y, "3.1 Debris Population Evolution", fontsize=SIZE_H3)
    pop_1 = (
        "The baseline scenario (S0) projects a 78 percent increase in the "
        "trackable debris population by 2100, from 36,500 to approximately "
        "65,000 objects. The growth is dominated by collisional cascading in "
        "the 750-900 km altitude band, where the spatial density already "
        "exceeds the critical threshold identified by Liou (2006). Under this "
        "scenario, an average of 0.8 catastrophic collisions per year are "
        "predicted in the 2070-2100 period, compared to 0.2 per year in "
        "the current epoch."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, pop_1)
    y += 4

    pop_2 = (
        "Improving post-mission disposal compliance to 95 percent (S1) "
        "reduces the 2100 population to 48,000 objects — a significant "
        "improvement but insufficient to prevent continued growth. The "
        "collision rate decreases to 0.5 per year, demonstrating that PMD "
        "alone cannot stabilise the environment. This finding is consistent "
        "with previous IADC studies and underscores the need for active "
        "measures."
    )
    y = _textbox(p, MARGIN, y, body_w, 90, pop_2)
    y += 4

    pop_3 = (
        "The ADR-only scenario (S2) achieves a more modest population of "
        "52,000 objects by 2100. Removing 5 high-mass derelicts per year "
        "from sun-synchronous orbits eliminates approximately 15,000 tonnes "
        "of potential collision mass over the simulation period. The collision "
        "rate drops to 0.4 per year, but the continued low PMD compliance "
        "introduces new derelicts faster than ADR can remove them."
    )
    y = _textbox(p, MARGIN, y, body_w, 90, pop_3)
    y += 4

    # Large results table
    y = _table(
        p,
        y,
        ["Scenario", "Pop. 2050", "Pop. 2100", "Coll./Year", "Delta %"],
        [
            ["S0: Baseline", "45,200", "65,000", "0.80", "—"],
            ["S1: PMD 95%", "40,100", "48,000", "0.50", "-26%"],
            ["S2: ADR Only", "42,800", "52,000", "0.40", "-20%"],
            ["S3: CA Only", "44,900", "63,500", "0.75", "-2%"],
            ["S4: Combined", "37,500", "38,200", "0.15", "-41%"],
        ],
        [120, 90, 90, 90, 80],
    )

    # ── PAGE 6: Results continued ────────────────────────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    pop_4 = (
        "The combined mitigation scenario (S4) is the only strategy that "
        "achieves near-stabilisation of the debris population. The 2100 "
        "population of 38,200 objects represents a mere 5 percent increase "
        "from the current catalogue. More importantly, the catastrophic "
        "collision rate drops to 0.15 per year — an 81 percent reduction "
        "compared to baseline. The synergy between high PMD compliance and "
        "active removal is substantial: the combined effect exceeds the sum "
        "of individual contributions by approximately 12 percent."
    )
    y = _textbox(p, MARGIN, y, body_w, 100, pop_4)
    y += 6

    y = _figure_caption(
        p,
        y,
        "Figure 4. Debris population trajectories for "
        "all five scenarios with 95% confidence intervals.",
    )
    y += 10

    # 3.2 Collision Avoidance Effectiveness
    y = _heading(p, y, "3.2 Collision Avoidance Effectiveness", fontsize=SIZE_H3)
    ca_1 = (
        "Enhanced collision avoidance (S3) with a lower Pc threshold of "
        "1e-5 increases the annual manoeuvre count per satellite from 2.1 "
        "to 8.7. While this reduces the realised collision rate by only "
        "7 percent, it provides a critical safety margin for high-value "
        "assets. The analysis reveals a diminishing returns curve: lowering "
        "the threshold from 1e-4 to 1e-5 captures 90 percent of avoidable "
        "conjunctions, but further reduction to 1e-6 would increase "
        "manoeuvres by a factor of 4 with minimal additional benefit."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, ca_1)
    y += 4

    ca_2 = (
        "Covariance realism remains the dominant source of uncertainty in "
        "collision probability estimation. Fig. 4 shows that 23 percent of "
        "high-Pc events were false alarms attributable to covariance inflation. "
        "Improving tracking accuracy through dedicated SSA sensors (e.g. the "
        "ESA Space Surveillance and Tracking programme) could reduce false "
        "alarms by up to 60 percent, significantly lowering the operational "
        "burden on constellation operators."
    )
    y = _textbox(p, MARGIN, y, body_w, 90, ca_2)
    y += 4

    # CA statistics table
    y = _table(
        p,
        y,
        ["Pc Threshold", "Manoeuvres/Sat/Year", "False Alarm %", "Fuel (kg/yr)"],
        [
            ["1e-3", "0.4", "5%", "0.02"],
            ["1e-4", "2.1", "12%", "0.11"],
            ["1e-5", "8.7", "23%", "0.45"],
            ["1e-6", "34.2", "41%", "1.78"],
        ],
        [110, 130, 110, 110],
    )

    # ── PAGE 7: Discussion ───────────────────────────────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    y = _heading(p, y, "4. Discussion")

    # 4.1 Cost-Effectiveness
    y = _heading(p, y, "4.1 Cost-Effectiveness Analysis", fontsize=SIZE_H3)
    cost_1 = (
        "The net present value (NPV) analysis reveals that Scenario S4 "
        "(combined mitigation) has the highest upfront cost at EUR 7.8 "
        "billion over 30 years, but delivers the greatest benefit when "
        "collision damage avoidance is valued at EUR 500 million per "
        "catastrophic event. The benefit-cost ratio (BCR) for S4 is 2.3, "
        "compared to 1.8 for S1 (PMD only) and 1.1 for S2 (ADR only). "
        "Enhanced collision avoidance (S3) alone has a BCR below 1.0, "
        "indicating that it is not cost-effective as a standalone strategy."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, cost_1)
    y += 4

    cost_2 = (
        "Sensitivity analysis shows that the BCR of S4 remains above 1.5 "
        "even under pessimistic assumptions (ADR cost of EUR 250 million "
        "per target, no learning curve, collision damage valued at EUR 200 "
        "million). The key cost driver is the ADR mission frequency: "
        "increasing from 5 to 10 removals per year raises the 30-year NPV "
        "cost by EUR 4.2 billion but reduces the collision rate by only an "
        "additional 0.05 per year, yielding diminishing marginal returns."
    )
    y = _textbox(p, MARGIN, y, body_w, 110, cost_2)
    y += 6

    # 4.2 Policy Implications
    y = _heading(p, y, "4.2 Policy Implications", fontsize=SIZE_H3)
    policy = (
        "These results have several implications for space debris policy. "
        "First, the 25-year post-mission disposal guideline should be "
        "tightened to 5 years for LEO orbits below 600 km, where "
        "atmospheric drag provides a natural de-orbit mechanism. Second, "
        "international cooperation on ADR funding is essential, as the "
        "benefits are non-excludable — a classic common-pool resource "
        "problem. Third, the establishment of a debris remediation fee, "
        "proportional to orbital lifetime and collision cross-section, "
        "could internalise the externality and incentivise responsible "
        "behaviour."
    )
    y = _textbox(p, MARGIN, y, body_w, 120, policy)
    y += 4

    policy_2 = (
        "The legal framework under the Outer Space Treaty (1967) and the "
        "Liability Convention (1972) currently assigns fault-based liability "
        "for space damage. However, the attribution of debris fragments to "
        "specific operators becomes practically impossible after a cascading "
        "collision event. We recommend that COPUOS consider a strict liability "
        "regime for debris-generating events, combined with mandatory "
        "financial guarantees for post-mission disposal."
    )
    y = _textbox(p, MARGIN, y, body_w, 100, policy_2)

    # ── PAGE 8: Conclusions + empty section edge case ────────
    p = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 60

    y = _heading(p, y, "5. Conclusions")
    conc_1 = (
        "This study demonstrates that no single mitigation strategy is "
        "sufficient to ensure the long-term sustainability of the LEO "
        "environment. The combined approach of high PMD compliance, "
        "targeted active debris removal, and enhanced collision avoidance "
        "(Scenario S4) is the only path that stabilises the debris population "
        "below the cascading threshold. The benefit-cost ratio of 2.3 "
        "confirms that proactive mitigation is economically justified, "
        "even under conservative assumptions."
    )
    y = _textbox(p, MARGIN, y, body_w, 100, conc_1)
    y += 4

    y = _textbox(p, MARGIN, y, body_w, 80, conc_1)
    y += 6

    # Adjacent headings with no body (edge case)
    y = _heading(p, y, "6. Future Work")
    y = _heading(p, y, "6.1 Planned Extensions", fontsize=SIZE_H3)
    future = (
        "Future work will extend the simulation to include GEO and MEO "
        "regimes, incorporate machine learning approaches for conjunction "
        "screening, and evaluate novel ADR technologies including ion-beam "
        "shepherd and electrodynamic tether concepts."
    )
    y = _textbox(p, MARGIN, y, body_w, 60, future)
    y += 6

    # Acknowledgements (short)
    y = _heading(p, y, "Acknowledgements")
    ack = (
        "The authors thank the ESA Space Debris Office, the NASA ODPO, "
        "and the IADC member agencies for providing data and computational "
        "resources. This work was funded under ESA Contract No. "
        "4000138721/22/D/MRP."
    )
    y = _textbox(p, MARGIN, y, body_w, 60, ack)

    page_count = doc.page_count
    doc.save(path)
    doc.close()
    print(f"Complex PDF written to {path}  ({page_count} pages)")


if __name__ == "__main__":
    import pathlib

    out = pathlib.Path(__file__).with_name("complex_sample.pdf")
    build_pdf(str(out))
