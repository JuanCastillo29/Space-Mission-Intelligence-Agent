"""Download public space-mission PDFs for the ingestion pipeline.

Sources:
  - NASA Technical Reports Server (NTRS)  — public domain
  - arXiv preprints                       — open access
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s  %(message)s")
log = logging.getLogger(__name__)

DEST_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"

PDFS: list[tuple[str, str]] = [
    # ── NASA Technical Reports (NTRS) ──────────────────────────────────────
    (
        "nasa_human_exploration_mars_crew_tasks.pdf",
        "https://ntrs.nasa.gov/api/citations/20190001401/downloads/20190001401.pdf",
    ),
    (
        "nasa_apollo_gnc_hardware_overview.pdf",
        "https://ntrs.nasa.gov/api/citations/20090016290/downloads/20090016290.pdf",
    ),
    (
        "nasa_gateway_nrho_reference_trajectory.pdf",
        "https://ntrs.nasa.gov/api/citations/20190030294/downloads/20190030294.pdf",
    ),
    (
        "nasa_sls_mission_planners_guide.pdf",
        "https://ntrs.nasa.gov/api/citations/20170005323/downloads/20170005323.pdf",
    ),
    (
        "nasa_nuclear_thermal_propulsion_mars.pdf",
        "https://ntrs.nasa.gov/api/citations/20140013259/downloads/20140013259.pdf",
    ),
    (
        "nasa_breakthrough_propulsion_interstellar.pdf",
        "https://ntrs.nasa.gov/api/citations/20180006480/downloads/20180006480.pdf",
    ),
    (
        "nasa_helical_engine.pdf",
        "https://ntrs.nasa.gov/api/citations/20190029294/downloads/20190029294.pdf",
    ),
    (
        "nasa_reduction_space_launch_cost.pdf",
        "https://ntrs.nasa.gov/api/citations/20200001093/downloads/20200001093.pdf",
    ),
    # ── arXiv — planetary science & space missions ─────────────────────────
    (
        "arxiv_atmospheric_diversity_sub_neptunes.pdf",
        "https://arxiv.org/pdf/2606.20464",
    ),
    (
        "arxiv_giant_impact_lunar_isotopic_crisis.pdf",
        "https://arxiv.org/pdf/2606.20398",
    ),
    (
        "arxiv_jovian_xray_uv_solar_activity.pdf",
        "https://arxiv.org/pdf/2606.20355",
    ),
    (
        "arxiv_melting_rocky_exoplanets.pdf",
        "https://arxiv.org/pdf/2606.20249",
    ),
    (
        "arxiv_warm_jupiters_tess_mahps.pdf",
        "https://arxiv.org/pdf/2606.20224",
    ),
    (
        "arxiv_kepler_plato_transit_timing.pdf",
        "https://arxiv.org/pdf/2606.19685",
    ),
    (
        "arxiv_planet_radio_emission_m_dwarfs.pdf",
        "https://arxiv.org/pdf/2606.19671",
    ),
    (
        "arxiv_fate_of_earth_sun_giant_phases.pdf",
        "https://arxiv.org/pdf/2606.19575",
    ),
    (
        "arxiv_wasp121b_atmospheric_asymmetries_jwst.pdf",
        "https://arxiv.org/pdf/2606.19487",
    ),
    (
        "arxiv_orbital_evolution_hierarchical_triples.pdf",
        "https://arxiv.org/pdf/2606.19456",
    ),
    # ── ESA Rosetta mission (comet 67P) ────────────────────────────────────
    (
        "esa_rosetta_surface_dynamics_67p.pdf",
        "https://arxiv.org/pdf/2509.11613",
    ),
    (
        "esa_rosetta_coma_evolution_67p.pdf",
        "https://arxiv.org/pdf/2507.13979",
    ),
    (
        "esa_rosetta_rotation_dynamics_cometary_nuclei.pdf",
        "https://arxiv.org/pdf/2507.06036",
    ),
    (
        "esa_rosetta_thermophysical_model_67p.pdf",
        "https://arxiv.org/pdf/2505.14016",
    ),
    (
        "esa_rosetta_refractory_ice_ratio_67p.pdf",
        "https://arxiv.org/pdf/2501.17864",
    ),
    (
        "esa_rosetta_boulder_migration_67p.pdf",
        "https://arxiv.org/pdf/2411.17108",
    ),
    (
        "esa_rosetta_dimethyl_sulfide_cometary.pdf",
        "https://arxiv.org/pdf/2410.08724",
    ),
    (
        "esa_rosetta_exocomets_evolution.pdf",
        "https://arxiv.org/pdf/2604.08190",
    ),
    # ── ESA Mars Express / ExoMars TGO ─────────────────────────────────────
    (
        "esa_mars_express_arsia_mons_cloud.pdf",
        "https://arxiv.org/pdf/2103.03919",
    ),
    (
        "esa_mars_express_minor_species_ml.pdf",
        "https://arxiv.org/pdf/2012.08175",
    ),
    (
        "esa_mars_express_venus_express_ephemeris.pdf",
        "https://arxiv.org/pdf/0906.2860",
    ),
    (
        "esa_exomars_tgo_co2_clouds.pdf",
        "https://arxiv.org/pdf/2406.01515",
    ),
    (
        "esa_exomars_tgo_water_ice_clouds.pdf",
        "https://arxiv.org/pdf/2208.11100",
    ),
    (
        "esa_exomars_tgo_dust_storm_2018.pdf",
        "https://arxiv.org/pdf/1912.08018",
    ),
    # ── ESA JUICE (Jupiter Icy Moons Explorer) ────────────────────────────
    (
        "esa_juice_pride_radio_interferometry.pdf",
        "https://arxiv.org/pdf/2509.02888",
    ),
    (
        "esa_juice_pride_operational.pdf",
        "https://arxiv.org/pdf/2408.14965",
    ),
    (
        "esa_juice_majis_spectral_calibration.pdf",
        "https://arxiv.org/pdf/2405.19021",
    ),
    # ── ESA Gaia ───────────────────────────────────────────────────────────
    (
        "esa_gaia_mission_overview.pdf",
        "https://arxiv.org/pdf/1609.04153",
    ),
    (
        "esa_gaia_focus_straylight_basic_angle.pdf",
        "https://arxiv.org/pdf/1608.00045",
    ),
    (
        "esa_gaia_cti_radiation_damage.pdf",
        "https://arxiv.org/pdf/2601.19353",
    ),
    # ── ESA BepiColombo (Mercury) ──────────────────────────────────────────
    (
        "esa_bepicolombo_venus_swingby_perturbation.pdf",
        "https://arxiv.org/pdf/2409.02015",
    ),
    (
        "esa_bepicolombo_mercury_solar_wind.pdf",
        "https://arxiv.org/pdf/2305.09498",
    ),
]


def download(name: str, url: str, dest: Path) -> bool:
    target = dest / name
    if target.exists() and target.stat().st_size > 1024:
        log.info("Already exists: %s", name)
        return True

    log.info("Downloading %s ...", name)
    result = subprocess.run(
        ["curl", "-fSL", "--max-time", "120", "-o", str(target), url],
        capture_output=True,
    )
    if result.returncode != 0:
        log.warning("Failed %s: %s", name, result.stderr.decode().strip())
        target.unlink(missing_ok=True)
        return False

    data = target.read_bytes()
    if len(data) < 1024 or not data[:5].startswith(b"%PDF"):
        log.warning("Skipping %s — not a valid PDF (%d bytes)", name, len(data))
        target.unlink(missing_ok=True)
        return False

    log.info("  saved %s (%.1f MB)", name, len(data) / 1_048_576)
    return True


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for name, url in PDFS:
        if download(name, url, DEST_DIR):
            ok += 1
        else:
            fail += 1
        time.sleep(0.5)

    log.info("Done: %d downloaded, %d failed out of %d", ok, fail, len(PDFS))
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
