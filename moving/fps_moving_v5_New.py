import json
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

# import from moving_utils_New;
# window constants renamed, SECONDS_PER_YEAR added for deg/yr velocities, and
# the production region module now supplies all clipped areas.
from moving_utils_New import (
    WINDOW_END,
    WINDOW_START,
    ROI_DEC as DEFAULT_ROI_DEC,
    ROI_RA as DEFAULT_ROI_RA,
    SECONDS_PER_DAY,
    SECONDS_PER_YEAR,
    angular_separation_degrees,
    analysis_bin_edges,
    event_class_pass,
    lima_sigma,
    psf_radius_degrees,
    spherical_mean_coordinates,
)
from moving.analysis_region_v5 import (
    DEFAULT_MC_SAMPLES,
    AnalysisRegion,
    estimate_spherical_area,
    load_ownership_centers,
    unit_vectors,
    unmasked_photon_count,
)
from moving.adaptive_seeding_v5 import adaptive_density_labels
from moving.coherent_null import coherent_scrambled_times


BIN_TIME = float(os.environ.get("FPS_BIN_TIME_DAYS") or 365.0)
MIN_TRACK_LENGTH = int(os.environ.get("FPS_MIN_TRACK_LENGTH") or 5)
# the velocity-consistency
# cut (min/max projected step ratio >= 0.6) rejected a PERFECT synthetic
# constant-velocity mover: per-bin centroid noise (~0.15 deg) on ~0.5 deg
# annual steps yields ratios of 0.3-0.5 even with zero acceleration, so the
# cut measured noise, not physics, and could never pass a real faint source.
# Default is now 0 (metric still computed and reported; anisotropy >= 0.7 and
# time-correlation >= 0.85 remain the smoothness discriminators - the same
# synthetic track scores 0.993 and 0.998 on those). Re-enable via env only
# with a threshold calibrated on injections. ---
MIN_VELOCITY_CONSISTENCY = float(
    os.environ.get("FPS_MIN_VELOCITY_CONSISTENCY") or 0.0
)

ROI_RA = float(os.environ.get("FPS_ROI_RA") or DEFAULT_ROI_RA)
ROI_DEC = float(os.environ.get("FPS_ROI_DEC") or DEFAULT_ROI_DEC)
if "FPS_ROI_RADIUS_DEG" not in os.environ:
    raise RuntimeError(
        "FPS_ROI_RADIUS_DEG is required: the data boundary must come from "
        "the tile's query_info.json, never from a hard-coded fallback."
    )
ANALYSIS_ROI_RADIUS_DEGREES = float(os.environ["FPS_ROI_RADIUS_DEG"])
EVENTS_FILE = os.environ.get("FPS_EVENTS_FILE") or None
OUTPUT_LABEL = os.environ.get("FPS_OUTPUT_LABEL") or "annual"
SAVE_PRE_TUBE = os.environ.get("FPS_SAVE_PRE_TUBE", "") == "1"

# one explicit BUFFER analysis region and one
# nearest-centre ownership rule.  The data boundary is used for density and
# on/off areas; the Voronoi cell is used only for final track ownership. ---
OWNERSHIP_CENTERS = os.environ.get("FPS_OWNERSHIP_CENTERS") or None
OWNERSHIP_TILE_ID = os.environ.get("FPS_OWNERSHIP_TILE_ID") or None
if bool(OWNERSHIP_CENTERS) != bool(OWNERSHIP_TILE_ID):
    raise ValueError(
        "FPS_OWNERSHIP_CENTERS and FPS_OWNERSHIP_TILE_ID must be set together."
    )
if OWNERSHIP_CENTERS:
    ownership_path = Path(OWNERSHIP_CENTERS)
    if not ownership_path.is_absolute():
        ownership_path = PROJECT_ROOT / ownership_path
    OWNERSHIP_TILE_IDS, OWNERSHIP_CENTER_VECTORS = load_ownership_centers(
        ownership_path
    )
    if OWNERSHIP_TILE_ID not in OWNERSHIP_TILE_IDS:
        raise ValueError(f"Unknown ownership tile {OWNERSHIP_TILE_ID}")
else:
    OWNERSHIP_TILE_IDS, OWNERSHIP_CENTER_VECTORS = (), None
ANALYSIS_REGION = AnalysisRegion(
    tile_id=OWNERSHIP_TILE_ID or OUTPUT_LABEL,
    center_ra_degrees=ROI_RA,
    center_dec_degrees=ROI_DEC,
    data_radius_degrees=ANALYSIS_ROI_RADIUS_DEGREES,
    ownership_tile_ids=OWNERSHIP_TILE_IDS,
    ownership_center_vectors=OWNERSHIP_CENTER_VECTORS,
)
MC_AREA_SAMPLES = int(
    os.environ.get("FPS_MC_AREA_SAMPLES") or DEFAULT_MC_SAMPLES
)
if MC_AREA_SAMPLES < 1:
    raise ValueError("FPS_MC_AREA_SAMPLES must be positive")
MAX_MC_AREA_SAMPLES = int(
    os.environ.get("FPS_MAX_MC_AREA_SAMPLES") or 262_144
)
MAX_MC_TO_POISSON_RATIO = float(
    os.environ.get("FPS_MAX_MC_TO_POISSON_RATIO") or 0.20
)
if MAX_MC_AREA_SAMPLES < MC_AREA_SAMPLES:
    raise ValueError("FPS_MAX_MC_AREA_SAMPLES must be >= FPS_MC_AREA_SAMPLES")
if not 0.0 < MAX_MC_TO_POISSON_RATIO < 1.0:
    raise ValueError("FPS_MAX_MC_TO_POISSON_RATIO must lie in (0, 1)")


MIN_PHOTONS_PER_BAND = 10

# the injection
# test showed the fixed per-band DBSCAN seeding (fixed min_samples = 5 in six
# narrow bands) was the sensitivity bottleneck: a real source's photons were
# split across bands so no single band reached the core condition, while the
# 1-2.2 GeV band percolated into one ROI-sized cluster every year. Redesign:
#
# 1. SEED_BANDS merges 2.2-10 GeV into ONE seeding band (a faint source's
#    photons pool instead of splitting), with eps set by the PSF of the
#    softest third of the merged band so soft photons are still captured.
#    The 1-100 GeV reporting bands (BAND_LABELS below) are unchanged.
# 2. min_samples is density-scaled continuously per band as
#    mu + 2.5*sqrt(mu) + 1 (including the core photon), with a floor of four.
#    This avoids the discontinuous jump from four to nine at mu=3.
# 3. The floor is loosened 5 -> 4: cluster FORMATION is relaxed while the
#    Li-Ma significance cut still gates which clusters become candidates.
#    False tracks that leak through are the job of the scramble-calibrated
#    likelihood stage - by design (loose seeding, tight confirmation).
#
# Env overrides: FPS_MIN_SAMPLES (floor), FPS_EPS_SCALE, FPS_MIN_SIGMA. ---
MIN_SAMPLES_FLOOR = int(os.environ.get("FPS_MIN_SAMPLES") or 4)
EPS_SCALE = float(os.environ.get("FPS_EPS_SCALE") or 1.0)
MIN_SIGNIFICANCE_ENV = float(os.environ.get("FPS_MIN_SIGMA") or 2.5)
ADAPTIVE_LOCAL_SEEDING = (
    os.environ.get("FPS_ADAPTIVE_LOCAL_SEEDING", "1") == "1"
)
SAVE_BIN_CANDIDATES = os.environ.get("FPS_SAVE_BIN_CANDIDATES", "") == "1"
INJECTION_STAGE_AUDIT_TRUTH_FILE = os.environ.get(
    "FPS_INJECTION_STAGE_AUDIT_TRUTH"
)
INJECTION_STAGE_AUDIT_MATCH_RADIUS_DEGREES = 0.6
if INJECTION_STAGE_AUDIT_TRUTH_FILE:
    _stage_audit_truth = json.loads(
        Path(INJECTION_STAGE_AUDIT_TRUTH_FILE).read_text()
    )
    INJECTION_STAGE_AUDIT_TRUTH_BY_BIN = {
        int(point["bin_index"]): point
        for point in _stage_audit_truth["per_bin_truth"]
    }
    INJECTION_STAGE_AUDIT = {
        "schema_version": 1,
        "truth_file": str(Path(INJECTION_STAGE_AUDIT_TRUTH_FILE).resolve()),
        "truth_match_radius_degrees": (
            INJECTION_STAGE_AUDIT_MATCH_RADIUS_DEGREES
        ),
        "band_records": [],
        "linking": {},
    }
else:
    INJECTION_STAGE_AUDIT_TRUTH_BY_BIN = {}
    INJECTION_STAGE_AUDIT = None
# explicit stationary-control execution mode. ---
STATIONARY_VALIDATION = (
    os.environ.get("FPS_STATIONARY_VALIDATION", "") == "1"
)


# Seeding bands: (E_min_MeV, E_max_MeV, eps reference energy in MeV).
SEED_BANDS = [
    (1000.0, 2154.4, 1467.8),      # 1-2.2 GeV, kept alive by density scaling
    (2154.4, 10000.0, 3162.3),     # 2.2-10 GeV merged - the workhorse band
    (10000.0, 21544.3, 14678.0),
    (21544.3, 46415.9, 31623.0),
    (46415.9, 100000.0, 68129.0),
]
ROI_AREA_DEG2 = ANALYSIS_REGION.area_deg2


def band_min_samples(photon_count, epsilon_degrees):
    """Density-scaled DBSCAN core condition (see comment block above)."""
    mu_background = photon_count * np.pi * epsilon_degrees ** 2 / ROI_AREA_DEG2
    return max(
        MIN_SAMPLES_FLOOR,
        int(np.ceil(mu_background + 2.5 * np.sqrt(mu_background) + 1.0)),
    )

INNER_RADIUS_SCALE = 2.0
OUTER_RADIUS_SCALE = 5.0
MIN_DONUT_AREA = 1e-10
# env-overridable via FPS_MIN_SIGMA (default 2.5).
MIN_SIGNIFICANCE = MIN_SIGNIFICANCE_ENV
MAX_BIN_GAP = 2

# the maximum link distance was a flat
# 1 degree per BIN, so its physical meaning silently changed with
# FPS_BIN_TIME_DAYS. It is now expressed in degrees per YEAR and scaled by
# the bin length, which makes sub-annual binning (the recommended mode for
# resolving the annual parallax wobble of slow, Planet-9-like movers)
# self-consistent. Both the rate and the bin length are env-configurable.
BIN_YEARS = BIN_TIME / 365.0
MAX_LINK_DEG_PER_YEAR = float(
    os.environ.get("FPS_MAX_LINK_DEG_PER_YEAR") or 1.0
)
MAX_LINK_DISTANCE_DEGREES = MAX_LINK_DEG_PER_YEAR * BIN_YEARS


LINK_GAP_PENALTY = 0.5

# previously every
# transition could skip a bin, so a length-5 track could span 8 bins with 3
# missing detections (e.g. the ROI-19 candidate). A per-track skip budget now
# caps the total number of skipped bins (default 2), enforced when tracks are
# reconstructed. ---
MAX_TOTAL_SKIPPED_BINS = int(
    os.environ.get("FPS_MAX_TOTAL_SKIPPED_BINS") or 2
)
MAX_LINK_STATES_PER_SKIP = int(
    os.environ.get("FPS_MAX_LINK_STATES_PER_SKIP") or 4
)
if MAX_LINK_STATES_PER_SKIP < 1:
    raise ValueError("FPS_MAX_LINK_STATES_PER_SKIP must be positive.")


# null-calibration mode. Setting
# FPS_SCRAMBLE_TIMES=1 randomly permutes photon arrival times, destroying any
# real spatio-temporal correlation while preserving the spatial and temporal
# marginal distributions. Running the full pipeline on scrambled data
# measures the false-track rate, which is how the validation thresholds in
# check_v5_New.py should be calibrated (the chi-square asymptotics are
# invalid after the DP linker's look-elsewhere optimization). ---
SCRAMBLE_TIMES = os.environ.get("FPS_SCRAMBLE_TIMES", "") == "1"
SCRAMBLE_SEED = int(os.environ.get("FPS_SCRAMBLE_SEED") or 0)
NULL_TIME_MODEL = os.environ.get("FPS_NULL_TIME_MODEL") or None


# standard Fermi-LAT event-quality cuts, applied
# BEFORE clustering so the prefilter and the fermipy likelihood stage see the
# same photon selection. Zenith cut removes Earth-limb photons; the event
# class cut keeps P8R3 SOURCE (bit 128); the GTI cut uses each photon file's
# own GTI extension. All are env-configurable, and each cut is skipped
# gracefully if the file lacks the needed column. ---
ZMAX_DEGREES = float(os.environ.get("FPS_ZMAX_DEG") or 90.0)
EVCLASS_BIT = int(os.environ.get("FPS_EVCLASS_BIT") or 128)
APPLY_GTI = os.environ.get("FPS_APPLY_GTI", "1") == "1"
ALLOW_MISSING_QUALITY = (
    os.environ.get("FPS_ALLOW_MISSING_QUALITY", "") == "1"
)


# the tube filters
# assume a straight constant-velocity track; a slow mover dominated by the
# annual parallax wobble (resolved with sub-annual bins) is NOT straight, so
# these thresholds must be relaxable without editing code. They keep their
# previous defaults. ---
MIN_TUBE_ANISOTROPY = float(os.environ.get("FPS_MIN_TUBE_ANISOTROPY") or 0.7)
MIN_TUBE_TIME_CORRELATION = float(
    os.environ.get("FPS_MIN_TUBE_TIME_CORRELATION") or 0.85
)
DISABLE_MOTION_FILTER = os.environ.get("FPS_DISABLE_MOTION_FILTER", "") == "1"


CATALOG_EXCLUSION_RADIUS_DEGREES = float(
    os.environ.get("FPS_CATALOG_EXCLUSION_RADIUS_DEG") or 0.5
)
CATALOG_ON_REGION_VETO = (
    os.environ.get("FPS_CATALOG_ON_REGION_VETO", "1") == "1"
)
CATALOG_ANNULUS_MASK = os.environ.get(
    "FPS_CATALOG_ANNULUS_MASK", "1"
) == "1"
CATALOG_MASK_R68_SCALE = float(
    os.environ.get("FPS_CATALOG_MASK_R68_SCALE") or 2.0
)
CATALOG_MASK_MIN_RADIUS_DEGREES = float(
    os.environ.get("FPS_CATALOG_MASK_MIN_RADIUS_DEG") or 0.5
)
CATALOG_MASK_MIN_FLUX = float(
    os.environ.get("FPS_CATALOG_MASK_MIN_FLUX") or 0.0
)
MIN_UNMASKED_ANNULUS_FRACTION = float(
    os.environ.get("FPS_MIN_UNMASKED_ANNULUS_FRACTION") or 0.25
)
MIN_UNMASKED_OFF_COUNTS = int(
    os.environ.get("FPS_MIN_UNMASKED_OFF_COUNTS") or 10
)
if CATALOG_MASK_R68_SCALE < 0.0:
    raise ValueError("FPS_CATALOG_MASK_R68_SCALE cannot be negative")
if not 0.0 < MIN_UNMASKED_ANNULUS_FRACTION <= 1.0:
    raise ValueError("FPS_MIN_UNMASKED_ANNULUS_FRACTION must lie in (0, 1]")
CATALOG_LOAD_RADIUS_DEGREES = (
    ANALYSIS_ROI_RADIUS_DEGREES
    + OUTER_RADIUS_SCALE * EPS_SCALE * psf_radius_degrees(SEED_BANDS[0][2])
    + max(
        CATALOG_MASK_R68_SCALE
        * EPS_SCALE * psf_radius_degrees(SEED_BANDS[0][2]),
        CATALOG_MASK_MIN_RADIUS_DEGREES,
    )
)


RESULTS_ROOT = Path(
    os.environ.get("MOVING_RESULTS_ROOT", PROJECT_ROOT / "moving" / "results_v5")
)
if not RESULTS_ROOT.is_absolute():
    RESULTS_ROOT = PROJECT_ROOT / RESULTS_ROOT
RESULTS_DIR = RESULTS_ROOT / OUTPUT_LABEL

def apply_unique_sky_ownership(records):
    """Keep tracks whose spherical midpoint belongs to this Voronoi tile."""
    if not OWNERSHIP_CENTERS and not OWNERSHIP_TILE_ID:
        return records
    if not OWNERSHIP_CENTERS or not OWNERSHIP_TILE_ID:
        raise ValueError(
            "FPS_OWNERSHIP_CENTERS and FPS_OWNERSHIP_TILE_ID must be set together."
        )
    kept = []
    for record in records:
        ra, dec = spherical_mean_coordinates(
            np.asarray([point["ra"] for point in record["points"]]),
            np.asarray([point["dec"] for point in record["points"]]),
        )
        owner = ANALYSIS_REGION.owner_tile(ra, dec)
        record["owner_tile_id"] = owner
        record["ownership_ra_degrees"] = float(ra)
        record["ownership_dec_degrees"] = float(dec)
        if owner == OWNERSHIP_TILE_ID:
            kept.append(record)
    print(
        f"Unique sky ownership kept {len(kept)}/{len(records)} tracks "
        f"for {OWNERSHIP_TILE_ID}.",
        flush=True,
    )
    return kept

# the stationary mask was hard-wired to the
# 4FGL-DR3 loader even though the data window (2016-2026) extends six years
# past the data underlying DR3. If FPS_CATALOG_FITS points at a newer catalog
# FITS source catalogue (the production analysis uses FL16Y-v41), sources are read directly from
# it; otherwise the original loader is used as a fallback. ---
CATALOG_FITS = os.environ.get("FPS_CATALOG_FITS") or None


def load_catalog_sources(roi_ra, roi_dec, radius_degrees):
    if CATALOG_FITS:
        with fits.open(CATALOG_FITS) as catalog_hdu_list:
            catalog_data = catalog_hdu_list["LAT_Point_Source_Catalog"].data
            catalog_ra = np.asarray(catalog_data["RAJ2000"], dtype = float)
            catalog_dec = np.asarray(catalog_data["DEJ2000"], dtype = float)
            catalog_flux = np.asarray(catalog_data["Flux1000"], dtype = float)
            catalog_names = [
                value.decode(errors="replace").strip()
                if isinstance(value, bytes) else str(value).strip()
                for value in catalog_data["Source_Name"]
            ]
        separations = angular_separation_degrees(
            roi_ra, roi_dec, catalog_ra, catalog_dec,
        )
        keep = separations < radius_degrees
        return [
            {
                "ra": float(catalog_ra[index]),
                "dec": float(catalog_dec[index]),
                "flux1000": float(catalog_flux[index]),
                "source_name": catalog_names[index],
            }
            for index in np.flatnonzero(keep)
        ]
    from stationary.stationary_utils import load_4fgl_sources_in_roi
    sources = load_4fgl_sources_in_roi(
        roi_ra = roi_ra,
        roi_dec = roi_dec,
        roi_radius_degrees = radius_degrees,
    )
    for source in sources:
        source.setdefault("flux1000", float("nan"))
        source.setdefault("source_name", "")
    return sources


CATALOG_SOURCES = load_catalog_sources(
    ROI_RA, ROI_DEC, CATALOG_LOAD_RADIUS_DEGREES,
)


CATALOG_RA = np.asarray(
    [source["ra"] for source in CATALOG_SOURCES], dtype = float
)
CATALOG_DEC = np.asarray(
    [source["dec"] for source in CATALOG_SOURCES], dtype = float
)
CATALOG_FLUX = np.asarray(
    [source.get("flux1000", float("nan")) for source in CATALOG_SOURCES],
    dtype=float,
)
CATALOG_VECTORS = unit_vectors(CATALOG_RA, CATALOG_DEC)
print(
    f"Loaded {len(CATALOG_SOURCES)} catalog sources in ROI for masking "
    f"(on-region exclusion="
    f"{'on' if CATALOG_ON_REGION_VETO else 'OFF for stationary validation'}, "
    f"minimum radius={CATALOG_EXCLUSION_RADIUS_DEGREES:.2f} deg, "
    f"annulus mask={'on' if CATALOG_ANNULUS_MASK else 'off'}, "
    f"mask radius=max({CATALOG_MASK_R68_SCALE:g}*r68,"
    f"{CATALOG_MASK_MIN_RADIUS_DEGREES:g} deg), "
    f"minimum Flux1000={CATALOG_MASK_MIN_FLUX:.3g}, "
    f"catalog = {CATALOG_FITS or '4FGL-DR3 default loader'}).",
    flush = True,
)


def catalog_separations(centroid_ra, centroid_dec):
    if len(CATALOG_SOURCES) == 0:
        return np.empty(0, dtype=float)
    return np.asarray(angular_separation_degrees(
        centroid_ra, centroid_dec, CATALOG_RA, CATALOG_DEC,
    ), dtype=float)


def near_catalog_source(centroid_ra, centroid_dec, epsilon_degrees):
    """Reject catalogue-centred on regions, including the largest r68."""
    separations = catalog_separations(centroid_ra, centroid_dec)
    exclusion = (
        max(CATALOG_EXCLUSION_RADIUS_DEGREES, epsilon_degrees)
        if CATALOG_ANNULUS_MASK else CATALOG_EXCLUSION_RADIUS_DEGREES
    )
    # retain separations and annulus masking while
    # permitting the stationary control to keep catalogue-centred clusters. ---
    is_near = (
        bool(np.any(separations < exclusion))
        if CATALOG_ON_REGION_VETO else False
    )

    return is_near, separations, exclusion


AREA_DIAGNOSTICS = {
    "annulus_measurements": 0,
    "boundary_clipped_measurements": 0,
    "boundary_clip_fractions": [],
    "masked_area_fractions": [],
    "unstable_overmasked_measurements": 0,
    "on_region_sources_between_fixed_mask_and_r68": 0,
    "min_samples": [],
    "annulus_mc_samples": [],
    "annulus_mc_to_poisson_ratios": [],
}


def mc_area_to_poisson_ratio(estimate, n_off):
    """Area-fraction standard error relative to 1/sqrt(N_off)."""
    fraction = estimate.retained_fraction
    if n_off <= 0 or fraction <= 0.0 or fraction >= 1.0:
        return 0.0
    relative_area_error = np.sqrt(
        fraction * (1.0 - fraction) / estimate.n_samples
    ) / fraction
    return float(relative_area_error * np.sqrt(n_off))


def adaptive_annulus_area(
    centroid_ra,
    centroid_dec,
    inner_degrees,
    outer_degrees,
    mask_vectors,
    mask_radii,
    n_off,
):
    """Refine until MC area noise is <=20% of N_off Poisson noise."""
    n_samples = MC_AREA_SAMPLES
    while True:
        estimate = estimate_spherical_area(
            centroid_ra,
            centroid_dec,
            inner_degrees,
            outer_degrees,
            ANALYSIS_REGION,
            mask_center_vectors=mask_vectors,
            mask_radii_degrees=mask_radii,
            n_samples=n_samples,
        )
        ratio = mc_area_to_poisson_ratio(estimate, n_off)
        if ratio <= MAX_MC_TO_POISSON_RATIO or n_samples >= MAX_MC_AREA_SAMPLES:
            return estimate, ratio
        n_samples = min(2 * n_samples, MAX_MC_AREA_SAMPLES)


def candidate_output_path():
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    return str(RESULTS_DIR / "candidates.json")


def pre_tube_candidate_output_path():
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    return str(RESULTS_DIR / "debug_pre_tube_candidates.json")


# fps now writes the summary.json that the batch
# driver run_fps_moving_v5_rois expects; previously the driver crashed with
# FileNotFoundError after every ROI because no such file was ever written. ---
def summary_output_path():
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    return str(RESULTS_DIR / "summary.json")



def injection_stage_audit_separation(bin_index, ra, dec):
    """Separation from injected truth, or None outside audit mode."""
    truth = INJECTION_STAGE_AUDIT_TRUTH_BY_BIN.get(int(bin_index))
    if truth is None:
        return None
    return float(angular_separation_degrees(
        ra, dec, float(truth["ra"]), float(truth["dec"]),
    ))


def injection_stage_audit_track_matches(track):
    """Apply the frozen recovery match to one internal trajectory."""
    separations = []
    for point in track:
        separation = injection_stage_audit_separation(
            point["bin_index"], point["ra"], point["dec"],
        )
        if separation is not None:
            separations.append(separation)
    return (
        len(separations) >= 3
        and float(np.median(separations))
        <= INJECTION_STAGE_AUDIT_MATCH_RADIUS_DEGREES
    )


def write_injection_stage_audit():
    """Write opt-in truth-matched diagnostics without changing selection."""
    if INJECTION_STAGE_AUDIT is None:
        return
    records = INJECTION_STAGE_AUDIT["band_records"]
    stages = (
        "seeded", "survived_catalogue_veto",
        "stable_background", "survived_significance",
    )
    INJECTION_STAGE_AUDIT["per_bin_stage_counts"] = {
        stage: len({
            int(record["bin_index"])
            for record in records
            if any(cluster.get(stage, False)
                   for cluster in record["truth_matched_clusters"])
        })
        for stage in stages
    }
    output = RESULTS_DIR / "injection_stage_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(INJECTION_STAGE_AUDIT, indent=2))


# Create the list of photon files ready for ingestion.
EVENTS_FILE_PATH = (
    Path(EVENTS_FILE) if EVENTS_FILE
    else PROJECT_ROOT / "data" / "events.txt"
)
if not EVENTS_FILE_PATH.is_absolute():
    EVENTS_FILE_PATH = PROJECT_ROOT / EVENTS_FILE_PATH
with open(EVENTS_FILE_PATH, "r") as events_file:
    PHOTON_FILES = [
        str(PROJECT_ROOT / line.strip()) for line in events_file if line.strip()
    ]

if not PHOTON_FILES:
    raise RuntimeError("No photon files found.")

print(
    f"{len(PHOTON_FILES)} found!",
    flush = True,
)


def gti_pass(times, gti_start, gti_stop):
    if len(gti_start) == 0:
        raise ValueError("GTI extension contains no intervals.")
    order = np.argsort(gti_start)
    gti_start = np.asarray(gti_start, dtype = float)[order]
    gti_stop = np.asarray(gti_stop, dtype = float)[order]
    interval_index = np.searchsorted(gti_start, times, side = "right") - 1
    inside = interval_index >= 0
    clipped_index = np.clip(interval_index, 0, len(gti_start) - 1)
    inside &= times <= gti_stop[clipped_index]
    return inside



RIGHT_ASCENSION_ARRAYS = []
DECLINATION_ARRAYS = []
ENERGY_ARRAYS = []
TIME_ARRAYS = []
RUN_ID_ARRAYS = []
EVENT_ID_ARRAYS = []

# Extract RA, DEC, energy, time, with each element as a file
for photon_file_path in PHOTON_FILES:
    print(f"  Loading {photon_file_path}...", flush = True)
    with fits.open(photon_file_path) as header_data_unit_list:
        data = header_data_unit_list[1].data

        # apply zenith / event-class / GTI cuts
        # per file, so the clustering stage sees the same standard selection
        # (evclass=128, zmax=90, GTI-filtered) as the fermipy stage. ---
        column_names = data.columns.names
        file_mask = np.ones(len(data), dtype = bool)
        if "ZENITH_ANGLE" in column_names:
            file_mask &= np.asarray(
                data["ZENITH_ANGLE"], dtype = float
            ) < ZMAX_DEGREES
        elif not ALLOW_MISSING_QUALITY:
            raise RuntimeError(
                f"{photon_file_path} has no ZENITH_ANGLE column."
            )
        # FPS_EVCLASS_BIT=0 disables the
        # event-class cut - needed when overlaying gtobssim-simulated events,
        # which carry no real class bits.
        if EVCLASS_BIT > 0:
            if "EVENT_CLASS" in column_names:
                try:
                    file_mask &= event_class_pass(
                        data["EVENT_CLASS"], EVCLASS_BIT
                    )
                except ValueError:
                    if not ALLOW_MISSING_QUALITY:
                        raise
            elif not ALLOW_MISSING_QUALITY:
                raise RuntimeError(
                    f"{photon_file_path} has no EVENT_CLASS column."
                )
        if APPLY_GTI:
            try:
                gti_data = header_data_unit_list["GTI"].data
                file_mask &= gti_pass(
                    np.asarray(data["TIME"], dtype = float),
                    gti_data["START"],
                    gti_data["STOP"],
                )
            except KeyError:
                if not ALLOW_MISSING_QUALITY:
                    raise RuntimeError(
                        f"{photon_file_path} has no GTI extension."
                    )
        removed = int(len(data) - file_mask.sum())
        if removed:
            print(
                f"    quality cuts removed {removed} photons "
                f"(zmax<{ZMAX_DEGREES:.0f}, evclass bit {EVCLASS_BIT}, "
                f"GTI={'on' if APPLY_GTI else 'off'}).",
                flush = True,
            )


        RIGHT_ASCENSION_ARRAYS.append(data["RA"][file_mask].copy())
        DECLINATION_ARRAYS.append(data["DEC"][file_mask].copy())
        ENERGY_ARRAYS.append(data["ENERGY"][file_mask].copy())
        TIME_ARRAYS.append(data["TIME"][file_mask].copy())
        if SCRAMBLE_TIMES and NULL_TIME_MODEL:
            missing_ids = {"RUN_ID", "EVENT_ID"} - set(column_names)
            if missing_ids:
                raise RuntimeError(
                    f"{photon_file_path} lacks coherent-null identifiers: "
                    + ", ".join(sorted(missing_ids))
                )
            RUN_ID_ARRAYS.append(data["RUN_ID"][file_mask].copy())
            EVENT_ID_ARRAYS.append(data["EVENT_ID"][file_mask].copy())

# Concatenate for DBSCAN, with each element as a photon
RIGHT_ASCENSION = np.concatenate(RIGHT_ASCENSION_ARRAYS)
DECLINATION = np.concatenate(DECLINATION_ARRAYS)
ENERGY = np.concatenate(ENERGY_ARRAYS)
TIME = np.concatenate(TIME_ARRAYS)
if SCRAMBLE_TIMES and NULL_TIME_MODEL:
    RUN_ID = np.concatenate(RUN_ID_ARRAYS)
    EVENT_ID = np.concatenate(EVENT_ID_ARRAYS)
PRE_FILTER_COUNT = len(RIGHT_ASCENSION)

# Purge NaN and Inf values; Ensure window start < time < window end.
VALID_MASK = (
    np.isfinite(RIGHT_ASCENSION)
    & np.isfinite(DECLINATION)
    & np.isfinite(ENERGY)
    & np.isfinite(TIME)
    & (TIME > WINDOW_START)
    & (TIME < WINDOW_END)
)

# Enforce the actual analysis cap. Downloaded files may include a larger
# margin for likelihood fitting, and injected PSF draws can cross the edge.
VALID_MASK &= angular_separation_degrees(
    RIGHT_ASCENSION,
    DECLINATION,
    ROI_RA,
    ROI_DEC,
) <= ANALYSIS_ROI_RADIUS_DEGREES

RIGHT_ASCENSION = RIGHT_ASCENSION[VALID_MASK]
DECLINATION = DECLINATION[VALID_MASK]
ENERGY = ENERGY[VALID_MASK]
TIME = TIME[VALID_MASK]
if SCRAMBLE_TIMES and NULL_TIME_MODEL:
    RUN_ID = RUN_ID[VALID_MASK]
    EVENT_ID = EVENT_ID[VALID_MASK]

INVALID_PHOTON_COUNT = PRE_FILTER_COUNT - len(RIGHT_ASCENSION)
print(
    f"Removed {INVALID_PHOTON_COUNT} photons. Loading {len(RIGHT_ASCENSION)} valid photons.",
    flush = True,
)

# optional time scrambling for null calibration
# (see comment at the SCRAMBLE_TIMES definition above). ---
if SCRAMBLE_TIMES:
    if NULL_TIME_MODEL:
        null_model_path = Path(NULL_TIME_MODEL)
        if not null_model_path.is_absolute():
            null_model_path = PROJECT_ROOT / null_model_path
        TIME = coherent_scrambled_times(
            RIGHT_ASCENSION, DECLINATION, RUN_ID, EVENT_ID,
            SCRAMBLE_SEED, null_model_path,
        )
        print(
            "FPS_SCRAMBLE_TIMES=1: coherent owner-cell empirical times "
            f"assigned (seed {SCRAMBLE_SEED}).",
            flush=True,
        )
    else:
        scramble_rng = np.random.default_rng(SCRAMBLE_SEED)
        TIME = scramble_rng.permutation(TIME)
        print(
            f"FPS_SCRAMBLE_TIMES=1: photon arrival times permuted "
            f"(seed {SCRAMBLE_SEED}) for false-track calibration.",
            flush = True,
        )



def remove_duplicates(candidates):
    if not candidates:
        return []

    UNIQUE_CANDIDATES = []
    candidates_by_bin = {}
    for candidate in candidates:
        candidates_by_bin.setdefault(candidate["bin_index"], []).append(
            candidate
        )

    for bin_index in sorted(candidates_by_bin):
        bin_candidates = sorted(
            candidates_by_bin[bin_index],
            key = lambda candidate: -candidate["sigma"],
        )
        consumed = [False] * len(bin_candidates)

        for candidate_index, seed_candidate in enumerate(bin_candidates):
            if consumed[candidate_index]:
                continue

            consumed[candidate_index] = True
            merged_candidates = [seed_candidate]

            for neighbor_index in range(candidate_index + 1, len(bin_candidates)):
                if consumed[neighbor_index]:
                    continue

                neighbor_candidate = bin_candidates[neighbor_index]
                merge_radius_degrees = max(
                    seed_candidate["epsilon_degrees"],
                    neighbor_candidate["epsilon_degrees"],
                )
                separation_degrees = angular_separation_degrees(
                    seed_candidate["ra"],
                    seed_candidate["dec"],
                    neighbor_candidate["ra"],
                    neighbor_candidate["dec"],
                )
                if separation_degrees > merge_radius_degrees:
                    continue

                consumed[neighbor_index] = True
                merged_candidates.append(neighbor_candidate)

            canonical_candidate = dict(seed_candidate)
            canonical_candidate["merged_bands"] = sorted(
                {candidate["band"] for candidate in merged_candidates}
            )
            canonical_candidate["merged_candidate_count"] = len(
                merged_candidates
            )
            UNIQUE_CANDIDATES.append(canonical_candidate)

    return UNIQUE_CANDIDATES


def build_link_predecessors(SORTED_CANDIDATES):
    candidates_by_bin = {}
    for candidate_index, candidate in enumerate(SORTED_CANDIDATES):
        candidates_by_bin.setdefault(candidate["bin_index"], []).append(
            candidate_index
        )

    predecessors = [[] for _ in range(len(SORTED_CANDIDATES))]
    total_edges = 0

    for candidate_index, candidate in enumerate(SORTED_CANDIDATES):
        for bin_gap in range(1, MAX_BIN_GAP + 1):
            previous_bin = candidate["bin_index"] - bin_gap
            for previous_index in candidates_by_bin.get(previous_bin, []):
                previous_candidate = SORTED_CANDIDATES[previous_index]
                elapsed_years = (
                    float(candidate["t_mid"])
                    - float(previous_candidate["t_mid"])
                ) / SECONDS_PER_YEAR
                if elapsed_years <= 0:
                    continue
                max_distance_for_gap = (
                    MAX_LINK_DEG_PER_YEAR * elapsed_years
                )
                distance = angular_separation_degrees(
                    previous_candidate["ra"],
                    previous_candidate["dec"],
                    candidate["ra"],
                    candidate["dec"],
                )
                if distance > max_distance_for_gap:
                    continue

                transition_penalty = (
                    distance / elapsed_years * BIN_YEARS
                    + LINK_GAP_PENALTY * (bin_gap - 1)
                )
                predecessors[candidate_index].append(
                    (previous_index, transition_penalty)
                )
                total_edges += 1

    return predecessors, total_edges


def link_candidates(SORTED_CANDIDATES):
    predecessors, total_edges = build_link_predecessors(SORTED_CANDIDATES)
    print(
        f"Built {total_edges} allowed inter-bin links for dynamic programming.",
        flush = True,
    )

    # Keep a bounded beam for each skip count. A single best state can hide a
    # legitimate overlapping track that reaches the same candidate, while
    # unrestricted path enumeration grows exponentially. The one-point base
    # state is always retained so unrelated earlier noise cannot prevent a
    # genuine track from starting later at this candidate.
    states = []
    for candidate_index, candidate in enumerate(SORTED_CANDIDATES):
        extension_states = {}
        for previous_index, transition_penalty in predecessors[candidate_index]:
            skipped_increment = max(
                0,
                int(candidate["bin_index"])
                - int(SORTED_CANDIDATES[previous_index]["bin_index"])
                - 1,
            )
            for previous_skips, previous_states in states[previous_index].items():
                for previous_rank, previous_state in enumerate(previous_states):
                    total_skips = previous_skips + skipped_increment
                    if total_skips > MAX_TOTAL_SKIPPED_BINS:
                        continue
                    extension_states.setdefault(total_skips, []).append({
                        "length": previous_state["length"] + 1,
                        "score": float(
                            previous_state["score"]
                            + float(candidate["sigma"])
                            - transition_penalty
                        ),
                        "previous": (
                            previous_index, previous_skips, previous_rank,
                        ),
                    })

        candidate_states = {}
        for skips, alternatives in extension_states.items():
            alternatives.sort(
                key = lambda state: (-state["length"], -state["score"])
            )
            candidate_states[skips] = alternatives[
                :MAX_LINK_STATES_PER_SKIP
            ]
        candidate_states.setdefault(0, []).append({
            "length": 1,
            "score": float(candidate["sigma"]),
            "previous": None,
        })
        states.append(candidate_states)

    ranked_TRAJECTORIES = []
    for endpoint_index, endpoint_states in enumerate(states):
        for endpoint_skips, alternatives in endpoint_states.items():
            for endpoint_rank, endpoint_state in enumerate(alternatives):
                if endpoint_state["length"] < MIN_TRACK_LENGTH:
                    continue
                track_indices = []
                current = (endpoint_index, endpoint_skips, endpoint_rank)
                while current is not None:
                    current_index, current_skips, current_rank = current
                    track_indices.append(current_index)
                    current = states[current_index][current_skips][
                        current_rank
                    ]["previous"]
                track_indices.reverse()
                ranked_TRAJECTORIES.append((
                    endpoint_state["length"],
                    endpoint_state["score"],
                    [SORTED_CANDIDATES[index] for index in track_indices],
                ))

    ranked_TRAJECTORIES.sort(
        key = lambda item: (-item[0], -item[1])
    )
    print(
        (
            f"Recovered {len(ranked_TRAJECTORIES)} candidate TRAJECTORIES "
            f"with >= {MIN_TRACK_LENGTH} linked detections and "
            f"<= {MAX_TOTAL_SKIPPED_BINS} skipped bins "
            f"(beam {MAX_LINK_STATES_PER_SKIP}/skip state)."
        ),
        flush = True,
    )
    return ranked_TRAJECTORIES


def default_motion_metrics():
    return {
        "tube_anisotropy": 0.0,
        "tube_time_correlation": 0.0,
        "velocity_consistency": 0.0,
        "velocity_fractional_scatter": None,
        "velocity_min_degrees_per_year": None,
        "velocity_max_degrees_per_year": None,
        "parallel_velocity_degrees_per_year": 0.0,
        "parallel_step_velocities_degrees_per_year": [],
    }


def trajectory_motion_metrics(track):
    if len(track) < 3:
        return default_motion_metrics()

    right_ascension = np.asarray(
        [point["ra"] for point in track], dtype = float
    )
    declination = np.asarray(
        [point["dec"] for point in track], dtype = float
    )

    # times were raw bin indices, so every
    # velocity came out in degrees per BIN and the JSON mixed deg/bin,
    # deg/month (30-day months) and deg/yr. Times are now converted to years
    # so every velocity in this module is in degrees per year. ---
    times = (
        np.asarray([point["t_mid"] for point in track], dtype = float)
        - WINDOW_START
    ) / SECONDS_PER_YEAR


    right_ascension_rad = np.deg2rad(right_ascension)
    declination_rad = np.deg2rad(declination)
    unit_vectors = np.column_stack((
        np.cos(declination_rad) * np.cos(right_ascension_rad),
        np.cos(declination_rad) * np.sin(right_ascension_rad),
        np.sin(declination_rad),
    ))

    centroid = unit_vectors.mean(axis = 0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return default_motion_metrics()
    centroid /= centroid_norm

    pole = np.array([0.0, 0.0, 1.0])
    east_axis = np.cross(pole, centroid)
    east_norm = np.linalg.norm(east_axis)
    if east_norm < 1.0e-9:
        east_axis = np.array([1.0, 0.0, 0.0])
    else:
        east_axis /= east_norm
    north_axis = np.cross(centroid, east_axis)

    along_centroid = unit_vectors @ centroid
    x_tangent = (unit_vectors @ east_axis) / along_centroid
    y_tangent = (unit_vectors @ north_axis) / along_centroid
    points = np.column_stack((x_tangent, y_tangent))
    points -= points.mean(axis = 0)

    covariance = np.cov(points, rowvar = False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    parallel_variance = float(eigenvalues[-1])
    if parallel_variance <= 0:
        return default_motion_metrics()
    perpendicular_variance = max(float(eigenvalues[0]), 0.0)
    anisotropy = 1.0 - perpendicular_variance / parallel_variance

    principal_axis = eigenvectors[:, -1]
    principal_projection = points @ principal_axis
    if np.std(principal_projection) == 0 or np.std(times) == 0:
        time_correlation = 0.0
    else:
        time_correlation = abs(
            float(np.corrcoef(times, principal_projection)[0, 1])
        )

    slope, intercept = np.polyfit(times, principal_projection, deg = 1)
    if slope < 0:
        principal_projection = -principal_projection
        slope = -slope

    bin_gaps = np.diff(times)
    projected_steps = np.diff(principal_projection)
    valid_gaps = bin_gaps > 0
    if not np.any(valid_gaps):
        velocity_consistency = 0.0
        velocity_fractional_scatter = None
        velocity_min_degrees_per_year = None
        velocity_max_degrees_per_year = None
        step_velocities_degrees_per_year = []
    else:
        step_velocities = projected_steps[valid_gaps] / bin_gaps[valid_gaps]
        step_velocities_degrees_per_year = [
            float(value)
            for value in np.rad2deg(step_velocities)
        ]
        positive_step_velocities = step_velocities[step_velocities > 0]
        if len(positive_step_velocities) != len(step_velocities):
            velocity_consistency = 0.0
            velocity_fractional_scatter = None
            velocity_min_degrees_per_year = float(
                np.rad2deg(step_velocities.min())
            )
            velocity_max_degrees_per_year = float(
                np.rad2deg(step_velocities.max())
            )
        else:
            mean_velocity = float(np.mean(step_velocities))
            velocity_fractional_scatter = float(
                np.std(step_velocities) / mean_velocity
            )
            min_velocity = float(np.min(step_velocities))
            max_velocity = float(np.max(step_velocities))
            velocity_consistency = float(min_velocity / max_velocity)
            velocity_min_degrees_per_year = float(np.rad2deg(min_velocity))
            velocity_max_degrees_per_year = float(np.rad2deg(max_velocity))

    return {
        "tube_anisotropy": float(anisotropy),
        "tube_time_correlation": float(time_correlation),
        "velocity_consistency": float(velocity_consistency),
        "velocity_fractional_scatter": velocity_fractional_scatter,
        "velocity_min_degrees_per_year": velocity_min_degrees_per_year,
        "velocity_max_degrees_per_year": velocity_max_degrees_per_year,
        "parallel_velocity_degrees_per_year": float(np.rad2deg(slope)),
        "parallel_step_velocities_degrees_per_year": (
            step_velocities_degrees_per_year
        ),
    }


def trajectory_straightness(track):
    metrics = trajectory_motion_metrics(track)
    return metrics["tube_anisotropy"], metrics["tube_time_correlation"]


def filter_tube_like_trajectories(ranked_trajectories):
    # the straight-tube filters can be disabled
    # (FPS_DISABLE_MOTION_FILTER=1) for the slow-mover / parallax mode, where
    # the expected path is an annual wobble, not a straight line. ---
    if DISABLE_MOTION_FILTER:
        print(
            "FPS_DISABLE_MOTION_FILTER=1: skipping tube/velocity filters "
            f"({len(ranked_trajectories)} trajectories pass through).",
            flush = True,
        )
        candidates = ranked_trajectories

    else:
        candidates = []
        for length, score, trajectory in ranked_trajectories:
            metrics = trajectory_motion_metrics(trajectory)
            if (
                metrics["tube_anisotropy"] >= MIN_TUBE_ANISOTROPY
                and metrics["tube_time_correlation"] >= MIN_TUBE_TIME_CORRELATION
                and metrics["velocity_consistency"] >= MIN_VELOCITY_CONSISTENCY
            ):
                candidates.append((length, score, trajectory))
        print(
            f"Motion filter: {len(candidates)}/{len(ranked_trajectories)} "
            f"trajectories survive (anisotropy >= {MIN_TUBE_ANISOTROPY:.2f}, "
            f"time correlation >= {MIN_TUBE_TIME_CORRELATION:.2f}, "
            f"velocity consistency >= {MIN_VELOCITY_CONSISTENCY:.2f}).",
            flush = True,
        )

    # Remove exact duplicates and clean nested prefixes only after filtering.
    # If a bad extension fails above, its valid shorter prefix remains.
    candidates.sort(key = lambda item: (-item[0], -item[1]))
    kept = []
    kept_keys = []
    kept_sets = []
    for item in candidates:
        key = tuple(
            (
                int(point["bin_index"]),
                round(float(point["ra"]), 10),
                round(float(point["dec"]), 10),
                point.get("band"),
            )
            for point in item[2]
        )
        if key in kept_keys:
            continue
        if any(
            len(key) < len(existing) and existing[:len(key)] == key
            for existing in kept_keys
        ):
            continue
        key_set = set(key)
        if any(
            len(key_set & existing_set)
            / min(len(key_set), len(existing_set)) >= 0.8
            for existing_set in kept_sets
        ):
            continue
        kept.append(item)
        kept_keys.append(key)
        kept_sets.append(key_set)
    print(
        f"Post-filter trajectory deduplication: {len(kept)}/{len(candidates)} kept.",
        flush = True,
    )
    return kept


def temporal_bins(
    RIGHT_ASCENSION_array,
    DECLINATION_array,
    energy_array,
    time_array,
    bin_size_days = BIN_TIME,
):

    bin_edges = analysis_bin_edges(bin_size_days)
    n_bins = len(bin_edges) - 1

    bin_indices = np.digitize(time_array, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    sort_order = np.argsort(bin_indices, kind = "stable")
    sorted_bin_indices = bin_indices[sort_order]
    sorted_RIGHT_ASCENSION = RIGHT_ASCENSION_array[sort_order]
    sorted_DECLINATION = DECLINATION_array[sort_order]
    sorted_energy = energy_array[sort_order]
    sorted_time = time_array[sort_order]

    unique_bin_indices, first_positions, counts = np.unique(
        sorted_bin_indices,
        return_index = True,
        return_counts = True,
    )

    slices = []
    for unique_position in range(len(unique_bin_indices)):
        bin_index = int(unique_bin_indices[unique_position])
        start = int(first_positions[unique_position])
        end = start + int(counts[unique_position])
        slices.append(
            {
                "bin_index": bin_index,
                "t_start": float(bin_edges[bin_index]),
                "t_stop": float(bin_edges[bin_index + 1]),
                "t_mid": float(
                    0.5 * (bin_edges[bin_index] + bin_edges[bin_index + 1])
                ),
                "ra": sorted_RIGHT_ASCENSION[start:end],
                "dec": sorted_DECLINATION[start:end],
                "energy": sorted_energy[start:end],
                "time": sorted_time[start:end],
            }
        )

    return slices


def build_OUTPUT_RECORDS(trajectory_items, BAND_LABELS):
    OUTPUT_RECORDS = []
    for candidate_number, _, _, track in trajectory_items:
        track_start = track[0]
        track_end = track[-1]
        ra_start = float(track_start["ra"])
        dec_start = float(track_start["dec"])
        ra_end = float(track_end["ra"])
        dec_end = float(track_end["dec"])
        t_mid_start = float(track_start["t_mid"])
        t_mid_end = float(track_end["t_mid"])

        # "total_photons" (detection-band
        # aperture counts) and "total_photons_per_band" (ALL photons of every
        # band inside the aperture) measured different things under
        # near-identical names (41 vs 241 for the ROI-19 candidate). Both are
        # kept but renamed to say exactly what they count, and the union of
        # detected bands (merged_bands, previously computed but dropped) is
        # now propagated as the write-up already claimed. ---
        detection_band_photons = int(
            sum(point["n_photons"] for point in track)
        )
        aperture_photons_per_band = {
            band_label: 0 for band_label in BAND_LABELS
        }
        for point in track:
            for band_label in BAND_LABELS:
                aperture_photons_per_band[band_label] += point[
                    "photons_per_band"
                ].get(band_label, 0)
        detected_bands = sorted({
            band
            for point in track
            for band in point.get("merged_bands", [point.get("band")])
            if band is not None
        })


        net_separation_degrees = float(
            angular_separation_degrees(
                ra_start, dec_start, ra_end, dec_end,
            )
        )

        # average velocity was reported in
        # degrees per 30-day month (undocumented; Table 1 quoted deg/yr).
        # Now computed directly in degrees per 365-day year. ---
        duration_years = (t_mid_end - t_mid_start) / SECONDS_PER_YEAR
        average_velocity_deg_per_year = (
            net_separation_degrees / duration_years
            if duration_years > 0 else 0.0
        )


        motion_metrics = trajectory_motion_metrics(track)

        OUTPUT_RECORDS.append({
            "candidate_number": int(candidate_number),
            "track_length": len(track),
            "ra_start": ra_start,
            "ra_end": ra_end,
            "dec_start": dec_start,
            "dec_end": dec_end,
            "detection_band_photons": detection_band_photons,
            "aperture_photons_per_band": aperture_photons_per_band,
            "detected_bands": detected_bands,
            "average_velocity_deg_per_year": average_velocity_deg_per_year,
            "bin_velocities_deg_per_year": motion_metrics[
                "parallel_step_velocities_degrees_per_year"
            ],
            "bin_indices": [int(point["bin_index"]) for point in track],
            # persist the full per-bin track
            # points. check_v5 reads candidate["points"] (per-bin ra, dec,
            # t_start, n_photons); the slim records never contained them, so
            # the validation script could not consume the search output. ---
            "points": [
                {
                    "bin_index": int(point["bin_index"]),
                    "ra": float(point["ra"]),
                    "dec": float(point["dec"]),
                    "t_start": float(point["t_start"]),
                    "t_stop": float(point["t_stop"]),
                    "n_photons": int(point["n_photons"]),
                    "sigma": float(point["sigma"]),
                    "expected_bg": float(point["expected_bg"]),
                    "band": point["band"],
                    "epsilon_degrees": float(point["epsilon_degrees"]),
                    "annulus_area_deg2": float(point["annulus_area_deg2"]),
                    "annulus_nominal_area_deg2": float(
                        point["annulus_nominal_area_deg2"]
                    ),
                    "annulus_region_retained_fraction": float(
                        point["annulus_region_retained_fraction"]
                    ),
                    "annulus_masked_area_fraction": float(
                        point["annulus_masked_area_fraction"]
                    ),
                    "annulus_unmasked_fraction": float(
                        point["annulus_unmasked_fraction"]
                    ),
                    "annulus_unmasked_photons": int(
                        point["annulus_unmasked_photons"]
                    ),
                    "annulus_catalog_masks": int(
                        point["annulus_catalog_masks"]
                    ),
                    "annulus_mc_samples": int(
                        point["annulus_mc_samples"]
                    ),
                    "annulus_mc_to_poisson_ratio": float(
                        point["annulus_mc_to_poisson_ratio"]
                    ),
                    "catalog_mask_radius_degrees": float(
                        point["catalog_mask_radius_degrees"]
                    ),
                    "background_stable": bool(point["background_stable"]),
                }
                for point in track
            ],

        })

    return OUTPUT_RECORDS


def write_slim_json(records, path):
    # Match stat_bins.py results.json formatting: indented per field,
    # but list/dict values stay inline on one line.
    if not records:
        Path(path).write_text("[]\n")
        return
    def format_record(record):
        parts = ["  {"]
        items = list(record.items())
        for index, (key, value) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            parts.append(
                f'    "{key}": {json.dumps(value)}{suffix}'
            )
        parts.append("  }")
        return "\n".join(parts)
    body = ",\n".join(format_record(record) for record in records)
    Path(path).write_text("[\n" + body + "\n]\n")




BAND_EDGES = np.logspace(3, 5, num = 7)
BAND_LABELS = [
    f"{BAND_EDGES[index] / 1000:.1f}-{BAND_EDGES[index + 1] / 1000:.1f} GeV"
    for index in range(len(BAND_EDGES) - 1)
]
ALL_CANDIDATES = []

# Slice global arrays into temporal bins.
TIME_SLICES = temporal_bins(
    RIGHT_ASCENSION,
    DECLINATION,
    ENERGY,
    TIME,
    bin_size_days = BIN_TIME,
)
print(
    f"\n {len(TIME_SLICES)} non-empty {BIN_TIME:.0f}-day time bins to process.\n",
    flush = True,
)

for time_slice in TIME_SLICES:
    print(
        (
            f"Processing Time Bin #{time_slice['bin_index']} "
            f"(MET {time_slice['t_start']:.0f}-{time_slice['t_stop']:.0f}, "
            f"{len(time_slice['time'])} photons)"
        ),
        flush = True,
    )

    slice_RIGHT_ASCENSION = time_slice["ra"]
    slice_DECLINATION = time_slice["dec"]
    slice_energy = time_slice["energy"]

    # iterate over SEED_BANDS (2.2-10 GeV
    # merged) instead of the six narrow reporting bands, with a per-band
    # density-scaled min_samples. See the SEED_BANDS comment block above. ---
    for energy_minimum, energy_maximum, eps_reference_energy in SEED_BANDS:
        mask = (slice_energy >= energy_minimum) & (
            slice_energy < energy_maximum
        )
        photon_count = mask.sum()
        label = f"{energy_minimum / 1000:.1f}-{energy_maximum / 1000:.1f} GeV"

        if photon_count < MIN_PHOTONS_PER_BAND:
            print(f"  [{label}] {photon_count} photons - skipping.", flush = True)
            continue

        band_RIGHT_ASCENSION = slice_RIGHT_ASCENSION[mask]
        band_DECLINATION = slice_DECLINATION[mask]
        band_vectors = unit_vectors(
            band_RIGHT_ASCENSION, band_DECLINATION,
        )

        # Epsilon from the band's PSF reference energy (for the merged band:
        # the softest third, so soft photons are still captured).
        epsilon_degrees = EPS_SCALE * psf_radius_degrees(eps_reference_energy)
        epsilon_radians = np.deg2rad(epsilon_degrees)
        # Translate to spherical coordinates in rad
        coordinates_radians = np.deg2rad(
            np.column_stack([band_DECLINATION, band_RIGHT_ASCENSION])
        )

        # the predeclared position diagnostic
        # found a material (>2x) local-density dependence.  Estimate the
        # background around each prospective core from a self-shadow-free
        # >=2*eps to 5*eps annulus, retain the global floor, apply the local
        # core test, and only then form connected components.  Density-query
        # zones conservatively expand the inner edge by their maximum member
        # offset, so the 2*eps self-shadow guard is exact for every photon.
        # The region-average path remains an explicit validation switch.
        # ---
        minimum_samples_band = band_min_samples(
            photon_count, epsilon_degrees,
        )
        if ADAPTIVE_LOCAL_SEEDING:
            labels, local_diagnostics = adaptive_density_labels(
                coordinates_radians,
                epsilon_radians,
                minimum_samples_floor=MIN_SAMPLES_FLOOR,
                inner_radius_scale=2.0,
                outer_radius_scale=5.0,
                density_zone_scale=0.5,
            )
            local_thresholds = local_diagnostics["local_min_samples"]
            threshold_minimum = int(np.min(local_thresholds))
            threshold_median = int(np.median(local_thresholds))
            threshold_maximum = int(np.max(local_thresholds))
            threshold_record = {
                "bin_index": int(time_slice["bin_index"]),
                "band": label,
                "photon_count": int(photon_count),
                "epsilon_degrees": float(epsilon_degrees),
                # Retain this scalar for existing summary consumers.
                "min_samples": threshold_median,
                "min_samples_mode": (
                    "adaptive_local_ge2eps_5eps_zoned_v2"
                ),
                "min_samples_minimum": threshold_minimum,
                "min_samples_median": threshold_median,
                "min_samples_maximum": threshold_maximum,
                "roi_average_comparison": int(minimum_samples_band),
                "n_core_points": int(
                    local_diagnostics["n_core_points"]
                ),
                "n_density_zones": int(
                    local_diagnostics["n_density_zones"]
                ),
                "density_zone_scale_epsilon": float(
                    local_diagnostics["density_zone_scale"]
                ),
                "maximum_zone_offset_epsilon": float(
                    local_diagnostics["maximum_zone_offset_epsilon"]
                ),
                "effective_inner_radius_scale_minimum": float(
                    local_diagnostics[
                        "effective_inner_radius_scale_minimum"
                    ]
                ),
                "effective_inner_radius_scale_maximum": float(
                    local_diagnostics[
                        "effective_inner_radius_scale_maximum"
                    ]
                ),
            }
            print(
                (
                    f"  [{label}] {photon_count} photons, "
                    f"eps={epsilon_degrees:.3f} deg, "
                    "adaptive min_samples "
                    f"{threshold_minimum}/{threshold_median}/"
                    f"{threshold_maximum} (min/median/max; "
                    f"ROI-average comparison {minimum_samples_band})"
                ),
                flush=True,
            )
        else:
            dbscan_model = DBSCAN(
                eps=epsilon_radians,
                min_samples=minimum_samples_band,
                metric="haversine",
            )
            labels = dbscan_model.fit_predict(coordinates_radians)
            threshold_record = {
                "bin_index": int(time_slice["bin_index"]),
                "band": label,
                "photon_count": int(photon_count),
                "epsilon_degrees": float(epsilon_degrees),
                "min_samples": int(minimum_samples_band),
                "min_samples_mode": "roi_average_diagnostic_v1",
            }
            print(
                (
                    f"  [{label}] {photon_count} photons, "
                    f"eps={epsilon_degrees:.3f} deg, "
                    f"min_samples = {minimum_samples_band}"
                ),
                flush=True,
            )
        AREA_DIAGNOSTICS["min_samples"].append(threshold_record)



        audit_band_record = None
        if INJECTION_STAGE_AUDIT is not None:
            truth_point = INJECTION_STAGE_AUDIT_TRUTH_BY_BIN.get(
                int(time_slice["bin_index"])
            )
            truth_distances = angular_separation_degrees(
                band_RIGHT_ASCENSION,
                band_DECLINATION,
                float(truth_point["ra"]),
                float(truth_point["dec"]),
            )
            truth_aperture = truth_distances <= epsilon_degrees
            if ADAPTIVE_LOCAL_SEEDING:
                epsilon_counts = local_diagnostics[
                    "epsilon_neighbor_counts"
                ]
                core_mask = epsilon_counts >= local_thresholds
                aperture_thresholds = local_thresholds[truth_aperture]
                aperture_counts = epsilon_counts[truth_aperture]
                aperture_core = core_mask[truth_aperture]
            else:
                aperture_thresholds = np.full(
                    int(np.sum(truth_aperture)), minimum_samples_band,
                    dtype=int,
                )
                aperture_counts = np.zeros(
                    int(np.sum(truth_aperture)), dtype=int,
                )
                aperture_core = labels[truth_aperture] >= 0
            audit_band_record = {
                "bin_index": int(time_slice["bin_index"]),
                "band": label,
                "epsilon_degrees": float(epsilon_degrees),
                "band_photon_count": int(photon_count),
                "photons_within_truth_epsilon": int(np.sum(truth_aperture)),
                "minimum_threshold_within_truth_epsilon": (
                    int(np.min(aperture_thresholds))
                    if len(aperture_thresholds) else None
                ),
                "maximum_neighbor_count_within_truth_epsilon": (
                    int(np.max(aperture_counts))
                    if len(aperture_counts) else None
                ),
                "core_points_within_truth_epsilon": int(
                    np.sum(aperture_core)
                ),
                "truth_matched_clusters": [],
            }
            INJECTION_STAGE_AUDIT["band_records"].append(
                audit_band_record
            )

        # Purge noise w/ -1
        cluster_ids = sorted(set(labels) - {-1})
        if audit_band_record is not None:
            audit_band_record["total_seed_clusters"] = len(cluster_ids)
        if not cluster_ids:
            print("    to 0 clusters.", flush = True)
            continue

        tree = BallTree(coordinates_radians, metric = "haversine")

        surviving = 0
        for cluster_id in cluster_ids:
            cluster_mask = labels == cluster_id
            cluster_RIGHT_ASCENSION = band_RIGHT_ASCENSION[cluster_mask]
            cluster_DECLINATION = band_DECLINATION[cluster_mask]

            # Compute the candidate centroid
            (
                centroid_RIGHT_ASCENSION,
                centroid_DECLINATION,
            ) = spherical_mean_coordinates(
                cluster_RIGHT_ASCENSION,
                cluster_DECLINATION,
            )
            centroid_radians = np.deg2rad(
                [[centroid_DECLINATION, centroid_RIGHT_ASCENSION]]
            )

            audit_cluster = None
            if audit_band_record is not None:
                truth_separation = injection_stage_audit_separation(
                    time_slice["bin_index"],
                    centroid_RIGHT_ASCENSION,
                    centroid_DECLINATION,
                )
                if (
                    truth_separation
                    <= INJECTION_STAGE_AUDIT_MATCH_RADIUS_DEGREES
                ):
                    audit_cluster = {
                        "cluster_id": int(cluster_id),
                        "centroid_truth_separation_degrees": float(
                            truth_separation
                        ),
                        "cluster_photon_count": int(np.sum(cluster_mask)),
                        "seeded": True,
                        "survived_catalogue_veto": False,
                        "stable_background": False,
                        "survived_significance": False,
                    }
                    audit_band_record["truth_matched_clusters"].append(
                        audit_cluster
                    )

            # Drop clusters that land on top of a known 4FGL stationary
            # source. Those are not moving candidates by definition.
            (
                is_near_catalog,
                source_separations,
                on_exclusion_radius,
            ) = near_catalog_source(
                centroid_RIGHT_ASCENSION,
                centroid_DECLINATION,
                epsilon_degrees,
            )
            if (
                CATALOG_ANNULUS_MASK
                and len(source_separations)
                and np.any(
                    (source_separations >= CATALOG_EXCLUSION_RADIUS_DEGREES)
                    & (source_separations < epsilon_degrees)
                )
            ):
                AREA_DIAGNOSTICS[
                    "on_region_sources_between_fixed_mask_and_r68"
                ] += 1
            if is_near_catalog:
                if audit_cluster is not None:
                    audit_cluster["catalogue_vetoed"] = True
                    audit_cluster["catalogue_nearest_separation_degrees"] = (
                        float(np.min(source_separations))
                        if len(source_separations) else None
                    )
                continue
            if audit_cluster is not None:
                audit_cluster["catalogue_vetoed"] = False
                audit_cluster["survived_catalogue_veto"] = True

            inner_degrees = INNER_RADIUS_SCALE * epsilon_degrees
            outer_degrees = OUTER_RADIUS_SCALE * epsilon_degrees

            # one spherical Monte-Carlo estimator is
            # used for the on region and the annulus.  The analysis predicate
            # is the actual BUFFER data boundary, not the ownership cell. ---
            signal_estimate = estimate_spherical_area(
                centroid_RIGHT_ASCENSION,
                centroid_DECLINATION,
                0.0,
                epsilon_degrees,
                ANALYSIS_REGION,
                n_samples=MC_AREA_SAMPLES,
            )
            signal_area = signal_estimate.area_deg2
            if signal_area <= 0.0:
                if audit_cluster is not None:
                    audit_cluster["background_rejection"] = (
                        "nonpositive_signal_area"
                    )
                continue


            # Use a 5x-2x donut to estimate the 1x local background.
            aperture_photon_count = tree.query_radius(
                centroid_radians,
                r = epsilon_radians,
                count_only = True,
            )[0]

            mask_radius_degrees = max(
                CATALOG_MASK_R68_SCALE * epsilon_degrees,
                CATALOG_MASK_MIN_RADIUS_DEGREES,
            )
            if CATALOG_ANNULUS_MASK and len(CATALOG_SOURCES):
                flux_eligible = (
                    np.isfinite(CATALOG_FLUX)
                    & (CATALOG_FLUX >= CATALOG_MASK_MIN_FLUX)
                    if CATALOG_MASK_MIN_FLUX > 0.0
                    else np.ones(len(CATALOG_SOURCES), dtype=bool)
                )
                intersects_annulus = (
                    source_separations
                    < outer_degrees + mask_radius_degrees
                ) & (
                    source_separations + mask_radius_degrees > inner_degrees
                )
                mask_indices = np.flatnonzero(
                    flux_eligible & intersects_annulus
                )
                mask_vectors = CATALOG_VECTORS[mask_indices]
                mask_radii = np.full(
                    len(mask_indices), mask_radius_degrees, dtype=float,
                )
            else:
                mask_indices = np.empty(0, dtype=int)
                mask_vectors = np.empty((0, 3), dtype=float)
                mask_radii = np.empty(0, dtype=float)

            candidate_vector = unit_vectors(
                [centroid_RIGHT_ASCENSION],
                [centroid_DECLINATION],
            )[0]
            photon_count_donut = unmasked_photon_count(
                band_vectors,
                candidate_vector,
                inner_degrees,
                outer_degrees,
                mask_center_vectors=mask_vectors,
                mask_radii_degrees=mask_radii,
            )
            donut_estimate, mc_to_poisson_ratio = adaptive_annulus_area(
                centroid_RIGHT_ASCENSION,
                centroid_DECLINATION,
                inner_degrees,
                outer_degrees,
                mask_vectors,
                mask_radii,
                photon_count_donut,
            )
            donut_area = donut_estimate.area_deg2

            AREA_DIAGNOSTICS["annulus_measurements"] += 1
            AREA_DIAGNOSTICS["annulus_mc_samples"].append(
                int(donut_estimate.n_samples)
            )
            AREA_DIAGNOSTICS["annulus_mc_to_poisson_ratios"].append(
                float(mc_to_poisson_ratio)
            )
            boundary_clip_fraction = (
                1.0 - donut_estimate.region_retained_fraction
            )
            if boundary_clip_fraction > 1.0 / MC_AREA_SAMPLES:
                AREA_DIAGNOSTICS["boundary_clipped_measurements"] += 1
            AREA_DIAGNOSTICS["boundary_clip_fractions"].append(
                float(boundary_clip_fraction)
            )
            AREA_DIAGNOSTICS["masked_area_fractions"].append(
                float(donut_estimate.masked_fraction_of_region)
            )

            unmasked_fraction = (
                donut_estimate.retained_fraction
                / donut_estimate.region_retained_fraction
                if donut_estimate.region_retained_fraction > 0.0 else 0.0
            )
            background_stable = (
                donut_area >= MIN_DONUT_AREA
                and unmasked_fraction >= MIN_UNMASKED_ANNULUS_FRACTION
                and photon_count_donut >= MIN_UNMASKED_OFF_COUNTS
            )
            if not background_stable:
                AREA_DIAGNOSTICS["unstable_overmasked_measurements"] += 1
                if audit_cluster is not None:
                    audit_cluster["background_rejection"] = (
                        "unstable_or_overmasked_annulus"
                    )
                    audit_cluster["annulus_unmasked_fraction"] = float(
                        unmasked_fraction
                    )
                    audit_cluster["annulus_unmasked_photons"] = int(
                        photon_count_donut
                    )
                continue
            if audit_cluster is not None:
                audit_cluster["stable_background"] = True

            # Use Li and Ma (1983), Equation 17,
            alpha = signal_area / donut_area
            n_on = float(aperture_photon_count)
            n_off = float(photon_count_donut)
            expected_background = alpha * n_off
            sigma = lima_sigma(n_on, n_off, alpha)

            if audit_cluster is not None:
                audit_cluster["sigma"] = float(sigma)
                audit_cluster["n_on"] = float(n_on)
                audit_cluster["n_off"] = float(n_off)
                audit_cluster["alpha"] = float(alpha)

            if sigma < MIN_SIGNIFICANCE:
                continue
            if audit_cluster is not None:
                audit_cluster["survived_significance"] = True

            # Count photons per energy band within the detection aperture.
            distances_to_centroid = angular_separation_degrees(
                centroid_RIGHT_ASCENSION,
                centroid_DECLINATION,
                slice_RIGHT_ASCENSION,
                slice_DECLINATION,
            )

            within_aperture = distances_to_centroid <= epsilon_degrees
            photons_per_band = {}
            for band_label_index, band_label in enumerate(BAND_LABELS):
                energy_low = BAND_EDGES[band_label_index]
                energy_high = BAND_EDGES[band_label_index + 1]
                in_band = (slice_energy >= energy_low) & (
                    slice_energy < energy_high
                )
                photons_per_band[band_label] = int(
                    (within_aperture & in_band).sum()
                )

            surviving += 1
            ALL_CANDIDATES.append(
                {
                    "ra": centroid_RIGHT_ASCENSION,
                    "dec": centroid_DECLINATION,
                    "n_photons": int(aperture_photon_count),
                    "expected_bg": expected_background,
                    "sigma": sigma,
                    "band": label,
                    "e_mid": eps_reference_energy,  # seed-band eps reference
                    "epsilon_degrees": epsilon_degrees,
                    "annulus_area_deg2": float(donut_area),
                    "annulus_nominal_area_deg2": float(
                        donut_estimate.nominal_area_deg2
                    ),
                    "annulus_region_retained_fraction": float(
                        donut_estimate.region_retained_fraction
                    ),
                    "annulus_masked_area_fraction": float(
                        donut_estimate.masked_fraction_of_region
                    ),
                    "annulus_unmasked_fraction": float(unmasked_fraction),
                    "annulus_unmasked_photons": int(photon_count_donut),
                    "annulus_catalog_masks": int(len(mask_indices)),
                    "annulus_mc_samples": int(donut_estimate.n_samples),
                    "annulus_mc_to_poisson_ratio": float(
                        mc_to_poisson_ratio
                    ),
                    "catalog_mask_radius_degrees": float(
                        mask_radius_degrees
                    ),
                    "on_catalog_exclusion_radius_degrees": float(
                        on_exclusion_radius
                    ),
                    "background_stable": True,
                    "photons_per_band": photons_per_band,
                    "bin_index": time_slice["bin_index"],
                    "t_start": time_slice["t_start"],
                    "t_stop": time_slice["t_stop"],
                    "t_mid": time_slice["t_mid"],
                }
            )

        # the log message claimed a
        # "3 * sigma" cut while the actual threshold is MIN_SIGNIFICANCE
        # (2.5); the message now reports the real value. ---
        print(
            (
                f"    -> {len(cluster_ids)} clusters, {surviving} survived "
                f"{MIN_SIGNIFICANCE:g} sigma local-background cut."
            ),
            flush = True,
        )


print(
    f"\n {len(ALL_CANDIDATES)} total candidates across all time bins. "
    f"Linking TRAJECTORIES...",
    flush = True,
)

UNIQUE_CANDIDATES = remove_duplicates(ALL_CANDIDATES)
if len(UNIQUE_CANDIDATES) != len(ALL_CANDIDATES):
    print(
        (
            f"Collapsed same-bin overlaps from {len(ALL_CANDIDATES)} to "
            f"{len(UNIQUE_CANDIDATES)} canonical candidates."
        ),
        flush = True,
    )
ALL_CANDIDATES = UNIQUE_CANDIDATES

if INJECTION_STAGE_AUDIT is not None:
    truth_bin_candidates = [
        candidate for candidate in ALL_CANDIDATES
        if injection_stage_audit_separation(
            candidate["bin_index"], candidate["ra"], candidate["dec"],
        ) <= INJECTION_STAGE_AUDIT_MATCH_RADIUS_DEGREES
    ]
    INJECTION_STAGE_AUDIT["linking"].update({
        "post_significance_truth_matched_bin_candidates": len(
            truth_bin_candidates
        ),
        "post_significance_truth_matched_unique_bins": len({
            int(candidate["bin_index"])
            for candidate in truth_bin_candidates
        }),
    })

# optional per-bin candidate dump
# (FPS_SAVE_BIN_CANDIDATES=1) so injection tests can check whether the
# injected source was detected per bin even when no track forms. ---
if SAVE_BIN_CANDIDATES:
    RESULTS_DIR.mkdir(parents = True, exist_ok = True)
    Path(RESULTS_DIR / "bin_candidates.json").write_text(json.dumps(
        [
            {
                "bin_index": c["bin_index"],
                "ra": float(c["ra"]),
                "dec": float(c["dec"]),
                "sigma": float(c["sigma"]),
                "band": c["band"],
                "n_photons": int(c["n_photons"]),
                "expected_bg": float(c["expected_bg"]),
                "merged_bands": c.get("merged_bands", [c["band"]]),
                "epsilon_degrees": float(c["epsilon_degrees"]),
                "t_start": float(c["t_start"]),
                "t_stop": float(c["t_stop"]),
            }
            for c in ALL_CANDIDATES
        ],
        indent = 2,
    ))
    print(
        f"Wrote {len(ALL_CANDIDATES)} per-bin candidates to "
        f"{RESULTS_DIR / 'bin_candidates.json'}",
        flush = True,
    )



# write the summary.json that the batch driver
# expects (it previously crashed reading a file nothing produced). ---
def write_summary(n_candidates, n_pre_filter_tracks):
    code_hash = hashlib.sha256()
    for source_path in (
        Path(__file__).resolve(), PROJECT_ROOT / "moving_utils_New.py",
        PROJECT_ROOT / "moving" / "coherent_null.py",
        PROJECT_ROOT / "moving" / "analysis_region_v5.py",
        PROJECT_ROOT / "moving" / "adaptive_seeding_v5.py",
    ):
        code_hash.update(source_path.read_bytes())
    masked_fractions = np.asarray(
        AREA_DIAGNOSTICS["masked_area_fractions"], dtype=float,
    )
    boundary_fractions = np.asarray(
        AREA_DIAGNOSTICS["boundary_clip_fractions"], dtype=float,
    )
    annulus_mc_samples = np.asarray(
        AREA_DIAGNOSTICS["annulus_mc_samples"], dtype=int,
    )
    annulus_mc_ratios = np.asarray(
        AREA_DIAGNOSTICS["annulus_mc_to_poisson_ratios"], dtype=float,
    )
    annulus_measurements = AREA_DIAGNOSTICS["annulus_measurements"]
    annulus_diagnostics = {
        "n_measurements": int(annulus_measurements),
        "n_boundary_clipped": int(
            AREA_DIAGNOSTICS["boundary_clipped_measurements"]
        ),
        "fraction_boundary_clipped": (
            AREA_DIAGNOSTICS["boundary_clipped_measurements"]
            / annulus_measurements if annulus_measurements else 0.0
        ),
        "median_boundary_clip_fraction": (
            float(np.median(boundary_fractions))
            if len(boundary_fractions) else None
        ),
        "masked_area_fraction_percentiles": (
            {
                str(percentile): float(np.percentile(
                    masked_fractions, percentile,
                ))
                for percentile in (0, 10, 25, 50, 75, 90, 100)
            }
            if len(masked_fractions) else {}
        ),
        "n_unstable_overmasked": int(
            AREA_DIAGNOSTICS["unstable_overmasked_measurements"]
        ),
        "n_on_region_sources_between_fixed_mask_and_r68": int(
            AREA_DIAGNOSTICS[
                "on_region_sources_between_fixed_mask_and_r68"
            ]
        ),
        "mc_samples_percentiles": (
            {
                str(percentile): float(np.percentile(
                    annulus_mc_samples, percentile,
                ))
                for percentile in (0, 50, 90, 100)
            } if len(annulus_mc_samples) else {}
        ),
        "mc_to_poisson_error_ratio_percentiles": (
            {
                str(percentile): float(np.percentile(
                    annulus_mc_ratios, percentile,
                ))
                for percentile in (0, 50, 90, 100)
            } if len(annulus_mc_ratios) else {}
        ),
    }
    summary = [{
        "roi_ra": ROI_RA,
        "roi_dec": ROI_DEC,
        "analysis_roi_radius_degrees": ANALYSIS_ROI_RADIUS_DEGREES,
        "analysis_region": ANALYSIS_REGION.metadata(),
        "bin_time_days": BIN_TIME,
        "n_track_candidates": int(n_candidates),
        "n_pre_filter_tracks": int(n_pre_filter_tracks),
        "scrambled_times": bool(SCRAMBLE_TIMES),
        "scramble_method": (
            "coherent_owner_cell_empirical_time_bootstrap_v1"
            if SCRAMBLE_TIMES and NULL_TIME_MODEL else
            "within_tile_time_permutation_v1" if SCRAMBLE_TIMES else None
        ),
        "max_link_deg_per_year": MAX_LINK_DEG_PER_YEAR,
        "max_total_skipped_bins": MAX_TOTAL_SKIPPED_BINS,
        "max_link_states_per_skip": MAX_LINK_STATES_PER_SKIP,
        "stage_a_code_sha256": code_hash.hexdigest(),
        "area_estimator": {
            "method": "uniform_solid_angle_low_discrepancy_monte_carlo",
            "base_samples_per_area": MC_AREA_SAMPLES,
            "maximum_samples_per_area": MAX_MC_AREA_SAMPLES,
            "maximum_mc_to_n_off_poisson_error_ratio": (
                MAX_MC_TO_POISSON_RATIO
            ),
            "annulus_diagnostics": annulus_diagnostics,
        },
        "min_samples_by_bin_and_band": AREA_DIAGNOSTICS["min_samples"],
        "disable_motion_filter": DISABLE_MOTION_FILTER,
        "stationary_validation": STATIONARY_VALIDATION,
        "operating_point": {
            "min_track_length": MIN_TRACK_LENGTH,
            "min_samples_floor": MIN_SAMPLES_FLOOR,
            "min_samples_rule": (
                "adaptive_local_ge2eps_5eps_poisson_2.5sigma_zoned_v2"
                if ADAPTIVE_LOCAL_SEEDING
                else "poisson_2.5sigma_continuous_roi_average_v1"
            ),
            "adaptive_local_seeding": ADAPTIVE_LOCAL_SEEDING,
            "adaptive_background_annulus_epsilon_scales": (
                {
                    "minimum_inner": 2.0,
                    "outer": 5.0,
                    "zone_inner_edge_guard": (
                        "maximum_member_offset_from_zone_representative"
                    ),
                }
                if ADAPTIVE_LOCAL_SEEDING else None
            ),
            "adaptive_density_zone_scale_epsilon": (
                0.5 if ADAPTIVE_LOCAL_SEEDING else None
            ),
            "eps_scale": EPS_SCALE,
            "min_significance": MIN_SIGNIFICANCE,
            "min_tube_anisotropy": MIN_TUBE_ANISOTROPY,
            "min_tube_time_correlation": MIN_TUBE_TIME_CORRELATION,
            "min_velocity_consistency": MIN_VELOCITY_CONSISTENCY,
            "zmax_degrees": ZMAX_DEGREES,
            "event_class_bit": EVCLASS_BIT,
            "apply_gti": APPLY_GTI,
            "allow_missing_quality": ALLOW_MISSING_QUALITY,
            "catalog_fits": CATALOG_FITS,
            "catalog_on_region_veto": CATALOG_ON_REGION_VETO,
            "catalog_annulus_mask": CATALOG_ANNULUS_MASK,
            "catalog_mask_radius_rule": (
                f"max({CATALOG_MASK_R68_SCALE:g}*r68,"
                f"{CATALOG_MASK_MIN_RADIUS_DEGREES:g} deg)"
            ),
            "catalog_mask_r68_scale": CATALOG_MASK_R68_SCALE,
            "catalog_mask_min_radius_degrees": (
                CATALOG_MASK_MIN_RADIUS_DEGREES
            ),
            "catalog_mask_min_flux1000_ph_cm2_s": CATALOG_MASK_MIN_FLUX,
            "catalog_on_exclusion_radius_rule": (
                "max(configured minimum, r68)"
                if CATALOG_ANNULUS_MASK else "configured fixed radius"
            ),
            "catalog_on_exclusion_minimum_degrees": (
                CATALOG_EXCLUSION_RADIUS_DEGREES
            ),
            "mc_area_base_samples": MC_AREA_SAMPLES,
            "mc_area_maximum_samples": MAX_MC_AREA_SAMPLES,
            "maximum_mc_to_n_off_poisson_error_ratio": (
                MAX_MC_TO_POISSON_RATIO
            ),
            "minimum_unmasked_annulus_fraction": (
                MIN_UNMASKED_ANNULUS_FRACTION
            ),
            "minimum_unmasked_off_counts": MIN_UNMASKED_OFF_COUNTS,
        },
    }]
    Path(summary_output_path()).write_text(json.dumps(summary, indent = 2))



# release per-bin detections without motion linking. ---
if STATIONARY_VALIDATION:
    if not SAVE_BIN_CANDIDATES:
        raise RuntimeError(
            "FPS_STATIONARY_VALIDATION requires FPS_SAVE_BIN_CANDIDATES=1"
        )
    print(
        "Stationary validation: saved per-bin detections and bypassed "
        "motion linking.",
        flush=True,
    )
    write_slim_json([], candidate_output_path())
    write_summary(0, 0)
    write_injection_stage_audit()
    raise SystemExit(0)



if len(ALL_CANDIDATES) == 0:
    print("Zero candidates survived - nothing to link.", flush = True)
    write_slim_json([], candidate_output_path())
    write_summary(0, 0)
    if INJECTION_STAGE_AUDIT is not None:
        INJECTION_STAGE_AUDIT["linking"].update({
            "truth_matched_pre_motion_filter_tracks": 0,
            "truth_matched_post_motion_filter_tracks": 0,
            "truth_matched_owned_output_tracks": 0,
        })
    write_injection_stage_audit()

else:
    SORTED_CANDIDATES = sorted(
        ALL_CANDIDATES,
        key = lambda candidate: (candidate["bin_index"], -candidate["sigma"]),
    )
    TRAJECTORIES = link_candidates(SORTED_CANDIDATES)
    if INJECTION_STAGE_AUDIT is not None:
        INJECTION_STAGE_AUDIT["linking"][
            "truth_matched_pre_motion_filter_tracks"
        ] = sum(
            injection_stage_audit_track_matches(item[2])
            for item in TRAJECTORIES
        )
    if SAVE_PRE_TUBE:
        PRE_TUBE_NUMBERED_TRAJECTORIES = [
            (candidate_number, *trajectory)
            for candidate_number, trajectory in enumerate(TRAJECTORIES, start = 1)
        ]
        PRE_TUBE_OUTPUT_RECORDS = build_OUTPUT_RECORDS(
            PRE_TUBE_NUMBERED_TRAJECTORIES,
            BAND_LABELS,
        )
        write_slim_json(
            PRE_TUBE_OUTPUT_RECORDS, pre_tube_candidate_output_path()
        )
        print(
            (
                f"Wrote {len(PRE_TUBE_OUTPUT_RECORDS)} pre-tube diagnostic "
                f"trajectories to {pre_tube_candidate_output_path()}"
            ),
            flush = True,
        )
    N_PRE_FILTER_TRACKS = len(TRAJECTORIES)
    TRAJECTORIES = filter_tube_like_trajectories(TRAJECTORIES)
    if INJECTION_STAGE_AUDIT is not None:
        INJECTION_STAGE_AUDIT["linking"][
            "truth_matched_post_motion_filter_tracks"
        ] = sum(
            injection_stage_audit_track_matches(item[2])
            for item in TRAJECTORIES
        )
    NUMBERED_TRAJECTORIES = [
        (candidate_number, *trajectory)
        for candidate_number, trajectory in enumerate(TRAJECTORIES, start = 1)
    ]

    OUTPUT_RECORDS = build_OUTPUT_RECORDS(
        NUMBERED_TRAJECTORIES,
        BAND_LABELS,
    )
    OUTPUT_RECORDS = apply_unique_sky_ownership(OUTPUT_RECORDS)
    if INJECTION_STAGE_AUDIT is not None:
        INJECTION_STAGE_AUDIT["linking"][
            "truth_matched_owned_output_tracks"
        ] = sum(
            injection_stage_audit_track_matches(record["points"])
            for record in OUTPUT_RECORDS
        )

    print(
        (
            f"\n>>> {len(OUTPUT_RECORDS)} TRAJECTORIES with "
            f">= {MIN_TRACK_LENGTH} linked detections:\n"
        ),
        flush = True,
    )
    print(
        (
            f"  {'Len':>4s} {'RA_start':>9s} {'DEC_start':>10s} "
            f"{'RA_end':>9s} {'DEC_end':>10s} {'Photons':>8s} "
            f"{'avg_v':>9s}  Bins"
        ),
        flush = True,
    )
    print("  " + "-" * 80, flush = True)

    for record in OUTPUT_RECORDS:
        bins_string = ",".join(
            str(bin_index) for bin_index in record["bin_indices"]
        )
        print(
            (
                f"  {record['track_length']:4d} "
                f"{record['ra_start']:9.4f} "
                f"{record['dec_start']:10.4f} "
                f"{record['ra_end']:9.4f} "
                f"{record['dec_end']:10.4f} "
                f"{record['detection_band_photons']:8d} "
                f"{record['average_velocity_deg_per_year']:9.3f}  "
                f"{bins_string}"
            ),
            flush = True,
        )

    write_slim_json(OUTPUT_RECORDS, candidate_output_path())
    write_summary(len(OUTPUT_RECORDS), N_PRE_FILTER_TRACKS)  # NEW (2.2)
    write_injection_stage_audit()
    print(
        (
            f"\nWrote {len(OUTPUT_RECORDS)} TRAJECTORIES to "
            f"{candidate_output_path()}"
        ),
        flush = True,
    )
