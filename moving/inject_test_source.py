"""Low-level photon injection utilities for the moving-source search.

Injects a synthetic moving point source directly into a copy of an ROI's
photon files, then prints the exact command to run the search on it. No
Fermitools executable is required; the realistic King option reads the
installed CALDB PSF FITS file. Injected photons are cloned from real events
in the same time bin AND the same photon file (LAT queries are chunked by
time, and each
file's GTI table only covers its own chunk - clones must stay in their source
file or the GTI cut removes them). Clones are repositioned onto a
constant-velocity track with energy-dependent PSF scatter. TIME,
ZENITH_ANGLE, EVENT_CLASS, incidence angle, and conversion type are inherited
from real events so quality cuts pass by construction. The current response
calibration installs the exposure-folded energy sampler from
``incident_e2_response_v1`` before calling these utilities.

Usage (from the project root):
    python moving/inject_test_source.py --roi-dir allsky_queries/data/roi_005 \
        --photons-per-year 30 --velocity 0.5 --seed 0

Then run the printed FPS_* command and compare candidates.json with the
printed ground-truth file. Lower --photons-per-year in steps
(30 -> 20 -> 12 -> 8) to find the recovery threshold: a first efficiency
measurement.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moving_utils_New import (
    WINDOW_START,
    SECONDS_PER_DAY,
    angular_separation_degrees,
    psf_radius_degrees,
)

BIN_SECONDS = 365.0 * SECONDS_PER_DAY
N_BINS = 10
# 2D Gaussian: 68% containment radius = 1.509 sigma
PSF_SIGMA_FROM_R68 = 1.0 / 1.509
PSF_IRF_NAME = "psf_P8R3_SOURCE_V3_FB.fits"


def resolve_psf_irf_file(path=None):
    """Find the installed P8R3_SOURCE_V3 FRONT/BACK PSF calibration file."""
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    caldb = os.environ.get("CALDB")
    if caldb:
        candidates.append(
            Path(caldb) / "data/glast/lat/bcf/psf" / PSF_IRF_NAME
        )
    fermi_dir = os.environ.get("FERMI_DIR")
    if fermi_dir:
        candidates.append(
            Path(fermi_dir) / "data/caldb/data/glast/lat/bcf/psf"
            / PSF_IRF_NAME
        )
    candidates.append(
        Path(sys.prefix) / "share/fermitools/data/caldb/data/glast/lat"
        / "bcf/psf" / PSF_IRF_NAME
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {PSF_IRF_NAME}. Set CALDB or FERMI_DIR, or pass "
        f"--psf-irf-file. Searched:\n  {searched}"
    )


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_mixture(centres, value):
    """Two neighbouring grid indices and linear weights, clamped at edges."""
    centres = np.asarray(centres, dtype=float)
    if value <= centres[0]:
        return ((0, 1.0),)
    if value >= centres[-1]:
        return ((len(centres) - 1, 1.0),)
    upper = int(np.searchsorted(centres, value))
    lower = upper - 1
    fraction = (value - centres[lower]) / (
        centres[upper] - centres[lower]
    )
    return ((lower, 1.0 - fraction), (upper, fraction))


class KingPsfSampler:
    """Sampler for the P8R3_SOURCE_V3 IRF's tabulated double-King PSF.

    The Fermitools Psf3 response interpolates *distributions*, rather than
    fitted parameters.  We implement that convention by drawing one of the
    four neighbouring (log-energy, cos(theta)) table distributions with its
    bilinear interpolation weight, then drawing from that cell's normalized
    double-King profile.
    """

    def __init__(self, irf_file=None):
        self.path = resolve_psf_irf_file(irf_file)
        self.sha256 = _sha256(self.path)
        self.tables = {}
        with fits.open(self.path, memmap=False) as hdul:
            for conversion_name in ("FRONT", "BACK"):
                data = hdul[f"RPSF_{conversion_name}"].data
                scaling = np.asarray(
                    hdul[f"PSF_SCALING_PARAMS_{conversion_name}"].data[
                        "PSFSCALE"
                    ][0],
                    dtype=float,
                )
                energy_centres = np.sqrt(
                    np.asarray(data["ENERG_LO"][0], dtype=float)
                    * np.asarray(data["ENERG_HI"][0], dtype=float)
                )
                costheta_centres = 0.5 * (
                    np.asarray(data["CTHETA_LO"][0], dtype=float)
                    + np.asarray(data["CTHETA_HI"][0], dtype=float)
                )
                self.tables[conversion_name] = {
                    "log_energy": np.log10(energy_centres),
                    "costheta": costheta_centres,
                    "scaling": scaling,
                    **{
                        name: np.asarray(data[name][0], dtype=float)
                        for name in (
                            "NTAIL", "SCORE", "STAIL", "GCORE", "GTAIL"
                        )
                    },
                }

    @staticmethod
    def _cell_core_fraction(table, theta_index, energy_index):
        score = table["SCORE"][theta_index, energy_index]
        stail = table["STAIL"][theta_index, energy_index]
        ntail = table["NTAIL"][theta_index, energy_index]
        return 1.0 / (1.0 + ntail * (stail / score) ** 2)

    @staticmethod
    def _king_cdf(x, sigma, gamma):
        x = np.asarray(x, dtype=float)
        return 1.0 - (
            1.0 + x * x / (2.0 * gamma * sigma * sigma)
        ) ** (1.0 - gamma)

    def _grid_mixture(self, energy_mev, theta_degrees, conversion_type):
        conversion_name = "FRONT" if int(conversion_type) == 0 else "BACK"
        table = self.tables[conversion_name]
        energy_mix = _axis_mixture(
            table["log_energy"], math.log10(float(energy_mev))
        )
        costheta_mix = _axis_mixture(
            table["costheta"], math.cos(math.radians(theta_degrees))
        )
        return table, tuple(
            (theta_index, energy_index, theta_weight * energy_weight)
            for theta_index, theta_weight in costheta_mix
            for energy_index, energy_weight in energy_mix
        )

    def _scale_radians(self, table, energy_mev):
        c0, c1, beta = table["scaling"]
        return math.sqrt(
            (c0 * (float(energy_mev) / 100.0) ** beta) ** 2 + c1 ** 2
        )

    def sample_separation_degrees(
        self, rng, energy_mev, theta_degrees, conversion_type,
    ):
        table, mixture = self._grid_mixture(
            energy_mev, theta_degrees, conversion_type
        )
        choice = rng.random()
        cumulative = 0.0
        theta_index, energy_index, _ = mixture[-1]
        for candidate_theta, candidate_energy, weight in mixture:
            cumulative += weight
            if choice <= cumulative:
                theta_index = candidate_theta
                energy_index = candidate_energy
                break
        core_fraction = self._cell_core_fraction(
            table, theta_index, energy_index
        )
        if rng.random() < core_fraction:
            sigma = table["SCORE"][theta_index, energy_index]
            gamma = table["GCORE"][theta_index, energy_index]
        else:
            sigma = table["STAIL"][theta_index, energy_index]
            gamma = table["GTAIL"][theta_index, energy_index]
        # Inverse radial CDF of the normalized 2D King/Moffat component.
        u = min(rng.random(), np.nextafter(1.0, 0.0))
        scaled_radius = sigma * math.sqrt(
            2.0 * gamma * ((1.0 - u) ** (1.0 / (1.0 - gamma)) - 1.0)
        )
        separation = math.degrees(
            self._scale_radians(table, energy_mev) * scaled_radius
        )
        # Psf3 normalizes on the physical 0--90 degree domain. At 1--100 GeV
        # this rejection is extremely rare, but makes that support explicit.
        if separation > 90.0:
            return self.sample_separation_degrees(
                rng, energy_mev, theta_degrees, conversion_type
            )
        return separation

    def radial_cdf(
        self, radius_degrees, energy_mev, theta_degrees, conversion_type,
    ):
        """Tabulated/interpolated ideal containment CDF used for validation."""
        table, mixture = self._grid_mixture(
            energy_mev, theta_degrees, conversion_type
        )
        scale = self._scale_radians(table, energy_mev)
        x = np.radians(radius_degrees) / scale
        total = np.zeros_like(np.asarray(x, dtype=float))
        for theta_index, energy_index, weight in mixture:
            core_fraction = self._cell_core_fraction(
                table, theta_index, energy_index
            )
            total += weight * (
                core_fraction * self._king_cdf(
                    x,
                    table["SCORE"][theta_index, energy_index],
                    table["GCORE"][theta_index, energy_index],
                )
                + (1.0 - core_fraction) * self._king_cdf(
                    x,
                    table["STAIL"][theta_index, energy_index],
                    table["GTAIL"][theta_index, energy_index],
                )
            )
        return total


def offset_on_sphere(ra_degrees, dec_degrees, separation_degrees, bearing):
    """Apply an angular displacement at a bearing (radians east of north)."""
    ra = math.radians(ra_degrees)
    dec = math.radians(dec_degrees)
    distance = math.radians(separation_degrees)
    new_dec = math.asin(
        math.sin(dec) * math.cos(distance)
        + math.cos(dec) * math.sin(distance) * math.cos(bearing)
    )
    new_ra = ra + math.atan2(
        math.sin(bearing) * math.sin(distance) * math.cos(dec),
        math.cos(distance) - math.sin(dec) * math.sin(new_dec),
    )
    return math.degrees(new_ra) % 360.0, math.degrees(new_dec)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-dir", required=True,
                        help="e.g. allsky_queries/data/roi_005")
    parser.add_argument("--out-root", default="allsky_queries/data_injected")
    parser.add_argument("--photons-per-year", type=int, default=30)
    parser.add_argument("--velocity", type=float, default=0.5,
                        help="track angular velocity, deg/yr")
    parser.add_argument("--position-angle", type=float, default=40.0,
                        help="track direction, deg east of north")
    parser.add_argument("--seed", type=int, default=0)
    # Optional track-midpoint offset on
    # the tangent plane at the ROI centre. Defaults keep the track centred
    # (the simple positive-control geometry); run_calibration_v5.py samples
    # these uniformly over the cap so the efficiency map includes edge
    # losses and is survey-averaged rather than best-case. ---
    parser.add_argument("--offset-x", type=float, default=0.0,
                        help="track midpoint offset east, tangent-plane deg")
    parser.add_argument("--offset-y", type=float, default=0.0,
                        help="track midpoint offset north, tangent-plane deg")
    # Injected photon energies can be drawn from
    # a power law E^-Gamma (1-100 GeV) instead of inheriting the local
    # background spectrum; PSF scatter follows the drawn energy. Default 2.0
    # (the analysis assumption); 0 = inherit the background spectrum (the
    # original clone behaviour). Running the efficiency grid at 1.5/2.0/2.5
    # yields a spectral-dependence band for diagnostic limit plots.
    parser.add_argument("--spectral-index", type=float, default=2.0,
                        help="source photon index Gamma; 0 = inherit bkg")
    parser.add_argument(
        "--psf-model", choices=("gaussian", "king"), default="gaussian",
        help="PSF scatter model (production calibration uses King)",
    )
    parser.add_argument(
        "--psf-irf-file", default=None,
        help=(
            "P8R3_SOURCE_V3 FRONT/BACK PSF FITS; with --psf-model king, "
            "defaults to the file in CALDB"
        ),
    )
    return parser.parse_args()


# Inverse-CDF power-law sampler used by the low-level utility.
def sample_power_law(rng, n, gamma, e_min=1000.0, e_max=100000.0):
    u = rng.random(n)
    if abs(gamma - 1.0) < 1e-9:
        return e_min * (e_max / e_min) ** u
    g = 1.0 - gamma
    return (e_min ** g + u * (e_max ** g - e_min ** g)) ** (1.0 / g)


# track midpoint can now be offset
# from the ROI centre; photons placed outside the ROI are naturally lost to
# the search (edge losses), which is exactly what a survey-averaged
# efficiency must include. ---
def track_position(roi_ra, roi_dec, velocity, position_angle_deg, t_years,
                   offset_x=0.0, offset_y=0.0):
    """Constant-velocity track with midpoint at (centre + offset)."""
    displacement = velocity * (t_years - 0.5 * N_BINS)
    pa = math.radians(position_angle_deg)
    x = offset_x + displacement * math.sin(pa)   # east, tangent-plane deg
    y = offset_y + displacement * math.cos(pa)   # north, tangent-plane deg
    dec = roi_dec + y
    ra = roi_ra + x / math.cos(math.radians(dec))
    return ra % 360.0, dec



# the injection logic is now a callable
# function so run_calibration_v5.py can drive it over a (rate x velocity x
# seed) grid; the CLI behaviour is unchanged. ---
def build_injection(
    roi_dir_relative, out_root="allsky_queries/data_injected",
    photons_per_year=30, velocity=0.5, position_angle=40.0, seed=0,
    offset_x=0.0, offset_y=0.0, spectral_index=2.0,
    psf_model="gaussian", psf_irf_file=None, quiet=False,
):
    args = argparse.Namespace(
        roi_dir=roi_dir_relative, out_root=out_root,
        photons_per_year=photons_per_year, velocity=velocity,
        position_angle=position_angle, seed=seed,
        offset_x=offset_x, offset_y=offset_y,
        spectral_index=spectral_index,
        psf_model=psf_model, psf_irf_file=psf_irf_file,
    )
    return _run_injection(args, quiet=quiet)


# One physical injection is routed into every overlapping
# analysis cap, preserving identical source photon rows across ROI copies. ---
def roi_memberships_for_truth(
    truth_points, roi_infos, cap_radius_degrees=None,
):
    """Return memberships using each tile's recorded analysis-buffer radius.

    ``cap_radius_degrees`` is retained only as an explicit diagnostic override.
    Production injection campaigns leave it unset so routing follows the same
    ``analysis_radius_degrees`` metadata as Stage A.
    """
    memberships = {}
    truth_ra = np.asarray([point["ra"] for point in truth_points], dtype=float)
    truth_dec = np.asarray([point["dec"] for point in truth_points], dtype=float)
    for roi_id, info in sorted(roi_infos.items()):
        if cap_radius_degrees is None:
            if "analysis_radius_degrees" not in info:
                raise ValueError(
                    f"{roi_id} lacks analysis_radius_degrees for injection routing"
                )
            roi_cap_radius = float(info["analysis_radius_degrees"])
        else:
            roi_cap_radius = float(cap_radius_degrees)
        if not 0.0 < roi_cap_radius <= 180.0:
            raise ValueError(
                f"{roi_id} has invalid injection cap radius {roi_cap_radius}"
            )
        separations = angular_separation_degrees(
            truth_ra,
            truth_dec,
            float(info["ra_degrees"]),
            float(info["dec_degrees"]),
        )
        bins = [
            int(truth_points[index]["bin_index"])
            for index in np.flatnonzero(separations <= roi_cap_radius)
        ]
        if bins:
            memberships[roi_id] = bins
    return memberships


def load_roi_infos(roi_root):
    roi_root = Path(roi_root)
    infos = {}
    for info_path in sorted(roi_root.glob("roi_*/query_info.json")):
        info = json.loads(info_path.read_text())
        roi_id = info.get("tile_id", info.get("label", info_path.parent.name))
        info["_roi_dir"] = str(info_path.parent)
        infos[roi_id] = info
    if not infos:
        raise ValueError(f"no ROI metadata below {roi_root}")
    return infos


def _extract_master_source_rows(handle, template_roi_dir):
    """Read only rows appended by build_injection, keyed by block basename."""
    original_paths = [
        Path(line.strip())
        for line in (Path(template_roi_dir) / "events.txt").read_text().splitlines()
        if line.strip()
    ]
    original_paths = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in original_paths
    ]
    injected_paths = [
        Path(line.strip())
        for line in Path(handle["events_file"]).read_text().splitlines()
        if line.strip()
    ]
    injected_paths = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in injected_paths
    ]
    original_by_name = {path.name: path for path in original_paths}
    source_rows = {}
    for injected_path in injected_paths:
        original_path = original_by_name[injected_path.name]
        with fits.open(original_path, memmap=True) as original_hdul:
            n_original = len(original_hdul[1].data)
        with fits.open(injected_path, memmap=False) as injected_hdul:
            appended = np.asarray(injected_hdul[1].data[n_original:]).copy()
        if len(appended):
            source_rows[injected_path.name] = appended
    return source_rows


def _write_target_injection(
    target_roi_dir, target_info, source_rows, truth, output_dir,
    cap_radius_degrees,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = [
        Path(line.strip())
        for line in (target_roi_dir / "events.txt").read_text().splitlines()
        if line.strip()
    ]
    manifest_paths = [
        path if path.is_absolute() else PROJECT_ROOT / path
        for path in manifest_paths
    ]
    output_paths = []
    n_injected = 0
    target_roi_id = target_info.get(
        "tile_id", target_info.get("label", target_roi_dir.name),
    )
    target_membership = roi_memberships_for_truth(
        truth["per_bin_truth"],
        {target_roi_id: target_info},
        cap_radius_degrees=cap_radius_degrees,
    )
    membership_bins = set(target_membership.get(target_roi_id, []))
    for source_path in manifest_paths:
        rows = source_rows.get(source_path.name)
        selected = None
        if rows is not None and len(rows):
            # A physical photon appears in each nominal ROI whose cap contains
            # its track position at that epoch.  Apply the same spatial mask
            # to the one shared realization; do not draw an independent copy
            # for each ROI.
            row_bins = np.floor(
                (np.asarray(rows["TIME"], dtype=float) - WINDOW_START)
                / BIN_SECONDS
            ).astype(int)
            row_bins = np.clip(row_bins, 0, N_BINS - 1)
            selected = rows[np.isin(row_bins, sorted(membership_bins))]
        if selected is None or len(selected) == 0:
            output_paths.append(source_path)
            continue

        destination = output_dir / source_path.name
        with fits.open(source_path, memmap=False) as hdul:
            events = hdul[1].data
            n_old = len(events)
            new_hdu = fits.BinTableHDU.from_columns(
                hdul[1].columns,
                nrows=n_old + len(selected),
                header=hdul[1].header,
            )
            for name in hdul[1].columns.names:
                source = np.asarray(selected[name])
                target = new_hdu.data[name][n_old:]
                if (
                    target.ndim == 2
                    and target.dtype == np.bool_
                    and source.ndim == 2
                    and np.issubdtype(source.dtype, np.integer)
                    and source.dtype.itemsize == 1
                ):
                    unpacked = np.unpackbits(
                        source.astype(np.uint8, copy=False),
                        axis=1,
                        bitorder="big",
                    )
                    target[:] = unpacked[:, :target.shape[1]]
                else:
                    target[:] = source
            out_hdus = fits.HDUList(
                [hdul[0].copy(), new_hdu]
                + [hdul[index].copy() for index in range(2, len(hdul))]
            )
            out_hdus.writeto(destination, overwrite=True)
        output_paths.append(destination)
        n_injected += len(selected)

    events_file = output_dir / "events.txt"
    events_file.write_text("\n".join(
        str(path.relative_to(PROJECT_ROOT))
        if path.is_relative_to(PROJECT_ROOT) else str(path)
        for path in output_paths
    ) + "\n")
    target_truth = dict(truth)
    target_truth.update({
        "target_roi": target_info.get(
            "tile_id", target_info.get("label", target_roi_dir.name),
        ),
        "n_injected_into_roi": n_injected,
        "cap_radius_degrees": cap_radius_degrees,
    })
    truth_file = output_dir / "injection_truth.json"
    truth_file.write_text(json.dumps(target_truth, indent=2))
    return {
        "label": output_dir.name,
        "out_dir": str(output_dir),
        "events_file": str(events_file),
        "truth_file": str(truth_file),
        "roi_ra": float(target_info["ra_degrees"]),
        "roi_dec": float(target_info["dec_degrees"]),
        "truth": target_truth,
        "query_info": target_info,
        "n_injected": n_injected,
    }


def build_overlap_injections(
    roi_root, template_roi, out_root="allsky_queries/data_injected",
    photons_per_year=30, velocity=0.5, position_angle=40.0, seed=0,
    offset_x=0.0, offset_y=0.0, spectral_index=2.0,
    psf_model="gaussian", psf_irf_file=None,
    cap_radius_degrees=None, quiet=False,
):
    """Create identical source photons in every cap intersecting the track."""
    roi_root = Path(roi_root)
    if not roi_root.is_absolute():
        roi_root = PROJECT_ROOT / roi_root
    roi_infos = load_roi_infos(roi_root)
    if template_roi not in roi_infos:
        raise ValueError(f"unknown template ROI {template_roi}")
    template_dir = Path(roi_infos[template_roi]["_roi_dir"])
    output_root = Path(out_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    master_root = output_root / "_master_injection"
    master = build_injection(
        template_dir,
        out_root=master_root,
        photons_per_year=photons_per_year,
        velocity=velocity,
        position_angle=position_angle,
        seed=seed,
        offset_x=offset_x,
        offset_y=offset_y,
        spectral_index=spectral_index,
        psf_model=psf_model,
        psf_irf_file=psf_irf_file,
        quiet=True,
    )
    try:
        source_rows = _extract_master_source_rows(master, template_dir)
        if cap_radius_degrees is None:
            cap_radii = {}
            for roi_id, info in roi_infos.items():
                if "analysis_radius_degrees" not in info:
                    raise ValueError(
                        f"{roi_id} lacks analysis_radius_degrees for "
                        "injection routing"
                    )
                cap_radii[roi_id] = float(info["analysis_radius_degrees"])
            cap_radius_source = "query_info.analysis_radius_degrees"
        else:
            cap_radii = {
                roi_id: float(cap_radius_degrees) for roi_id in roi_infos
            }
            cap_radius_source = "explicit_override"
        invalid_radii = {
            roi_id: radius
            for roi_id, radius in cap_radii.items()
            if not 0.0 < radius <= 180.0
        }
        if invalid_radii:
            raise ValueError(f"invalid injection cap radii: {invalid_radii}")
        memberships = roi_memberships_for_truth(
            master["truth"]["per_bin_truth"],
            roi_infos,
            cap_radius_degrees=cap_radius_degrees,
        )
        used_radii = sorted({
            cap_radii[roi_id] for roi_id in memberships
        })
        radius_tag = (
            f"{used_radii[0]:g}" if len(used_radii) == 1 else "metadata"
        )
        shared_label = (
            f"overlap_{master['label']}_cap{radius_tag}"
        )
        handles = []
        for roi_id in sorted(memberships):
            target_info = roi_infos[roi_id]
            target_dir = Path(target_info["_roi_dir"])
            output_dir = output_root / shared_label / roi_id
            handle = _write_target_injection(
                target_dir,
                target_info,
                source_rows,
                master["truth"],
                output_dir,
                cap_radii[roi_id],
            )
            handle["membership_bins"] = memberships[roi_id]
            if handle["n_injected"] > 0:
                handles.append(handle)
        result = {
            "label": shared_label,
            "template_roi": template_roi,
            "truth": master["truth"],
            "memberships": memberships,
            "handles": handles,
            "n_target_rois": len(handles),
            "cap_radius_degrees": (
                used_radii[0] if len(used_radii) == 1 else None
            ),
            "cap_radius_source": cap_radius_source,
            "cap_radius_degrees_by_target_roi": {
                roi_id: cap_radii[roi_id] for roi_id in sorted(memberships)
            },
            "same_physical_photons": True,
        }
        manifest = output_root / shared_label / "overlap_injection.json"
        manifest.write_text(json.dumps({
            key: value for key, value in result.items() if key != "handles"
        }, indent=2))
        if not quiet:
            print(
                f"Injected one shared source into {len(handles)} overlapping "
                f"ROI cap(s): {', '.join(h['query_info']['tile_id'] for h in handles)}"
            )
        return result
    finally:
        shutil.rmtree(master["out_dir"], ignore_errors=True)



def main():
    args = parse_args()
    _run_injection(args, quiet=False)


def format_search_command(
    events_file, roi_ra, roi_dec, roi_radius_degrees, label,
    catalog_fits="data/gll_psc_v41.fit",
):
    """Return the standalone search command for one injected ROI."""
    return (
        f'FPS_EVENTS_FILE="{events_file}" \\\n'
        f"FPS_ROI_RA={roi_ra} FPS_ROI_DEC={roi_dec} \\\n"
        f"FPS_ROI_RADIUS_DEG={roi_radius_degrees} \\\n"
        f"FPS_OUTPUT_LABEL=injection/{label} \\\n"
        f"FPS_CATALOG_FITS={catalog_fits} \\\n"
        f"python moving/fps_moving_v5_New.py"
    )


def _run_injection(args, quiet=False):
    rng = np.random.default_rng(args.seed)
    psf_model = getattr(args, "psf_model", "gaussian")
    if psf_model not in {"gaussian", "king"}:
        raise ValueError(f"unknown PSF model: {psf_model}")
    king_psf = (
        KingPsfSampler(getattr(args, "psf_irf_file", None))
        if psf_model == "king" else None
    )

    roi_dir = PROJECT_ROOT / args.roi_dir
    info = json.loads((roi_dir / "query_info.json").read_text())
    roi_ra, roi_dec = info["ra_degrees"], info["dec_degrees"]
    if "analysis_radius_degrees" not in info:
        raise ValueError(
            f"{roi_dir}: query_info.json lacks analysis_radius_degrees"
        )
    analysis_radius = float(info["analysis_radius_degrees"])
    roi_radius = float(info.get("search_radius_degrees", analysis_radius))

    photon_files = [
        line.strip()
        for line in (roi_dir / "events.txt").read_text().splitlines()
        if line.strip()
    ]

    # Pass 1: read TIME/RA/DEC per file; verify metadata consistency and
    # collect per-bin template pools as (file_index, row_index) pairs.
    pools = {k: [] for k in range(N_BINS)}
    all_seps = []
    for file_index, rel in enumerate(photon_files):
        with fits.open(PROJECT_ROOT / rel) as hdul:
            times = np.asarray(hdul[1].data["TIME"], dtype=float)
            ras = np.asarray(hdul[1].data["RA"], dtype=float)
            decs = np.asarray(hdul[1].data["DEC"], dtype=float)
        all_seps.append(
            angular_separation_degrees(ras, decs, roi_ra, roi_dec)
        )
        bins = np.floor((times - WINDOW_START) / BIN_SECONDS).astype(int)
        bins = np.clip(bins, 0, N_BINS - 1)
        for k in range(N_BINS):
            for row in np.flatnonzero(bins == k):
                pools[k].append((file_index, int(row)))

    # GUARD: the photons must actually surround the claimed ROI centre. If
    # query_info.json was rewritten after a generate-centers rerun (the
    # Fibonacci layout moves EVERY centre when --n-rois changes), the data
    # and metadata no longer match and any search on this folder is invalid.
    median_sep = float(np.median(np.concatenate(all_seps)))
    if median_sep > roi_radius:
        raise SystemExit(
            f"ABORT: median photon separation from the claimed ROI centre "
            f"({roi_ra:.3f}, {roi_dec:.3f}) is {median_sep:.1f} deg - the "
            f"photons in {roi_dir} were downloaded for a DIFFERENT centre. "
            f"Delete this folder and its query_manifest.json entry, then "
            f"re-run lat_roi_query.py submit-download."
        )

    empty_bins = [k for k in range(N_BINS) if not pools[k]]
    if empty_bins:
        raise SystemExit(
            f"ABORT: no template photons in bins {empty_bins}; the photon "
            f"files do not cover the full analysis window."
        )

    # Choose clones per bin, grouped by source file.
    chosen_by_file = {i: [] for i in range(len(photon_files))}
    truth_points = []
    for k in range(N_BINS):
        pool = pools[k]
        picks = rng.choice(
            len(pool), size=args.photons_per_year,
            replace=len(pool) < args.photons_per_year,
        )
        for p in picks:
            file_index, row = pool[int(p)]
            chosen_by_file[file_index].append(row)
        ra_k, dec_k = track_position(
            roi_ra, roi_dec, args.velocity, args.position_angle, k + 0.5,
            args.offset_x, args.offset_y,
        )
        truth_points.append({"bin_index": k, "ra": ra_k, "dec": dec_k})

    spectrum_tag = (
        f"g{args.spectral_index:g}" if args.spectral_index > 0 else "gbkg"
    )
    label = (
        f"{info['label']}_inj_v{args.velocity:g}"
        f"_n{args.photons_per_year}_s{args.seed}_{spectrum_tag}"
        f"{'_psfking' if psf_model == 'king' else ''}"
    )
    out_dir = PROJECT_ROOT / args.out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 2: rewrite every file, appending its own clones (keeps GTIs valid).
    n_total_injected = 0
    out_paths = []
    for file_index, rel in enumerate(photon_files):
        src = PROJECT_ROOT / rel
        dst = out_dir / src.name
        rows = chosen_by_file[file_index]
        with fits.open(src) as hdul:
            events = hdul[1].data
            n_old = len(events)
            if rows:
                injected = events[np.asarray(rows)]  # fancy index -> copy
                t_years = (
                    np.asarray(injected["TIME"], dtype=float) - WINDOW_START
                ) / BIN_SECONDS
                # energies drawn from the
                # target power law (and written into the clones) rather than
                # inherited from the background spectrum; Gamma=0 inherits. ---
                if args.spectral_index > 0:
                    energy = sample_power_law(
                        rng, len(rows), args.spectral_index,
                    )
                    injected["ENERGY"] = energy
                else:
                    energy = np.asarray(injected["ENERGY"], dtype=float)

                new_ra = np.empty(len(rows))
                new_dec = np.empty(len(rows))
                for i in range(len(rows)):
                    ra_i, dec_i = track_position(
                        roi_ra, roi_dec, args.velocity,
                        args.position_angle, t_years[i],
                        args.offset_x, args.offset_y,
                    )
                    if psf_model == "gaussian":
                        sigma = (
                            psf_radius_degrees(energy[i])
                            * PSF_SIGMA_FROM_R68
                        )
                        new_dec[i] = dec_i + rng.normal(0.0, sigma)
                        new_ra[i] = (
                            ra_i + rng.normal(0.0, sigma)
                            / math.cos(math.radians(new_dec[i]))
                        ) % 360.0
                    else:
                        separation = king_psf.sample_separation_degrees(
                            rng,
                            energy[i],
                            float(injected["THETA"][i]),
                            int(injected["CONVERSION_TYPE"][i]),
                        )
                        new_ra[i], new_dec[i] = offset_on_sphere(
                            ra_i, dec_i, separation,
                            rng.uniform(0.0, 2.0 * math.pi),
                        )
                injected["RA"] = new_ra
                injected["DEC"] = new_dec

                new_hdu = fits.BinTableHDU.from_columns(
                    hdul[1].columns, nrows=n_old + len(rows),
                    header=hdul[1].header,
                )
                for name in hdul[1].columns.names:
                    new_hdu.data[name][n_old:] = injected[name]
                out_hdus = fits.HDUList(
                    [hdul[0].copy(), new_hdu]
                    + [hdul[i].copy() for i in range(2, len(hdul))]
                )
                out_hdus.writeto(dst, overwrite=True)
                n_total_injected += len(rows)
            else:
                hdul.writeto(dst, overwrite=True)
        out_paths.append(dst)

    # Permit isolated temporary overlap-injection
    # roots outside the repository as well as the historical in-tree root. ---
    (out_dir / "events.txt").write_text(
        "\n".join(
            str(path.relative_to(PROJECT_ROOT))
            if path.is_relative_to(PROJECT_ROOT) else str(path)
            for path in out_paths
        ) + "\n"
    )

    truth = {
        "roi": info["label"],
        "roi_ra": roi_ra,
        "roi_dec": roi_dec,
        "velocity_deg_per_year": args.velocity,
        "position_angle_deg": args.position_angle,
        # Record the track geometry.
        "offset_x_deg": args.offset_x,
        "offset_y_deg": args.offset_y,
        # Record the injected spectrum.
        "spectral_index": args.spectral_index,
        "psf_model": psf_model,
        "psf_irf": (
            {
                "name": "P8R3_SOURCE_V3",
                "event_types": "FRONT+BACK (evtype=3)",
                "path": str(king_psf.path),
                "sha256": king_psf.sha256,
                "interpolation": (
                    "bilinear mixture of tabulated distributions in "
                    "log10(energy) and cos(theta)"
                ),
            }
            if king_psf is not None else {
                "name": "energy_dependent_gaussian",
                "r68_model": "max(0.8*(E/GeV)^-0.8, 0.1) deg",
                "sigma_from_r68": PSF_SIGMA_FROM_R68,
            }
        ),
        "photons_per_year": args.photons_per_year,
        "detected_count_convention": (
            "exact_integer_photons_per_annual_bin_not_poisson_mean"
        ),
        "detected_energy_distribution": {
            "form": (
                "power_law_in_detected_energy"
                if args.spectral_index > 0 else "inherited_background_energy"
            ),
            "photon_index": (
                args.spectral_index if args.spectral_index > 0 else None
            ),
            "minimum_mev": 1000.0,
            "maximum_mev": 100000.0,
            "response_forward_folded": False,
            "interpretation": (
                "conditional detector-level calibration; incident source "
                "flux requires a separate exposure-response conversion"
            ),
        },
        "seed": args.seed,
        "n_injected": int(n_total_injected),
        "per_bin_truth": truth_points,
    }
    (out_dir / "injection_truth.json").write_text(json.dumps(truth, indent=2))

    if not quiet:
        print(
            f"Injected {n_total_injected} photons "
            f"({args.photons_per_year}/yr x {N_BINS} bins) along a "
            f"{args.velocity:g} deg/yr track."
        )
        print(f"Ground truth: {out_dir / 'injection_truth.json'}")
        print("\nRun the search on the injected data with:\n")
        print(format_search_command(
            out_dir / "events.txt",
            roi_ra,
            roi_dec,
            analysis_radius,
            label,
        ))
        print(
            f"\nThen inspect moving/results_v5/injection/{label}/candidates.json"
            f" and compare with the truth file."
        )

    # Machine-readable handle for the calibration driver.
    return {
        "label": label,
        "out_dir": str(out_dir),
        "events_file": str(out_dir / "events.txt"),
        "truth_file": str(out_dir / "injection_truth.json"),
        "roi_ra": roi_ra,
        "roi_dec": roi_dec,
        "truth": truth,
    }



if __name__ == "__main__":
    main()
