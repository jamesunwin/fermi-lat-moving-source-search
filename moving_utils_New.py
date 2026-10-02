import numpy as np


ROI_RA = 7.614279
ROI_DEC = 4.86103
ROI_RADIUS_DEGREES = 7.5
SECONDS_PER_DAY = 86400.0
SECONDS_PER_MONTH = 30.0 * SECONDS_PER_DAY

# Frozen 2016-03-09..2026-03-09 analysis window. Velocities use a 365-day
# analysis year.
WINDOW_START = 479174404
WINDOW_END = 794707205
FERMI_START = WINDOW_START
FERMI_END = WINDOW_END
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY

ENERGY_MIN_MEV = 1000.0
ENERGY_MAX_MEV = 100000.0


def event_class_pass(event_class_column, class_bit):
    """Decode EVENT_CLASS integer, FITS X-bit, or packed-byte columns."""
    values = np.asarray(event_class_column)
    if class_bit <= 0:
        return np.ones(len(values), dtype=bool)
    bit_indices = [
        index for index in range(class_bit.bit_length())
        if class_bit & (1 << index)
    ]
    if values.ndim == 1 and np.issubdtype(values.dtype, np.integer):
        return (values & class_bit) > 0
    if values.ndim == 2 and values.dtype == np.bool_:
        # Astropy exposes FITS X columns in their documented MSB-first order.
        columns = [values.shape[1] - 1 - index for index in bit_indices]
        if columns and min(columns) >= 0:
            return np.any(values[:, columns], axis=1)
    if (
        values.ndim == 2
        and np.issubdtype(values.dtype, np.integer)
        and values.dtype.itemsize == 1
    ):
        # A structured-array rewrite can expose an N-bit FITS X column as
        # ceil(N/8) packed bytes. Position zero is the MSB of the first byte.
        width = values.shape[1] * 8
        passed = np.zeros(len(values), dtype=bool)
        for index in bit_indices:
            position = width - 1 - index
            if position < 0:
                continue
            byte_index, offset = divmod(position, 8)
            passed |= (values[:, byte_index] & (1 << (7 - offset))) != 0
        if bit_indices and max(bit_indices) < width:
            return passed
    raise ValueError("Unsupported EVENT_CLASS representation.")


def angular_separation_degrees(ra1, dec1, ra2, dec2):
    ra1 = np.radians(ra1)
    dec1 = np.radians(dec1)
    ra2 = np.radians(ra2)
    dec2 = np.radians(dec2)

    sin_half_dec = np.sin((dec2 - dec1) / 2.0)
    sin_half_ra = np.sin((ra2 - ra1) / 2.0)
    argument = (
        sin_half_dec * sin_half_dec
        + np.cos(dec1) * np.cos(dec2) * sin_half_ra * sin_half_ra
    )
    return np.degrees(2.0 * np.arcsin(np.minimum(1.0, np.sqrt(argument))))


def circular_mean_degrees(angles_degrees, weights = None):
    angles_radians = np.radians(angles_degrees)
    mean_sin = np.average(np.sin(angles_radians), weights = weights)
    mean_cos = np.average(np.cos(angles_radians), weights = weights)
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)


def spherical_mean_coordinates(ra_degrees, dec_degrees, weights = None):
    """Mean sky position from weighted Cartesian unit vectors."""
    ra = np.radians(np.asarray(ra_degrees, dtype = float))
    dec = np.radians(np.asarray(dec_degrees, dtype = float))
    if ra.shape != dec.shape or ra.size == 0:
        raise ValueError("RA and Dec must be non-empty arrays with equal shape.")
    vectors = np.column_stack((
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ))
    mean_vector = np.average(vectors, axis = 0, weights = weights)
    norm = float(np.linalg.norm(mean_vector))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("Spherical mean is undefined for this sky distribution.")
    mean_vector /= norm
    mean_ra = float(
        np.degrees(np.arctan2(mean_vector[1], mean_vector[0])) % 360.0
    )
    mean_dec = float(np.degrees(np.arcsin(np.clip(mean_vector[2], -1.0, 1.0))))
    return mean_ra, mean_dec


def analysis_bin_edges(bin_size_days = 365.0):
    """Fixed analysis bin edges, with the remainder included in the last bin."""
    bin_size_seconds = float(bin_size_days) * SECONDS_PER_DAY
    if bin_size_seconds <= 0.0:
        raise ValueError("bin_size_days must be positive.")
    span_seconds = float(WINDOW_END - WINDOW_START)
    n_bins = max(1, int(np.floor(span_seconds / bin_size_seconds)))
    edges = WINDOW_START + np.arange(n_bins + 1, dtype = float) * bin_size_seconds
    edges[-1] = float(WINDOW_END)
    return edges


def psf_radius_degrees(energy_mev):
    return max(0.8 * (energy_mev / 1000.0) ** (-0.8), 0.1)


def circle_overlap_area(radius_1, radius_2, separation):
    if separation >= radius_1 + radius_2:
        return 0.0

    if separation <= abs(radius_1 - radius_2):
        return np.pi * min(radius_1, radius_2) ** 2

    radius_1_squared = radius_1 ** 2
    radius_2_squared = radius_2 ** 2
    separation_squared = separation ** 2

    term_1 = np.arccos(
        np.clip(
            (
                separation_squared + radius_1_squared - radius_2_squared
            )
            / (2.0 * separation * radius_1),
            -1.0,
            1.0,
        )
    )
    term_2 = np.arccos(
        np.clip(
            (
                separation_squared + radius_2_squared - radius_1_squared
            )
            / (2.0 * separation * radius_2),
            -1.0,
            1.0,
        )
    )
    term_3 = 0.5 * np.sqrt(
        max(
            0.0,
            (-separation + radius_1 + radius_2)
            * (separation + radius_1 - radius_2)
            * (separation - radius_1 + radius_2)
            * (separation + radius_1 + radius_2),
        )
    )
    return radius_1_squared * term_1 + radius_2_squared * term_2 - term_3


def roi_limited_annulus_area(
    centroid_distance_from_roi_center,
    inner_radius_degrees,
    outer_radius_degrees,
    roi_radius_degrees = ROI_RADIUS_DEGREES,
):
    outer_overlap_area = circle_overlap_area(
        roi_radius_degrees,
        outer_radius_degrees,
        centroid_distance_from_roi_center,
    )
    inner_overlap_area = circle_overlap_area(
        roi_radius_degrees,
        inner_radius_degrees,
        centroid_distance_from_roi_center,
    )
    return max(0.0, outer_overlap_area - inner_overlap_area)


# Clip the on-region circle to the analysis region with the same geometry used
# for the background annulus.
def roi_limited_circle_area(
    centroid_distance_from_roi_center,
    radius_degrees,
    roi_radius_degrees = ROI_RADIUS_DEGREES,
):
    return circle_overlap_area(
        roi_radius_degrees,
        radius_degrees,
        centroid_distance_from_roi_center,
    )



def lima_sigma(n_on, n_off, alpha):
    expected_background = alpha * n_off
    if n_on <= expected_background:
        return 0.0

    if n_off == 0:
        return float(np.sqrt(2.0 * n_on * np.log((1.0 + alpha) / alpha)))

    term_on = n_on * np.log(
        ((1.0 + alpha) / alpha) * (n_on / (n_on + n_off))
    )
    term_off = n_off * np.log((1.0 + alpha) * (n_off / (n_on + n_off)))
    return float(np.sqrt(max(0.0, 2.0 * (term_on + term_off))))


def signed_ra_offsets_degrees(ra_values, reference_ra):
    return (ra_values - reference_ra + 180.0) % 360.0 - 180.0


def tangent_plane_coordinates(ra_values, dec_values, reference_ra, reference_dec):
    x_values = signed_ra_offsets_degrees(
        ra_values,
        reference_ra,
    ) * np.cos(np.radians(reference_dec))
    y_values = dec_values - reference_dec
    return x_values, y_values
