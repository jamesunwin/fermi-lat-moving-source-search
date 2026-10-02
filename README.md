# Fermi-LAT Moving Gamma-Ray Source Search

This repository contains the code and compact numerical release for an all-sky
search for steadily emitting point sources moving across the gamma-ray sky. The
analysis uses ten years of 1--100 GeV Fermi-LAT Pass 8 SOURCE-class data and is
calibrated over angular speeds of 0.25--0.8 deg yr^-1.

The current analysis identity is `fl16y_v41_incident_e2_response_v1`. Artificial
sources have an incident photon spectrum proportional to E^-2. Their detected
energies are drawn from that spectrum folded through the position-dependent LAT
exposure, with energy dispersion neglected. This energy assignment determines
the PSF scatter and clustering band of each artificial photon.

## Result at a glance

- Real-sky candidate tracks: 577
- Randomized-sky expectation: 532.750 +/- 26.988 tracks
- Empirical plus-one probability: 4/61 = 0.0656
- 90% upper limit on recovered signal tracks: 80.757
- Best exact-count population limit: 98.185 sources at 35 photons per annual
  bin and 0.5 deg yr^-1
- Best common-flux population limit: 123.051 sources at
  6.486e-10 ph cm^-2 s^-1 and 0.25 deg yr^-1

The real-data search and 60 randomized-sky campaigns depend only on measured
photon properties and the frozen search rules. The incident-spectrum update
changes the artificial-source recovery calibration, physical-flux response,
effective volume, and model translations; it does not change the real or
randomized candidate counts.

## Repository layout

- `moving/`: production search, injection, geometry, and null-generation code.
- `release/fl16y_v41_incident_e2_response_v1/`: the current numerical release,
  illustrative model translations, figures, compact inputs, manifest, and
  verifier.
- `tests/`: focused tests that do not require the large photon dataset or the
  excluded injection workspaces.
- `docs/`: data-access and reproducibility notes.

The 2,250 production injection workspaces, 120 validation workspaces, original
LAT event files, and energy-resolved exposure cube are intentionally not stored
in Git because of their size. Their compact derived tables and identifying
hashes are included in the release.

## Quick verification

Create the environment and verify every released file:

```bash
conda env create -f environment-fps.yml
conda activate fps
python release/fl16y_v41_incident_e2_response_v1/verify_release.py
python -m unittest discover -v -s tests
```

The PSF test is skipped when the local Fermi-LAT CALDB is unavailable; the
release-manifest and numerical-consistency checks do not require CALDB.

## Paper

[arxiv link will appear here]

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is released under
the BSD 3-Clause License. Fermi-LAT data and instrument-response products remain
subject to their originating collaboration and archive terms.
