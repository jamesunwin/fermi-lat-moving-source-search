# Reproducibility

## Analysis identity

All current response and interpretation products use
`fl16y_v41_incident_e2_response_v1`. The artificial-source model assumes an
incident spectrum dN/dE proportional to E^-2 over 1--100 GeV. At sky direction
Omega, detected energies are sampled from a distribution proportional to
E^-2 E10(E,Omega), where E10 is the energy-resolved ten-year LAT exposure.
Energy dispersion is neglected.

The calibration comprises 2,250 source tests: five exact annual photon counts,
three angular speeds, three Galactic-latitude bands, ten representative sky
regions per band, and five repetitions. The separate annual-count validation
contains 120 sources.

## Prepared-data and batch-search paths

The preparation and batch-search commands use `allsky_queries/data` as their
common default prepared-data directory. For clarity, the same path can be
given explicitly, together with the externally downloaded FL16Y-v41 catalogue:

```bash
python moving/prepare_allsky_v5.py prepare \
  --output-root allsky_queries/data
python moving/run_fps_moving_v5_rois_New.py \
  --roi-root allsky_queries/data \
  --catalog-fits data/gll_psc_v41.fit
```

The prepared photon files and catalogue are external inputs and are not
redistributed in this compact repository.

## Compact and external products

The committed release contains all tabulated responses used in the physical
limits. Full trial workspaces and public LAT photon files are external because
they are too large for a source repository. Their absence does not affect
verification of the released tables: `verify_release.py` checks file hashes,
analysis identities, dimensions, headline values, and cross-product numerical
relations.

## Rebuilding from the compact release

The source-number limits follow from the released recovered-track upper limit
and efficiencies. The model tables use the released effective-volume kernel.
No extrapolation is made beyond the declared count, angular-speed, or
luminosity support.

Paper: [arxiv link will appear here]
