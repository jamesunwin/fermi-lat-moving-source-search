# FL16Y incident-E^-2 response release

This is the sole current numerical release. Artificial sources use an incident
E^-2 spectrum folded through the position-dependent, energy-resolved ten-year
LAT exposure. Detected energies therefore follow a density proportional to
E^-2 E10(E,Omega) over 1--100 GeV; energy dispersion is neglected.

## Contents

- `headline_results.json`: compact headline values.
- `results/`: candidate, null, efficiency, flux-limit, effective-volume, and
  abundance products.
- `models/`: reproducible illustrative dressed-PBH and evaporating-PBH tables.
- `figures/`: PDF and PNG versions of all response-dependent figures.
- `inputs/`: compact survey geometry, exposure summary, and null time model.
- `manifest.json`: SHA-256 inventory of released files.
- `verify_release.py`: integrity and scientific-consistency verifier.

The real search and randomized-sky ensemble are unchanged by the artificial
source spectrum. The injected-source calibration, physical-flux conversion,
effective volume, and illustrative model translations use the response defined
here.

Run `python verify_release.py` from this directory, or pass no arguments when
calling it from the repository root.

Paper: [arxiv link will appear here]
