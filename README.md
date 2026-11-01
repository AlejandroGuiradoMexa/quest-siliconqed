# Silicon cavities for colour centres and fast electrons

A small-scale rebuild of the QUEST doctoral project on silicon quantum electrodynamics at the deep nanoscale
(theory, design and modelling of silicon cavities with deep-subwavelength confinement, their coupling to fast electrons
and to embedded colour centres). It was prepared as a work sample for the application to the QUEST PhD scholarship at
DTU Electro (Horizon Europe MSCA Doctoral Network QUEST).

Project page: https://quest-siliconqed.alejandroguirado.com (`site/`, Cloudflare Worker `quest-siliconqed`), with a paper-style version at `/paper`.

| Module | Part of the project | What is done here |
|---|---|---|
| A | Modal expansions | Quasinormal modes (QNMs): the 1D dielectric barrier of Kristensen et al., Adv. Opt. Photon. 12, 612 (2020) reproduced; a 2D finite-element QNM solver with PML (NGSolve) checked against the exact QNMs of a cylinder; 2D Purcell factor from a QNM pair vs the direct solution |
| B | Cavity design for fabrication | The G-centre nanobeam of Kim et al., Nano Lett. 25, 4347 (2025) in a 2D effective-index model, tuned to 1278 nm; a central silicon bowtie; bridge-width sweep; QNM perturbation theory vs re-simulation; random fabrication errors |
| C | Fast electrons | An aloof electron along the beam: coupling to the cavity mode vs electron energy, and the energy-loss spectrum from a single QNM vs the direct solution |
| D | Colour centres | A G centre in the cavity: dipole moment from the measured lifetime, coupling strength, Purcell factor, Lindblad master equation and coupled-mode frequencies; reading of the measured lifetimes of Kim et al. |

These are educational results from public data and simplified tools, not results of the QUEST network.

## Layout

```
src/sqed/       materials, barrier1d, cylinder (exact QNMs), fem (2D Maxwell + PML, QNMs), geometry, cavity
                (design helpers, perturbation theory), electron, jc (master equation)
scripts/        run_moduleA.py ... run_moduleD.py (write results/*.json), plots.py (figures), build_site.py
tests/          pytest unit tests (run in CI)
data/           Si optical constants (Green 2008, refractiveindex.info, CC0)
results/        module results (JSON) and figures/ (PDF + SVG)
site/           index.html (Web), paper.html (Paper), plots.js, page.js, data.json, figures/, fonts/ (Latin Modern, GUST licence)
```

## Reproduce

Requirements: Python ≥ 3.11 and NGSolve (`pip install ngsolve`).

```bash
python -m venv .venv && .venv/bin/pip install -e . ngsolve pytest
cd scripts
../.venv/bin/python run_moduleA.py                                   # ~25 min
OMP_NUM_THREADS=1 ../.venv/bin/python run_moduleB.py                 # ~30 min (5 processes)
../.venv/bin/python run_moduleC.py                                   # needs results/moduleB.json
../.venv/bin/python run_moduleD.py
../.venv/bin/python plots.py && ../.venv/bin/python build_site.py
cd ../site && python3 -m http.server 8790
```

## Simplifications

- The cavity is modelled in 2D (top view of a 220 nm membrane, effective index of the TE0 slab mode). The out-of-plane
  radiation is missing, so Q is an in-plane value (an upper bound); 3D mode volumes are estimated as A_eff × h_eff.
- Silicon is non-dispersive at 1278 nm in the 2D model; the electron is a line charge (the 2D analogue of an electron beam).
- The G centre is a two-level system; its dipole moment follows from published lifetime, Debye–Waller factor and a range of quantum efficiencies.
