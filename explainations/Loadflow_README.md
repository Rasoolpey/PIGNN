# Load-flow Step — Solver implementation and validation

This document explains the load-flow implementation in the project, how to run the demo, implementation notes, validation strategy, expected inputs/outputs, and next steps for contingency analysis and comparison with DIgSILENT results.

## Purpose

The load-flow step computes steady-state voltages, angles, branch flows and losses on the graph representation of the power system. The implementation is designed to operate on the per-phase graph structures produced by the visualization/graph builder step and supports physics-aware solver components.

## Where the code lives

- `load_flow_demo.py` — demo script that runs the load-flow on sample scenarios and prints / stores results.
- `physics/load_flow_solver.py` — primary solver implementation (Newton-Raphson or iterative solver depending on scenario).
- `physics/fixed_load_flow_solver.py` — helper routines and fixed-point style solvers used for special cases.
- `physics/impedance_matrix.py` — constructs Z/Y matrices from branch data and handles per-phase impedance conversions.
- `core/graph_base.py` and `data/graph_builder.py` — graph inputs consumed by the solver.

## How to run

1. Ensure required Python packages are installed (numpy, scipy, h5py; see project notes).
2. From repository root run:

   ```bash
   python load_flow_demo.py
   ```

3. The demo loads a sample `.h5` scenario, runs the solver, and outputs results to the console and (optionally) to files under `Contingency Analysis/contingency_out/`.

4. For headless runs, ensure the demo writes outputs to disk and doesn't rely on interactive plotting.

## Contract

- Inputs: graph object or path to scenario HDF5 with bus data, generator injections, load vectors, branch impedance data, and initial voltage guesses.
- Outputs: per-bus voltages (magnitude and angle), per-branch flows and losses, convergence status and iteration counts. Outputs are saved in JSON/HDF5 or printed depending on demo config.
- Error modes: non-convergence, singular Ybus (islanding without reference), missing generator or slack bus specification.

## Solver details and choices

- The implementation uses a Newton-Raphson style method for balanced subproblems and supports modular substitution with alternative solvers in `physics/fixed_load_flow_solver.py`.
- For per-phase unbalanced analysis, the solver builds per-phase admittance/impedance matrices using `physics/impedance_matrix.py` and solves each phase while handling coupling terms as needed.
- The solver returns a detailed diagnostics object including residual norms and iteration history to aid debugging and comparison.

## Validation and comparison plan (current)

- Unit tests: small synthetic networks with known solutions should be added to automatically verify correctness (not included yet).
- Manual validation: compare voltages and flows produced by `load_flow_demo.py` with reference outputs from DIgSILENT for the same scenario. The repository already contains some scenario outputs under `Contingency Analysis/contingency_scenarios/` and `contingency_out/`.
- Quantitative metrics for comparison: per-bus voltage magnitude error (abs and percentage), voltage angle difference (degrees), per-branch MVA/active/reactive flow errors, and convergence iteration counts.

## Edge cases and pitfalls

- Non-convergence: for ill-conditioned networks, add improved initialization (flat start vs previous solution) and damping or continuation methods.
- Islanding and reference bus: when the network separates into islands, ensure each island has a slack/reference or attach a virtual slack to stabilize the solve.
- Missing data: branches without impedance must be cleaned or assigned realistic defaults before solving.

## Storage and reproducibility

- The demo and solver can write outputs into the `Contingency Analysis/contingency_out/` folder for later analysis and comparison with DIgSILENT exported data. Keep the scenario HDF5 and solver config together so experiments are reproducible.

## Next steps (immediate)

- Implement automated numeric comparison utilities that read DIgSILENT exports and compute the metrics listed above.
- Add a small harness to run a batch of contingency scenarios, store solver results, and produce CSV/JSON summaries for each scenario (this will be the core of Stage 3).
- Add unit tests for small networks and a regression test comparing a known scenario to a stored baseline.

---

If you want, I can add the comparison harness and a small script to parse DIgSILENT exported CSVs and compute per-bus/per-branch error metrics; tell me where the DIgSILENT outputs will be placed or I can add a config option to point to them.