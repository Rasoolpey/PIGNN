# Visualization Step — Phase graph visualization

This document explains the visualization step of the project: how graphs for the three phases are derived from the input dataset, which files and modules implement visualization, how to run the demo, the expected inputs/outputs, validation notes and next steps.

## Purpose

The visualization step provides a clear, human-inspectable representation of the power network for each phase (A, B, C). This helps verify graph topology, node and edge attributes, and identify data issues before running physics-based solvers or learning models.

## Where the code lives

- `visualization_demo.py` — top-level demo script demonstrating how to build and visualize three phase graphs.
- `visualization/graph_plotter.py` — plotting utilities used by the demo.
- `core/graph_base.py`, `core/node_types.py`, `core/edge_types.py` — graph data structures and type definitions used to build the visualizable graph.
- `data/h5_loader.py`, `data/enhanced_h5_loader.py`, `data/graph_builder.py` — data ingestion and graph-building logic. The demo loads `.h5` scenario files located in `data/` or `Contingency Analysis/contingency_scenarios/`.

## How to run

1. Prepare a Python environment with the usual scientific stack (numpy, scipy, matplotlib, h5py, networkx if not already installed).
2. From the repository root run:

   python visualization_demo.py

3. The script will load a sample `.h5` scenario, construct three per-phase graphs and display / save figures (depending on the script's internal settings).

If the machine is headless or you want to save plots only, modify `visualization/graph_plotter.py` to set the backend or change calls to `plt.show()` to `plt.savefig()`.

## Data inputs

- Primary inputs are HDF5 scenario files (example: `data/scenario_0.h5` or files inside `Contingency Analysis/contingency_scenarios/`).
- Graph construction expects node and branch attributes (for example: impedances, phases, bus identifiers). The loader extracts attributes and builds per-phase edges so that each resulting graph represents the electrical connectivity for that phase.

## Outputs

- Visual plots of each phase graph. Depending on the plotting configuration, these may be displayed on screen or saved to disk.
- The demo may also optionally save graph objects (notebooks or serialized pickles) if configured.

## Contract (small)

- Inputs: path to `.h5` scenario with expected dataset keys (bus ids, branches, impedances, phase info).
- Outputs: PNG or interactive plot windows showing graphs for phases A/B/C; and optionally a serialized graph object.
- Error modes: missing datasets in the HDF5 file, malformed impedance arrays, or mismatched bus IDs will raise exceptions in the loader.

## Validation and what to look for

- Confirm that node counts match the expected number of buses for each phase.
- Check that edges connect the correct bus IDs and that per-edge attributes (impedance, length, status) are populated.
- Look for disconnected subgraphs which may indicate missing data or phase-coupling that isn't modeled by per-phase graphs.

## Edge cases and pitfalls

- Unbalanced networks: per-phase graphs can differ in topology. Visual inspection helps spot phase-specific islands.
- Missing phase assignment: some branches may be missing explicit phase labels — the graph builder uses heuristics and might assign defaults.
- Multi-terminal elements: transformers and multi-terminal devices may be represented differently; ensure the graph builder normalizes these consistently.

## Next steps (technical)

- Add programmatic save options for plots and deterministic layout seeding to make comparisons repeatable.
- Write unit tests for the graph builder to assert expected node/edge counts and attribute shapes for a small synthetic network.
- Integrate graph visualization with the load-flow pipeline to overlay computed voltages or flows on the plots.

## Notes about reproducibility

- If you share plots in papers or reports, store the exact scenario file and the seed used for layout. Consider adding a small script that exports a zipped artifacts bundle (scenario + plot + script args) for reproducibility.

---

If you need a more opinionated, interactive visualization (e.g., Plotly with hover tooltips for bus attributes), we can add a small web viewer in a later step.
