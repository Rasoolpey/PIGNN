# Physics-Informed Graph Learning for Power Systems (PIGNN)

This repository contains code, data, and documentation for a research project implementing physics-informed graph learning methods for power system analysis. The project follows a staged roadmap; each stage implements a distinct capability that together form a pipeline for graph-based power system modeling, load-flow simulation, and contingency analysis.

This README is intentionally high-level and will be updated as the project progresses.

## Goals and roadmap (high-level)

- Stage 1 — Visualization (completed): build graph representations of power networks and produce visualizations for the three electrical phases to inspect topology and attributes.
- Stage 2 — Load-flow solver (completed): implement a load-flow solver (physics-informed and/or conventional) operating on the graph representation. Validate results against expected operating points.
- Stage 3 — Contingency analysis (in-progress): run N-1 and selected N-k scenarios, store scenario outputs, and compare results with DIgSILENT outputs (or other reference tools).
- Stage 4 — Model integration and learning: integrate physics-informed graph neural networks (GNNs) to learn corrections or accelerate repeated power flow computations.
- Stage 5 — Evaluation and deployment: comprehensive evaluation, automated comparison pipelines, and packaging for reproducible experiments.

## What is in this repository (quick map)

- `core/` — graph base classes and node/edge type definitions.
- `data/` — loaders and sample `.h5` scenarios used for experiments.
- `physics/` — load-flow and solver implementations, impedance handling, and coupling models.
- `visualization/` — graph plotting utilities and demo scripts.
- `Contingency Analysis/` — outputs and scenario data for contingency experiments.
- `explainations/` — documentation and step-by-step notes. (See the READMEs here for details on Visualization and Loadflow stages.)
- `load_flow_demo.py`, `visualization_demo.py` — demo scripts used to run the two completed stages.

## How the pieces fit together

The research pipeline is organized so that the graph construction and visualization (Stage 1) provide a quick sanity check on topology and data attributes. The load-flow solver (Stage 2) consumes the same graph structures and physical parameters to compute voltages, flows, and losses. Contingency analysis (Stage 3) will iterate the load flow under perturbed network states (line/generator outages) and compare outcomes with DIgSILENT reference outputs to quantify differences.

## Quick start (local)

1. Ensure you have a Python 3.8+ environment.
2. Install dependencies (if using a virtual environment):

   pip install -r requirements.txt

3. Inspect or run the visualization demo:

   python visualization_demo.py

4. Run the load-flow demo:

   python load_flow_demo.py

Notes
- There is no `requirements.txt` included in the repository by default. Add one that matches your environment. The code uses numpy, scipy, h5py, matplotlib and may use networkx; add packages as needed.
- The `explainations/` folder contains two more detailed READMEs that document the completed stages in depth.

## Next steps

- Add automated tests and a minimal `requirements.txt`.
- Implement the contingency analysis pipeline and the comparison utilities against DIgSILENT outputs.
- Update this README with experiment results and a reproducible script for running the full pipeline.

---

For details about the Visualization and Load-flow steps see the two READMEs in the `explainations/` folder.
