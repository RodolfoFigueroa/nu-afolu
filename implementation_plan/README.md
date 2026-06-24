# Chen SSP Notebook Implementation Plan

This directory contains the implementation roadmap for evaluating whether Chen
SSP urban-surface projections can be transformed into scenario-compatible
artifacts for the downstream carbon model.

The carbon model itself is intentionally downstream of the land-use pipeline. It
expects only the two aggregate artifacts produced by the existing Dagster flow:

- `area_table`: area by year and AFOLU class.
- `transition_table`: transition area by start year, start class, and end class.

The implementation described here should determine whether Chen projections can
support diagnostic future versions of those same artifacts. It should not treat
Chen-derived artifacts as approved carbon-model inputs until the assumptions and
allocation method are reviewed.

## Source Context

Read these project docs before implementing the notebooks:

- [`docs/upstream.md`](../docs/upstream.md) documents the historical Dagster
  artifacts and the exact table contracts.
- [`docs/data_provenance.md`](../docs/data_provenance.md) documents Chen SSP
  provenance, compatibility concerns, and feasibility assumptions.

Those files are the source of truth for artifact semantics. The planning files
in this directory describe how to turn that context into a reproducible marimo
analysis workflow.

## Documentation Map

- [`notebook_roadmap.md`](notebook_roadmap.md) defines the planned marimo
  notebooks, their inputs, outputs, analyses, and acceptance criteria.
- [`analysis_contracts.md`](analysis_contracts.md) defines shared contracts,
  assumptions, naming conventions, and validation expectations used across the
  notebook workflow.
- [`pseudo_transition_methods.md`](pseudo_transition_methods.md) describes
  candidate methods for converting Chen settlement deltas into pseudo-transition
  artifacts.
- [`marimo_style_guide.md`](marimo_style_guide.md) defines readability and
  structure requirements for the notebooks.

## Intended Notebook Sequence

The recommended workflow is:

1. `00_artifact_inventory.py`
2. `01_historical_contract_checks.py`
3. `02_chen_2020_compatibility.py`
4. `03_spatial_resolution_diagnostics.py`
5. `04_historical_settlement_transition_priors.py`
6. `05_chen_future_deltas.py`
7. `06_pseudo_transition_allocation.py`
8. `07_carbon_model_scenario_runs.py`
9. `08_scenario_comparison_readout.py`

The sequence is intentionally staged. Early notebooks validate the historical
baseline and Chen compatibility. Middle notebooks derive future settlement
demand and allocation assumptions. Later notebooks create pseudo-artifacts, run
the carbon model, and compare scenario results.

## Implementation Principles

- Prefer explicit, verbose analysis over compact code-only notebooks.
- Use markdown cells to introduce every major section, explain why it exists,
  and interpret outputs.
- Use plots and tables when they help the reader understand an important
  intermediate or final result. Do not add them mechanically after every code
  block.
- Keep reusable mechanics in `src/nu_afolu/` once they stabilize.
- Preserve the historical `area_table` and `transition_table` contracts exactly
  for any pseudo-artifacts.
- Treat Chen-derived outputs as diagnostic scenario artifacts unless a later
  review promotes a method.
- Record provenance for every scenario output: SSP, baseline choice, allocation
  method, calibration choice, code version, and source data version when known.

## Success Criteria

The implementation is successful when a reader can open the notebooks in order
and understand:

- what historical artifacts are available;
- whether those artifacts satisfy the expected contracts;
- how Chen 2020 urban area compares with GLC 2020 settlements;
- how resolution mismatch affects the analysis;
- which historical transitions inform future settlement-source allocation;
- how future Chen settlement deltas are calculated;
- how candidate pseudo-transition methods behave;
- how carbon outputs vary by SSP and allocation method;
- which assumptions are strong enough for decision-making and which remain
  unresolved.
