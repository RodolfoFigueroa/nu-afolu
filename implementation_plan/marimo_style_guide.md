# Marimo Notebook Style Guide

This project uses marimo notebooks for reproducible data-science workflows. The
Chen SSP analysis will be long and assumption-heavy, so the notebooks should be
written as explanatory analysis documents rather than compact scratchpads.

Err on the side of verbosity. A future reader should understand not only what a
cell computes, but why it exists and how to interpret its output.

## Notebook Structure

Every notebook should include these high-level sections as markdown cells:

1. Title
2. Goal
3. Inputs
4. Assumptions
5. Analysis sections
6. Outputs
7. Interpretation
8. Limitations
9. Next step

The section names can vary if the notebook reads better another way, but the
same information should be present.

## Markdown Cell Expectations

Use markdown cells before every major code block or analysis section.

Markdown cells should:

- explain the purpose of the section;
- name the artifact or concept being inspected;
- define any metric before it is computed;
- explain why a plot or table matters;
- call out assumptions before applying them;
- summarize what the reader should notice after important outputs.

Avoid notebooks that are mostly code with occasional headings. The intended
reader includes both humans and AI agents who need enough narrative context to
continue the work safely.

## Code Cell Expectations

Code cells should be focused and readable.

Prefer:

- small cells with one conceptual purpose;
- named intermediate variables;
- explicit units in variable names where useful, such as `_m2` or `_ha`;
- local helper functions only when they make the notebook easier to read;
- imported reusable functions from `src/nu_afolu/` once logic stabilizes.

Avoid:

- large cells that load data, compute metrics, plot, and save outputs all at
  once;
- hidden assumptions embedded in one-line transformations;
- hard-coded paths without an explanatory configuration cell;
- reimplementing the same helper logic across notebooks after it stabilizes.

## Recommended Opening Pattern

Each notebook should begin with a markdown title cell:

```markdown
# 02 Chen 2020 Compatibility

This notebook compares Chen 2020 urban area against GLC-derived 2020
`settlements` area. The goal is to decide whether Chen is compatible enough to
support downstream scenario construction.
```

Then include a short markdown cell for inputs and assumptions before the first
substantive code cell.

## Recommended Closing Pattern

Each notebook should end with a markdown interpretation cell that states:

- what passed;
- what failed or remains uncertain;
- which artifacts were produced;
- whether the next notebook can proceed;
- what assumptions should be carried forward.

If the notebook produces no durable artifact, say so explicitly.

## Visualizations And Tables

Use plots and tables when they materially improve understanding. They are most
useful when the reader needs to compare zones or SSPs, inspect distributions,
see outliers, review validation failures, or understand final scenario results.

Do not add a plot or table after every code block by default. If a code cell
creates a simple intermediate object whose meaning is already clear from nearby
markdown, it does not need a displayed artifact.

Every important plot or table that is displayed should be introduced and
interpreted.

Before a visualization:

- explain what it compares;
- define the axes or grouped quantities;
- state what a concerning pattern would look like.

Before a table:

- explain why the rows and columns matter;
- state whether the table is diagnostic, intermediate, or a final output;
- identify any sorting, filtering, or thresholding that affects interpretation.

After a visualization or table:

- summarize the observed pattern;
- identify whether the result supports or weakens the analysis workflow;
- point out zones, SSPs, or methods that need special attention.

Prefer tables for exact values, rankings, validation reports, and artifact
schemas. Prefer plots for trends, distributions, outliers, scenario comparisons,
and relationships between quantities such as Chen urban area versus GLC
settlement area.

## Outputs And Provenance

Any notebook that writes artifacts should include a markdown section documenting:

- output paths;
- artifact schema;
- units;
- scenario and method identifiers;
- baseline and calibration choices;
- whether outputs are final, diagnostic, or intermediate.

For Chen-derived pseudo-artifacts, always state that they are diagnostic scenario
artifacts unless a later review has approved them as carbon-model inputs.

## Error Handling And Warnings

Do not let important warnings disappear in logs.

When a notebook detects a risky condition, show it in a visible table or
markdown summary. Examples:

- missing zone artifacts;
- failed table-contract checks;
- large Chen/GLC 2020 mismatch;
- small-zone ratio instability;
- negative Chen deltas;
- source-class allocation exhaustion;
- failed mass-balance checks.

The notebook should make it obvious whether the reader can continue to the next
step.

## Reusable Logic

Notebooks may start with exploratory code, but stable mechanics should move into
`src/nu_afolu/`.

Good candidates for reusable code include:

- artifact loading helpers;
- table-contract validators;
- Chen/GLC compatibility metric functions;
- transition-prior calculations;
- pseudo-transition allocation functions;
- mass-balance checks;
- scenario manifest writing.

When logic is moved into `src/nu_afolu/`, keep the notebook markdown explaining
what the helper does and why it is being used.

## Naming Conventions

Use notebook filenames with numeric prefixes so the intended reading order is
clear:

```text
00_artifact_inventory.py
01_historical_contract_checks.py
02_chen_2020_compatibility.py
03_spatial_resolution_diagnostics.py
04_historical_settlement_transition_priors.py
05_chen_future_deltas.py
06_pseudo_transition_allocation.py
07_carbon_model_scenario_runs.py
08_scenario_comparison_readout.py
```

Use clear variable names that include units when relevant:

- `area_m2`
- `area_ha`
- `settlement_delta_m2`
- `emissions_tco2e`
- `ratio_error`
- `absolute_error_m2`

## Tone

The notebooks should read like a careful analysis memo:

- direct;
- explicit about uncertainty;
- generous with context;
- careful about causal or validation claims;
- clear about what is observed, projected, assumed, or diagnostic.
