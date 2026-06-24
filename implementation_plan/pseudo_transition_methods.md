# Pseudo-Transition Methods

Chen SSP projections can constrain future urban or settlement growth, but they
do not provide a complete AFOLU transition matrix. This document describes
candidate methods for allocating Chen-derived settlement demand back to source
classes so that diagnostic pseudo-`transition_table` artifacts can be created.

These methods are not approved carbon-model inputs by default. They are
candidate scenario-construction methods that must be evaluated and documented.

## Shared Inputs

Every method should use the same conceptual inputs:

- baseline AFOLU area by zone;
- Chen settlement-growth demand by zone, SSP, and interval;
- AFOLU label list from `nu_afolu.constants.LABEL_LIST`;
- historical transition evidence where applicable;
- explicit handling of negative Chen deltas;
- explicit output provenance.

The first implementation should focus on settlement expansion. It should not
invent non-settlement AFOLU-to-AFOLU transitions without a separate data source
or assumption.

## Shared Outputs

Every method must produce:

- pseudo-`area_table` objects with years and AFOLU columns;
- pseudo-`transition_table` objects with `year`, `start`, and `end`
  coordinates;
- validation reports;
- provenance records including SSP, baseline, calibration, and method.

The transition table should represent the selected interval semantics clearly.
If the workflow uses decadal intervals, then `year=2020` should be documented as
the start of the 2020 -> 2030 interval rather than silently mimicking an annual
2020 -> 2021 transition.

## Method 1: Historical Settlement-Transition Shares

### Intent

Allocate future settlement expansion according to historical source-class shares
for observed transitions into `settlements`.

This is the most interpretable first method because it uses observed transition
behavior from the same upstream artifact family as the carbon model inputs.

### Required Historical Prior

From historical `transition_table` artifacts:

- select transitions where `end == "settlements"`;
- exclude `start == "settlements"` when estimating new settlement expansion
  shares;
- compute source-class shares by zone where data are sufficient;
- compute pooled source-class shares as a fallback.

### Allocation Logic

For each zone, SSP, and future interval:

1. Read settlement-growth demand for the interval.
2. Choose source shares:
   - zone-specific shares if stable and non-empty;
   - pooled shares otherwise.
3. Multiply demand by source shares.
4. Cap allocations by available source-class area.
5. Redistribute capped remainder according to remaining eligible shares.
6. Record any unresolved demand if all eligible sources are exhausted.
7. Emit transitions from allocated source classes into `settlements`.
8. Carry unmodified area for classes not involved in settlement expansion.

### Strengths

- Directly grounded in historical project artifacts.
- Easy to explain in markdown and review.
- Provides a defensible default for early diagnostic carbon runs.

### Weaknesses

- Assumes historical settlement expansion patterns remain relevant.
- May be weak in zones with little observed settlement growth.
- Can allocate growth to classes that are historically common but spatially
  implausible in future decades.

### Required Diagnostics

- Share table by zone and pooled fallback.
- Number of zones using zone-specific versus pooled shares.
- Allocation exhaustion by source class.
- Unresolved demand by zone, SSP, and interval.

## Method 2: Source-Availability-Constrained Allocation

### Intent

Allocate future settlement expansion using source availability as the dominant
constraint. Historical shares can still guide proportional allocation, but the
method emphasizes that a class cannot supply more land than exists.

### Allocation Logic

For each zone, SSP, and future interval:

1. Identify eligible non-settlement source classes.
2. Compute available area for each source class at the start of the interval.
3. Optionally weight availability by historical source shares.
4. Allocate settlement demand proportionally to weighted available area.
5. Cap allocations at available source area.
6. Redistribute any remainder until demand is satisfied or eligible area is
   exhausted.
7. Record unresolved demand if needed.

### Strengths

- Prevents impossible source depletion.
- Can handle zones where historical transition priors are sparse.
- Makes area constraints visible.

### Weaknesses

- Availability alone does not mean suitability for urban expansion.
- Without spatial suitability, it may overuse abundant classes that are unlikely
  sources.

### Required Diagnostics

- Eligible source area by interval.
- Weighted allocation shares.
- Exhausted classes.
- Unresolved demand.
- Difference from pure historical-share allocation.

## Method 3: Priority Or Ranking Allocation

### Intent

Allocate future settlement expansion according to an explicit priority order of
source classes.

This method is useful as a transparent sensitivity case. It is easy to explain
and can encode expert judgment, but it should not be treated as more empirical
than the assumptions used to define the ranking.

### Default Priority Order

Use this first-pass priority order unless a later review changes it:

1. `croplands`
2. `pastures`
3. `grasslands`
4. `shrublands`
5. `other`
6. `forests_secondary`
7. `forests_primary`
8. `forests_mangroves`
9. `wetlands`
10. `flooded`

`settlements` should not be used as a source for new settlement expansion.

The order is intentionally conservative about converting high-carbon or
hydrologically sensitive classes. It is still an assumption and must be labeled
as such in the notebook.

### Allocation Logic

For each zone, SSP, and future interval:

1. Start with the first source class in the priority list.
2. Allocate as much settlement demand as possible from that class.
3. Move to the next class only after the current class is exhausted.
4. Continue until demand is satisfied or no eligible source area remains.
5. Record unresolved demand if needed.

### Strengths

- Very transparent.
- Useful for bounding or stress-testing carbon-model sensitivity.
- Requires fewer historical data assumptions.

### Weaknesses

- Strongly assumption-driven.
- Ignores observed transition shares unless the ranking is derived from them.
- May create unrealistic abrupt source-class exhaustion.

### Required Diagnostics

- Source classes used by interval.
- Classes exhausted by interval.
- Share of demand allocated by priority rank.
- Difference from historical-share and availability-constrained methods.

## Optional Diagnostic: Settlement-Only Baseline

### Intent

Represent settlement growth without assigning a detailed source class. This can
be useful as a diagnostic only if the carbon model can accept or approximate
such a case.

### Caution

The existing transition-table contract requires a concrete `start` class and
`end` class. A settlement-only baseline therefore cannot be a normal
`transition_table` unless it chooses a source class, distributes across a
synthetic source, or the carbon model provides a supported aggregate mode.

Default recommendation:

- Do not pass this method to the carbon model in the first implementation.
- Use it only as a demand-side diagnostic in notebooks.

## Negative Delta Handling

All methods should share the first-pass negative delta policy:

- Detect negative Chen urban-area deltas.
- Report them by zone, SSP, and interval.
- Clip negative settlement demand to zero for expansion allocation.
- Record clipped negative area as unresolved mismatch.
- Do not create `settlements -> non-settlements` transitions unless a separate
  de-urbanization method is designed.

## Required Validation For Every Method

Before any pseudo-artifact is passed to the carbon model:

- values must be finite and non-negative;
- labels must match `LABEL_LIST`;
- transition dimensions must be `year`, `start`, and `end`;
- missing transition combinations must be explicit zeros;
- allocated source area must not exceed available source area;
- settlement gains must reconcile with the demand table after clipping and
  unresolved demand accounting;
- area-table and transition-table totals must reconcile within a documented
  tolerance;
- provenance must be recoverable from paths, metadata, or a saved manifest.

