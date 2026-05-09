# Data Card: DiaTrend-Style CGM Exports (Research Use)

## Dataset name and access

The **DiaTrend** study data are distributed via Synapse (see the project README link). Access is **credential-based** and governed by the data contributor’s terms. This public repository **does not redistribute** subject-level files.

## What this codebase assumes on disk

Under `data/raw/`:

- `demographics.xlsx` with at least columns: `Subject`, `Age`, `Hemoglobin A1C`.  
- Per-subject workbooks: `Subject<N>.xlsx` with a sheet named **`CGM`** and a column **`mg/dl`**.

Filenames are parsed with the pattern `Subject<id>.xlsx` to join demographics rows.

## Synthetic test fixtures

`tests/fixtures/` contains **fabricated** spreadsheets to run CI without private data. They are **not** clinically realistic longitudinal records and must never be used to report performance.

## Preprocessing summary

- **CGM**: coerce `mg/dl` to numeric; strip text artifacts; **drop** rows where glucose is missing after coercion (no interpolation in the default path).  
- **Demographics**: strip whitespace; coerce `Age` via regex digit extraction; coerce HbA1c numerically; fill remaining NaNs with the column mean **within the loaded demographics table** (research default, not a clinical imputation policy).  
- **Scaling**: `StandardScaler` fit on the stacked feature matrix of all successfully loaded subjects before modeling.

## Proxy label (research only)

Default binary label:

- Class **1** if `HbA1c <= 7.0`, else class **0** (aligned with the prior project script).

This is a **coarse operationalization** for software demonstration. It is **not** equivalent to time-in-range, hypoglycemia risk, or provider-defined control.

## Known limitations

- **Leaky feature setup** if HbA1c is used as input and also defines the label—documented for honesty in modeling experiments.  
- **Short CGM windows** and irregular sampling are not modeled explicitly; summaries collapse time structure to three scalars.  
- **Population shift**: models trained on one cohort may fail silently on new devices, geographies, or insulin modalities.

## Responsible use

Do not merge synthetic fixtures with real patient exports without clear provenance separation. Maintain audit logs of which Synapse snapshot was used for any reported metric.
