# Feature Selection Rerun Workflow

## Why a full rerun is required

If feature selection becomes part of the formal analysis pipeline, all downstream stages must be rerun on the selected feature set:

1. model fitting
2. OOF threshold selection
3. temporal test inference
4. calibration / DCA / SHAP / coefficient plots
5. README and manuscript text

Otherwise, the reported figures describe the old model rather than the feature-selected model.

## Current archived baseline

These values are archived from the current public artifact manifest and result exports and should be treated as the "before feature selection" baseline.

### Main rolling-landmark relapse task

- test AUC: `0.841386067137627`
- test PR-AUC: `0.333887622324112`
- test Brier: `0.06388086865989832`
- test threshold: `0.13`

### Patient-level aggregation

- patient-level AUC: `0.7488708220415538`
- patient-level PR-AUC: `0.4529085447315304`
- patient-level threshold: `0.25`

### Current top global drivers

- `Time_In_Normal`
- `Ever_Hypo_Before`
- `ThyroidW`
- `FT3_Current`
- `Delta_TSH_k0`

## Required data input

The training scripts load the source dataset from:

```text
1003.xlsx
```

Expected location for rerun:

```text
/root/medicalAI/1003.xlsx
```

The public repository does not contain this file, and it is not currently present in the local workspace. Real reruns are blocked until the file is provided.

## Recommended rerun rule

To avoid optimistic bias, feature selection should be executed inside the model-development pipeline, not after reading final test results.

Preferred options:

1. nested feature selection inside each training fold
2. or train-set-only feature selection followed by untouched temporal test evaluation

Not recommended:

1. selecting features using all data
2. selecting features after seeing test metrics and then reusing the same test set as if it were untouched

## UV-based environment bootstrap

Once the dataset is available, the intended commands are:

```bash
cd /root/medicalAI
uv venv
source .venv/bin/activate
uv sync
```

If additional packages are needed during refactor, add them with `uv add ...` instead of editing environment state by hand.

## Planned execution order after data is available

1. create a snapshot of pre-selection outputs
2. implement real feature-selection logic in the training scripts
3. rerun the affected scripts on `1003.xlsx`
4. regenerate plots from the new metrics
5. rewrite README and manuscript-facing text from rerun outputs only
