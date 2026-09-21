# Pacific Classifier — Package Builder

Build a **heart-failure phenotype classifier** that runs on *your* institution's
variables, using the synthetic reference data from the
[Pacific Preserved project](https://pacific-preserved.fr/).

This repository is a **toolkit for data scientists**. You point it at the list of
variables your institution actually measures, and it trains an XGBoost ensemble on
**synthetic Pacific data restricted to those variables**, then packages the result
into a self-contained classifier you can hand to end users to classify new subjects.

- **Project website**: https://pacific-preserved.fr/
- **Publications**: https://pacific-preserved.fr/communication/
- **Partners**: https://pacific-preserved.fr/partners/

## Citation

If you use this classifier in your research, please cite:

> Hulot, J.-S., Hervé, P.-Y., Girerd, N. *et al.* Multimodal data-driven clustering analysis of heart failure patients identifies two distinct patterns of heart failure with preserved ejection fraction: Results from the PACIFIC-PRESERVED study. *Nat Commun* (2026). https://doi.org/10.1038/s41467-026-77642-6

DOI: [10.1038/s41467-026-77642-6](https://doi.org/10.1038/s41467-026-77642-6)

---

## ⚠️ Disclaimer — research and testing use only (not for clinical use)

> [!WARNING]
> **The classifiers produced by this toolkit are provided for research and testing
> purposes only. They are NOT medical devices and must NOT be used for clinical
> diagnosis, treatment, patient management, or any other medical decision-making.**

Each classifier is trained **entirely on a small synthetic reference cohort**. As a
result it may **overfit to artifacts of that synthetic data** and is not guaranteed to
generalize to real patients: a model that scores well on the synthetic data may behave
very differently on real measurements. Performance figures reported by this pipeline
reflect the synthetic data only and should **not** be interpreted as evidence of
real-world clinical validity.

Even for serious research, a classifier intended to produce trustworthy, real-world
results should be **trained on real data** rather than on the synthetic reference.
Parties interested in doing so should contact the project partners through the project
website ([pacific-preserved.fr/partners](https://pacific-preserved.fr/partners/)) to
discuss data access and collaboration.

This software and the classifiers it produces are provided **"as is", without warranty
of any kind**, express or implied. To the fullest extent permitted by law, the Pacific
Preserved project, FEALINX, and the project's partners and contributors accept **no
liability** for any use of this toolkit or its outputs. Any clinical, diagnostic, or
other use is undertaken **entirely at the user's own risk and responsibility**, and the
user is solely responsible for validating results and for compliance with all applicable
laws, regulations, and approvals.

---

## Who is this for?

There are two distinct roles, and **two distinct pieces of documentation**:

| Role | What they do | Tool | Documentation |
|------|--------------|------|---------------|
| **Builder** (data scientist) | Builds a classifier customized to their institution's variable set | `create_exportable_package.py` (this repo) | **This README** |
| **End user** (researcher) | Classifies new subjects with the finished package (research / testing only) | `pacific_classifier.py` (inside the generated package) | The `README.md` *inside* the generated package |

If you received a finished package and just want to classify data, you don't need
this repository — read the README that ships *inside* your package instead.

---

## How it works

The classifier is **never trained on your subjects' data**. It is trained entirely on
the project's **synthetic** reference cohort. Your dataset is used only to decide
*which variables* the classifier is allowed to use — namely the **intersection** of
the variables you have and the variables present in the synthetic reference.

```
  Synthetic reference            Your variable list
  (759 subjects × 557 vars)      (whatever your institution measures)
            │                              │
            └───────────── ∩ ──────────────┘
                           │
                  Common variables only
                           │
        ┌──────────────────┼───────────────────┐
        │ (optional) SHAP feature reduction      │
        └──────────────────┼───────────────────┘
                           │
              Train XGBoost ensemble on the
              synthetic data (those variables)
                           │
              Self-contained classifier package
              you distribute to end users
```

> **Important:** at build time only the **column names** of your file are read — your
> row values are ignored. The values matter later, when end users feed *real*
> measurements into the finished classifier (see [Data validity & harmonization](#data-validity--harmonization)).

### The classification task

The target is the `clustername` column — **four classes**, named after heart-failure
phenotypes (these are data labels, not diagnostic categories the tool is validated to assign):

| Class | Label meaning |
|-------|---------------|
| `healthier` | The lower-risk / control-like class (may include healthier controls) |
| `rEF` | Named after heart failure with **reduced** ejection fraction (HFrEF) |
| `pEF1` | Named after HFpEF subtype 1 (preserved ejection fraction) |
| `pEF2` | Named after HFpEF subtype 2 (preserved ejection fraction) |

> Do not confuse these four classes with the coarser `group` metadata
> (`noHF` / `HFpEF` / `HFrEF`), which is only used to improve missing-value
> imputation when end users run the finished classifier.

### The synthetic reference data

`data/synthetics_per_clustername.xlsx` is the canonical reference. It contains
**759 synthetic subjects × 557 variables**, spanning several modalities:

| Modality | Variables |
|----------|-----------|
| OLINK proteomics | 479 |
| Clinical | 31 |
| Laboratory | 22 |
| Echocardiography | 16 |
| CT imaging | 9 |

All rows are synthetic (SMOTE-generated) — there are no real subjects and no missing
values in the reference. This file defines the **exact variable names** your data must
match, and is the single source of truth for the format described below.

---

## Installation

```bash
# Option A — conda (creates an environment named "pacific")
conda env create -f environment.yml
conda activate pacific

# Option B — pip (Python >= 3.11)
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`,
`matplotlib`, `seaborn`, `tqdm`, `openpyxl` (and `pytest` for the test suite).

---

## Quick start

```bash
# 1. (Optional) Simulate an external dataset to try the pipeline end-to-end.
#    This keeps ~10 random variables from the synthetic reference.
python generate_test_data.py --output_file test_data/my_external_data.xlsx

# 2. Build a classifier package from a variable set.
python create_exportable_package.py \
    --external_data test_data/my_external_data.xlsx \
    --output_dir output/my_package \
    --package_name MyInstitution \
    --n_models 50
```

The finished, distributable classifier lands in
`output/my_package/step3_package/MyInstitution/`.

---

## Preparing your data

Your file must be an Excel workbook (`.xlsx`). Only **one sheet is required**.

### Required sheet: `training_datas`

- **First column** = `Participant` (subject identifier; becomes the row index, read as text).
- **Remaining columns** = your variables. Column **names must match the Pacific
  reference exactly** (case-sensitive) — this is what the intersection matches on.
- Values are numeric. Missing values may be left as `NaN`.

| Participant | NPPB | Biplane.LVEF | sex#male | NYHA.classification#class I | … |
|-------------|------|--------------|----------|-----------------------------|---|
| subj_001    | 2.45 | 58.0         | 1.0      | 0.0                         | … |
| subj_002    | 3.12 | 41.0         | 0.0      | 1.0                         | … |

### Naming conventions (must match exactly)

- **Compound names** use a dot separator: `NYHA.classification`, `Biplane.LVEF`
  (not `NYHA classification`).
- **Categorical variables** are one-hot encoded with a `#` separator:
  `variable#value` (e.g. `sex#male`, `NYHA.classification#class I`,
  `CKD.categories#A2`). For one subject, exactly one of a variable's one-hot columns
  is `1.0`; if the value is missing, **all** of that variable's one-hot columns are `NaN`.

A single mismatch (wrong case, a space instead of a `.`, a missing `#`) silently drops
that variable from the intersection. After a build, always check
`variable_intersection_report.txt` to confirm which variables were actually used.

### Optional sheets (auto-generated when absent)

You normally only need `training_datas`. If the following sheets are missing, the
pipeline generates sensible defaults; provide them only if you want finer control:

| Sheet | Purpose | Layout if you supply it |
|-------|---------|--------------------------|
| `objective_datas` | Metadata / labels (not required for building) | `Participant` index + columns such as `group`, `clustername` |
| `reverse_dict` | Variable → modality map | Variable names in the **index**, a column literally named `Modality` |
| `categorical_features` | Which variables are one-hot | Variable names in the **index** |
| `variable_hierarchy` | Variable importance ranking | Variable names in the **index**, columns `ranking`, `modality` |

> If `categorical_features` is absent, any column whose name contains `#` is treated
> as categorical automatically.

---

## Building the classifier

```bash
python create_exportable_package.py \
    --external_data /path/to/your_data.xlsx \
    --output_dir /path/to/output \
    --package_name YourInstitutionName \
    [--skip_reduction] \
    [--n_models 50]
```

### Command-line options

| Option | Description | Default |
|--------|-------------|---------|
| `--external_data`, `-e` | Path to your data file (only its column names are used) | **Required** |
| `--output_dir`, `-o` | Output directory | **Required** |
| `--package_name`, `-n` | Name for the generated package | **Required** |
| `--reference_synthetics`, `-r` | Path to the reference synthetic data | `data/synthetics_per_clustername.xlsx` |
| `--hyperparams_file`, `-p` | Custom hyperparameters Excel file | Built-in defaults |
| `--skip_reduction` | Skip the feature-reduction step | off |
| `--n_models` | Number of models to train (**must be > 3**) | `50` |
| `--reduce_rounds` | Max feature-reduction iterations | `50` |
| `--quiet`, `-q` | Suppress verbose logging | off |

> With no `--hyperparams_file`, a built-in default hyperparameter set is used.
> `data/default_hyperparameters.xlsx` (a table of pre-optimized trials) is **not**
> loaded automatically — pass it with `-p` if you want to use it.

### What the pipeline does

1. **Step 0 — Validate & intersect.** Loads the reference and your file, computes the
   variable intersection, and writes `variable_intersection_report.txt`. Fails if there
   are **no** common variables.
2. **Step 1 — Feature reduction** *(skip with `--skip_reduction`)*. Iteratively removes
   low-importance variables using SHAP values against a random baseline.
3. **Step 2 — Train the ensemble.** Trains `--n_models` XGBoost models on the
   (reduced) synthetic data, then selects the best **ensemble strategy** — `voted`
   (majority vote) or `probas` (averaged probabilities) — and the best number of models
   by **macro-F1**.
4. **Step 3 — Package.** Assembles a self-contained classifier package (code, models,
   reduced synthetic reference, label encoder, docs) under `step3_package/<name>/`.

---

## Output

```
output_dir/
├── pipeline_log.txt                     # Full execution log
├── variable_intersection_report.txt     # Which variables were used / ignored
├── step1_reduction/                      # Only if reduction was not skipped
│   ├── reduce_00/ … reduce_NN/           #   per-round importances.xlsx + reduced_variables.csv
│   └── f1_scores_evolution.csv
├── step2_classifier/
│   ├── models/
│   │   ├── xgb_model_0.json …            #   trained models
│   │   ├── clustername_LabelEncoder.pkl
│   │   └── f1_scores.xlsx                #   per-model and ensemble F1 scores
│   └── confusion_matrix.pdf
└── step3_package/
    └── YourInstitutionName/              # ← the distributable package
        ├── pacific_classifier.py         #   classifier the end user runs
        ├── README.md                     #   END-USER documentation
        ├── LICENSE
        ├── requirements.txt
        └── data/
            ├── synthetic_pacific_data.xlsx   # reference data, reduced to the used variables
            ├── models/*.json
            ├── clustername_LabelEncoder.pkl
            ├── reduced_variables.csv
            ├── best_model_<voted|probas>.txt
            └── pipeline_summary.xlsx
```

The `step3_package/YourInstitutionName/` directory is **fully self-contained** — it has
no dependency on this repository or on the full Pacific data, and is what you distribute
to end users.

### Reading performance

The classifier is evaluated with **macro-F1** (cross-validation on synthetic data).
Check `step2_classifier/models/f1_scores.xlsx` and `pipeline_log.txt`:

| Macro-F1 (synthetic data only) | Fit to the synthetic reference — *not* a measure of real-world validity |
|---------------------------------|------------------------------------------------------------------------|
| ≥ 0.90 | Very high fit to the synthetic reference |
| 0.80 – 0.90 | High fit to the synthetic reference |
| 0.70 – 0.80 | Moderate fit; consider adding variables |
| < 0.70 | **Warning** — too little variable overlap; the model barely fits even the synthetic data |

A macro-F1 below **0.70** triggers an explicit warning in the log; it means the model
barely fits even the synthetic reference. Adding more (matching) variables can improve
that fit — but a higher synthetic-data score is **never** evidence that the package is
fit for real-world or clinical use (see the disclaimer above).

---

## Distributing to end users

Hand the entire `step3_package/YourInstitutionName/` folder to your end users. They
will:

```bash
pip install -r requirements.txt
python pacific_classifier.py -f /path/to/new_subjects.xlsx
```

Everything they need — including their own README and the imputation logic for handling
missing values — is inside the package. As the builder, your responsibility is to make
sure the data those end users will feed in is compatible with the reference (next section).

---

## Data validity & harmonization

Because the classifier learns from synthetic data on **specific scales and units**, the
real measurements eventually fed into it must be **harmonized to the synthetic
reference** — otherwise predictions are not meaningful, even when every variable name
matches. The pipeline does **not** check this for you. Before relying on a package,
verify that the data your institution will classify is consistent with
`data/synthetics_per_clustername.xlsx` on:

- **Order of magnitude** — compare per-variable quantiles / mean / variance against the
  reference; the dynamic range of each variable should be consistent.
- **Units** — correct systematic shifts from unit mismatches
  (e.g. mg/dL vs mmol/L, proportions in `[0,1]` vs percentages in `[0,100]`).
- **Omics preprocessing** — for proteomic (OLINK) and other assays, match the
  preprocessing level (raw intensity vs NPX log-scale vs Z-normalized) to the reference
  distribution.

This is the single most important manual check, and it is on the builder to get right.

---

## Testing

```bash
# Run the full suite (heavy end-to-end tests included)
python -m pytest tests/ -v

# Fast subset only (skip the slow integration tests)
python -m pytest tests/ -v -m "not slow"
```

The suite covers Excel I/O and sheet fallbacks, input validation and the variable
intersection, data preparation and training, ensemble metrics, and the full
end-to-end pipeline.

---

## Troubleshooting

**"No common variables found"**
Your column names don't match the reference. Variable names are case-sensitive; use `.`
inside compound names and `#` for categorical one-hot columns. Inspect
`variable_intersection_report.txt`.

**Low macro-F1 (< 0.70)**
Too few overlapping variables, or systematic differences from the reference. Add more
matching variables and re-check data harmonization.

**`ValueError: not enough models` / ensemble selection fails**
`--n_models` must be **greater than 3** (the ensemble selector reserves the first 3).

**Memory errors**
Lower `--n_models`; ≥ 16 GB RAM is recommended for the default settings.

---

## License

Copyright © 2025 FEALINX — Pacific Project.

- **Code** — [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.html) (AGPL-3.0) or later.
  <img src="images/agplv3-155x51.png" align="right" width="100px"/>
- **Synthetic data & trained models** — [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).
  <img src="images/by-nc-sa.eu_.png" align="right" width="100px"/>

See [`LICENSE`](LICENSE) for the full terms.

## Support

Questions or issues? Please **open an issue** on the project repository:
https://github.com/pboutinaud/pacific_classifier/issues
