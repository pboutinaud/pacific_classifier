# Pacific Classifier

## Overview

This package classifies heart failure subjects into four groups based on the Pacific study classification system:

| Group | Label meaning |
|-------|---------------|
| **healthier** | The lower-risk / control-like class (may include healthier controls) |
| **rEF** | Named after heart failure with reduced ejection fraction (HFrEF) |
| **pEF1** | Named after HFpEF subtype 1 |
| **pEF2** | Named after HFpEF subtype 2 |

The classifier uses an ensemble of XGBoost models trained on a small synthetic dataset patterned after the Pacific study (not on real data).

---

## ⚠️ Disclaimer — research and testing use only (not for clinical use)

> [!WARNING]
> **This classifier is provided for research and testing purposes only. It is NOT a
> medical device and must NOT be used for clinical diagnosis, treatment, patient
> management, or any other medical decision-making.**

This classifier was trained **entirely on a small synthetic cohort**, not on real
patients. It may **overfit to artifacts of that synthetic data** and is not guaranteed
to generalize to real patients: predictions and any performance metrics shown should
**not** be interpreted as evidence of real-world clinical validity. Outputs are not a
diagnosis and must not be relied upon for patient care.

Even for serious research, a classifier intended to produce trustworthy, real-world
results should be **trained on real data** rather than on the synthetic reference.
Parties interested in doing so should contact the project partners through the project
website ([pacific-preserved.fr/partners](https://pacific-preserved.fr/partners/)) to
discuss data access and collaboration.

This software is provided **"as is", without warranty of any kind**, express or implied.
To the fullest extent permitted by law, the Pacific Preserved project, FEALINX, and the
project's partners and contributors accept **no liability** for any use of this classifier
or its outputs. Any clinical, diagnostic, or other use is undertaken **entirely at the
user's own risk and responsibility**, and the user is solely responsible for validating
results and for compliance with all applicable laws, regulations, and approvals.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the installation (classifies synthetic data)
python pacific_classifier.py

# 3. Classify your own data
python pacific_classifier.py -f /path/to/your_data.xlsx
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.11
- See `requirements.txt` for package dependencies

## Preparing Your Data

Your data must be in Excel format (`.xlsx`) with the following structure.

### Required Sheet: `training_datas`

| Column | Description |
|--------|-------------|
| `Participant` | Subject identifier (first column, used as index) |
| Feature columns | Numeric values matching the variables in `synthetic_pacific_data.xlsx` |

**Example:**

| Participant | NPPB | LDLR | smoking#never | smoking#former | ... |
|-------------|------|------|---------------|----------------|-----|
| subject_001 | 2.45 | 1.23 | 1.0 | 0.0 | ... |
| subject_002 | 3.12 | 0.98 | 0.0 | 1.0 | ... |

### Recommended Sheet: `objective_datas`

Providing subject group information improves missing value imputation quality.

| Column | Description |
|--------|-------------|
| `Participant` | Subject identifier (matching `training_datas`) |
| `group` | Subject group: `noHF`, `HFpEF`, `HFrEF`, or `Unknown` |

If this sheet is not provided, all subjects are assigned to the `Unknown` group.

### Data Format Requirements

1. **Variable names must match exactly** - Compare your columns with `data/synthetic_pacific_data.xlsx`
2. **Numeric values**: All features should be numeric (float32)
3. **Missing values**: Use `NaN` for missing data (will be imputed automatically)
4. **Categorical variables**: Must be one-hot encoded with `#` separator
   - Example: `smoking#never`, `smoking#former`, `smoking#current`
   - For a category, exactly one column should be 1.0, others 0.0
   - For missing: all columns for that variable should be NaN

### Reference Data Format

See `data/synthetic_pacific_data.xlsx` for:
- Complete list of expected variables
- Correct column naming conventions
- Example data structure

## Usage

### Basic Classification

```bash
python pacific_classifier.py -f /path/to/your_data.xlsx
```

### Specify Output Directory

```bash
python pacific_classifier.py -f /path/to/your_data.xlsx -o /path/to/output
```

### All Options

```bash
python pacific_classifier.py --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `-f`, `--to_classify_file` | Excel file to classify | Synthetic data |
| `-o`, `--output_dir` | Output directory | `./output` |
| `-p`, `--classifier_package_dir` | Package data directory | `./data` |
| `-e`, `--ensemble` | Force ensemble method (`vote` or `proba`) | Auto-detect |
| `-m`, `--max_iter` | Max iterations for missing value imputation | 5 |
| `-q`, `--quiet` | Suppress verbose output | False |
| `-t`, `--testing` | Testing mode (introduce artificial NaN) | None |

## Output

Results are saved to an Excel file in the output directory.

### Output File: `classified_data.xlsx`

#### Sheet: `training_datas`
- Original feature data (after imputation if missing values were present)

#### Sheet: `objective_datas`
Contains the classification results. Any columns already present in your input
`objective_datas` (e.g. `group`, `subject_id`) are preserved, and the classifier
adds two columns:

| Column | Description |
|--------|-------------|
| `Participant` | Subject identifier (row index) |
| `group` | Original subject group, if you provided one (otherwise `Unknown`) |
| `predicted_clustername` | Predicted class label (`healthier`, `rEF`, `pEF1`, or `pEF2`) |
| `predicted_cluster` | The same prediction, encoded as an integer |

The prediction is produced by the ensemble method selected for this package (majority
vote or averaged probability); the method used is reported in the console output.

### Console Output

The classifier prints:
- Data loading summary
- Missing value imputation progress (if applicable)
- Ensemble method used
- Classification performance metrics, if ground-truth labels are available (these reflect fit to the provided / synthetic data only — not real-world or clinical validity)

**Example output:**

```
Objective data loaded from your_data.xlsx
Groups found in objective data: {'HFpEF': 150, 'HFrEF': 80, 'noHF': 70}
Running in normal mode, no holes punched in the dataset.
Found missing values in datas, filling them grouped by group...
Cluster  HFpEF
100%|████████████████████████████████████████| 5/5 [00:05<00:00,  1.00s/it]
...
Using averaged probability ensemble method for classification.
Saved classified synthetics data to output/classified_data.xlsx
```

## Missing Value Handling

The classifier automatically handles missing values using MissForest imputation:

1. **Grouped imputation**: Subjects are grouped by their `group` label for more accurate imputation
2. **Synthetic context**: Synthetic Pacific data is used to provide imputation context
3. **Iterative refinement**: Multiple iterations improve imputation accuracy

For best results:
- Provide the `group` column in `objective_datas`
- Minimize missing values in your input data when possible

## Interpreting Results

### Classification Labels

> These are descriptions of the synthetic-data class labels, **not** clinical
> interpretations of a subject. See the disclaimer at the top of this file.

| Label | Description |
|-------|-------------|
| `healthier` | The lower-risk / control-like class; may include healthier controls |
| `rEF` | Named after the reduced-ejection-fraction phenotype |
| `pEF1` | Named after preserved-ejection-fraction subtype 1 |
| `pEF2` | Named after preserved-ejection-fraction subtype 2 |

### Prediction

The result for each subject is the predicted class in `predicted_clustername`
(with its integer code in `predicted_cluster`). The classifier computes per-class
probabilities internally to reach this prediction, but only the final predicted class
is written to the output file.

### Ensemble Methods

- **voted**: Majority vote across all models (more robust to outliers)
- **probed**: Average of predicted probabilities (smoother predictions)

The package automatically selects whichever ensemble method scored highest on the synthetic training / validation data (this reflects fit to the synthetic data only, not real-world performance).

## Troubleshooting

### "Missing columns in input data"
Your data is missing variables required by the classifier. Check:
- Column names match `synthetic_pacific_data.xlsx` exactly (case-sensitive)
- All required variables are present

### "No objective data found"
The `objective_datas` sheet is missing or incorrectly formatted. The classifier will still work but imputation quality may be reduced.

### Memory Errors
- Reduce the number of subjects per batch
- Ensure sufficient RAM (8GB+ recommended)

### Slow Imputation
- Reduce `--max_iter` parameter
- Consider pre-imputing data with fewer missing values

## Package Contents

```
PackageName/
├── pacific_classifier.py      # Main classifier script
├── README.md                  # This file
├── LICENSE                    # License information
├── requirements.txt           # Pip requirements
├── data/
│   ├── synthetic_pacific_data.xlsx    # Reference data for format and imputation
│   ├── models/                        # Trained XGBoost models
│   ├── clustername_LabelEncoder.pkl   # Label encoder
│   ├── reduced_variables.csv          # List of variables used
│   ├── best_model_*.txt               # Best ensemble method info
│   └── pipeline_summary.xlsx          # Package creation summary
└── output/                    # Default output directory (created on first run)
```

## Citation

If you use this classifier in your research, please cite:

> [Pacific Project Citation - To be added upon publication]

## License

Copyright (c) 2025 FEALINX - Pacific Project

- **Code**: Licensed under AGPL-3.0
- **Data and Models**: Licensed under CC BY-NC-SA 4.0

## Support

For questions or issues, please open an issue on the project repository:
https://github.com/pboutinaud/pacific_classifier/issues

