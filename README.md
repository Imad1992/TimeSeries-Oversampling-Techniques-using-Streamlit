# TimeSeries Oversampling Lab (Streamlit)

A small, educational **Streamlit** app for **time‑series oversampling / data augmentation**.

You upload your own time‑series CSV (time + value), choose augmentation techniques, and the app will:

- Generate **synthetic samples** when your dataset is small (or when you set a higher target size)
- Show **how many rows were added per technique**
- Visualize **original vs synthetic** (overlay plot + per-tech histograms)
- Show **summary statistics** (mean, std, percentiles)
- Export **downloadable CSV files** (original, synthetic per technique, merged datasets)

This project is meant for learning and experimentation (not a production generative model).

---

## Demo (What you will see)

### Main workflow
1. Upload CSV
2. Select:
   - time column
   - value column
3. Choose augmentation techniques
4. Set target total rows
5. Click **Run oversampling**
6. Download generated CSVs

### Visualizations included
- **Overlay chart**: original line + synthetic scatter points (per technique)
- **Distribution comparisons**: histograms of original vs synthetic
- **Stats**: count, mean, std, p05/p50/p95, min, max

---

## Project structure

```text
timeseries-oversampling-lab/
├─�� app.py
├── requirements.txt
└── README.md
```

---

## Input CSV format

Your CSV must have at least two columns:

- a time column (example: `timestamp`)
- a numeric value column (example: `value`)

Example:

```csv
timestamp,value
2026-01-01 00:00:00,10.0
2026-01-01 01:00:00,10.5
2026-01-01 02:00:00,10.2
```

If your columns are named differently, select them inside the app.

### Data cleaning
The app will:
- parse the time column using `pandas.to_datetime(...)`
- convert value to numeric
- drop invalid rows (NaN timestamps or values)
- sort by timestamp

---

## Augmentation techniques included (what they do)

The app includes **classical time-series augmentation** methods:

1. **jitter**  
   Adds small Gaussian noise to the values.

2. **scaling**  
   Randomly scales values up/down (multiplicative factor).

3. **magwarp (magnitude warp)**  
   Applies a smooth random multiplicative curve over time.

4. **timewarp (time warp)**  
   Slightly shifts timestamps using a smooth curve, then interpolates values.

5. **winslice (window slicing)**  
   Uses random windows from the series to generate local synthetic points.

6. **bootstrap (block bootstrap)**  
   Samples contiguous blocks of values to preserve short-range autocorrelation.

> Note: These methods generate synthetic points that resemble the original distribution. They are not deep generative models (GAN/VAE).

---

## How synthetic row counts are decided

You configure:
- **Target total rows**: desired total size after augmentation
- The app computes:
  - `to_add = target_rows - parsed_rows`

Then it splits `to_add` across the selected techniques (equal split).  
Example: `to_add = 1000` and 4 techniques selected → each technique gets ~250 synthetic rows.

If `to_add = 0`, the app will not generate synthetic rows, and it will show a warning.

---

## Downloaded output files

In the app you can download:

- `original.csv`  
  Cleaned original time series with columns:
  - `timestamp`
  - `value`

- `synthetic_<technique>.csv`  
  Synthetic data for each technique.

- `merged_original_plus_<technique>.csv`  
  Original + synthetic for that technique, including a `source` column.

- `merged_all_techniques.csv`  
  Original + all synthetic outputs combined, with `source` labels.

---

## Installation

### 1) Clone the repo
```bash
git clone <your-repo-url>
cd timeseries-oversampling-lab
```

### 2) Create a virtual environment (recommended)

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the app

```bash
streamlit run app.py
```

Streamlit will print a URL, usually:

- `http://localhost:8501`

Open it in your browser.

---

## requirements.txt

```txt
streamlit
numpy
pandas
matplotlib
scipy
```

---

## Troubleshooting

### Error: `File does not exist: app.py`
You ran Streamlit in a folder that doesn’t contain `app.py`.

Fix:
```bash
cd path/to/project-folder
streamlit run app.py
```

### Error: `ModuleNotFoundError: No module named 'numpy'`
You installed packages in a different Python environment.

Fix:
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Or use a virtual environment (recommended).

### Synthetic summary is empty / no synthetic rows generated
That means `to_add = 0` (target rows not bigger than parsed rows).

Fix:
- Increase **Target total rows** in the app
- Click **Run oversampling** again

### Preview panel / plots not visible
Scroll down in the Streamlit page; plots appear after you click **Run oversampling**.

---

## Notes / limitations
- This app is for **experiments** and teaching.
- Oversampling can accidentally cause **data leakage** if used incorrectly in ML pipelines.
- For supervised tasks (classification), oversampling should be done carefully **within training splits** and (often) **per class**.

---

## Roadmap ideas (optional)
- Multivariate time series support (multiple value columns)
- Per-class oversampling (if your dataset includes labels)
- More
