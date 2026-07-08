# Contributing to ShopStream

## Important Rules

- Do **NOT** push directly to the `main` branch.
- Always work on **your own branch**.
- If you run into a Git issue (merge conflict, rejected push, etc.), don't worry—just reach out to **Sravan** before trying random fixes.

---

# One-Time Setup

Clone the repository:

```bash
git clone https://github.com/sravan-kb/shopstream.git
cd shopstream
```

Create your branch (only once):

```bash
git checkout -b your-name
```

Example:

```bash
git checkout -b abhay
```

Push your branch once:

```bash
git push -u origin your-name
```

---

# Every Time You Work

Switch to your branch:

```bash
git checkout your-name
```

Get the latest changes:

```bash
git pull
```

Save your changes:

```bash
git add .
git commit -m "Describe your changes"
git push
```

Then message **Sravan** to review and merge your work.

---

# Project Structure

| Folder | What to keep here |
|--------|-------------------|
| `dataset/` | Kaggle dataset, cleaned datasets, CSV files |
| `notebooks/` | Jupyter notebooks (`.ipynb`) for experiments, EDA, testing |
| `hive/` | Hive SQL scripts and Hive analysis |
| `pyspark/` | PySpark scripts for data processing and EDA |
| `ml/` | Machine learning models, training scripts, saved models |
| `airflow/` | Airflow DAGs and pipeline files |
| `dashboard/` | Streamlit dashboard code |
| `docs/` | Architecture diagrams, screenshots, reports, presentation files |

---

# Commit Message Examples

Good:

- `Add Hive sales analysis`
- `Implement PySpark EDA`
- `Train Random Forest model`
- `Create Airflow DAG`
- `Update Streamlit dashboard`

Avoid:

- `update`
- `changes`
- `final`
- `test`