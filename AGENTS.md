# AGENTS.md — Customer Churn Prediction (Telco)

## Quick start

```bash
uv sync                          # install dependencies
streamlit run src/dashboard.py   # launch dashboard
```

- Python >=3.14, package manager is **uv** (not pip/poetry).
- `src/config.py` calls `st.set_page_config` on import — must be the first Streamlit import.
- Data is `data/dataset.csv` (raw CSV). `@st.cache_data` on `load_data` caches it.
- `.streamlit/config.toml` defines the custom theme.

## Project structure

```
src/dashboard.py   # Streamlit app entrypoint
src/config.py      # page config + constants (MODEL_COMPARISON, costs, thresholds)
src/data.py        # load_data() with type coercion + NaN drop
src/charts.py      # Altair chart functions
notebooks/         # EDA.ipynb, Baseline.ipynb, Model.ipynb (incremental analysis)
data/dataset.csv   # Telco churn dataset
```

## Notebooks

- Run inside `.venv` (kernel: `.venv/bin/python`).
- Download the Kaggle dataset automatically via `kagglehub` — requires a Kaggle account + API credentials (`~/.kaggle/kaggle.json`).

## Notable conventions

- **Category dtype**: object columns are cast to `category` in `load_data()`.
- **TotalCharges**: loaded as string, coerced to numeric (non-numeric rows dropped).
- **SeniorCitizen**: converted from int (0/1) to bool.
- No test suite, no CI, no lint/typecheck commands configured in the repo.
