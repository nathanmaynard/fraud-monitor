Notebooks (run after `make data` / `make smoke`):

1. `01_eda.ipynb` — prevalence by month, missingness (-1 sentinels), hypothesis tests
   (fraud rate by device_os / email_is_free / foreign_request — chi-square; income by fraud — Mann-Whitney).
2. `02_model_explain.ipynb` — SHAP global + per-application explanations for the model card.
