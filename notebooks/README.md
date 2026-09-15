Notebooks — run after `uv run fm all`; execute with `uv run jupyter nbconvert --to notebook --execute --inplace <nb>`.

1. `01_eda.ipynb` — prevalence by month, `-1` sentinel missingness, chi-square / Mann–Whitney tests,
   what shifts in the held-out months. Ends with the decisions carried into the pipeline.
2. `02_model_explain.ipynb` — SHAP on the deployed model: global drivers, whether the age proxies do age's job
   (they do, partially), and a per-application explanation for reviewers / GDPR Art. 22.
