Notebooks (run after `make data` / `uv run fm all`; execute with `uv run jupyter nbconvert --to notebook --execute --inplace <nb>`):

1. `01_eda.ipynb` — prevalence by month, `-1` sentinel missingness, chi-square / Mann–Whitney tests,
   what shifts in the held-out months. Ends with the decisions carried into the pipeline.
2. `02_model_explain.ipynb` — SHAP global + per-application explanations for the model card. *(todo)*
