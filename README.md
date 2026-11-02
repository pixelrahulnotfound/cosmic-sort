# Cosmic Sort 2026 — Pipeline & Architecture

Competition: **Cosmic Sort 2026** (Kaggle). Classify SDSS photometric objects into
`STAR`, `GALAXY`, `QSO` from magnitudes, errors, size and sky position. Metric is
**row-averaged multiclass log loss**.

Submitted artifact: **`cosmic_sort_v10.ipynb`** — a single-cell, self-contained
Kaggle notebook that trains everything from the competition CSVs and writes
`submission.csv`. It runs in well under 3 hours on a free 2×Tesla T4 notebook.

---

## 1. Rule compliance

| Rule | How it is satisfied |
|------|---------------------|
| **R1 / R8** — competition data only | Reads only `train.csv`, `test.csv`, `sample_submission.csv` from the mounted competition. No datasets, no kernel sources, no precomputed arrays. |
| **R2** — no redshift | `redshift` is dropped explicitly in `base_feats` / the feature column filters. It is never a feature. |
| **R3** — no inter-team sharing | All models are trained inside the notebook. |
| **R4** — open-source / pretrained | LightGBM, XGBoost, CatBoost, scikit-learn, PyTorch and the open-source **TabICL v2** checkpoint (`jingang/TabICL`, revision `4dcd344…`, `tabicl-classifier-v2-20260212.ckpt`). |
| **R5** — code upload | The submitted notebook is the code that produced `submission.csv`. |
| **R8** — reproduce in <3 h on free hardware | Runs on 2×T4 well inside the limit; see §8. |

The notebook needs **internet** to `pip install tabicl==2.2.0` and to download the
pretrained TabICL checkpoint (allowed as a pretrained model under R4).

---

## 2. Repository layout

```
cosmic_sort_v10.ipynb              # submitted notebook (one code cell)
README.md                          # this file
.gitignore                         # keeps competition data / artifacts out of git
submissions/
  kernels/cosmic_sort_v10.py       # readable source of the notebook
```

---

## 3. High-level architecture

```
train.csv / test.csv / sample_submission.csv
                │
                ▼
        feature engineering
        ├── base_feats  (colours, S/N, size, galactic coords)
        └── add_l2      (stellar-locus residuals, colour S/N, interactions)
                │
     ┌──────────┴─────────────────────────────┐
     │                                        │
     ▼ (L2 features)                          ▼ (base features)
 CPU worker: 6 GBDTs                    two GPU workers, 5-fold CV
 lgb7, xgb7, cb7,                       ├── GPU0: base TabICL v2
 lgb1337, xgb1337, lgbrew7              └── GPU1: fine-tuned TabICL v2
     │                                        │
     └──────────────┬─────────────────────────┘
                    ▼
        member probability matrices
        (train + test predictions)
                    │
                    ▼
        cross-fitted fusion / calibration
        geometric SLSQP blend
          → Dirichlet (temperature + class weights + bias)
          → segment-adaptive temperature/bias (decision-tree segments)
                    │
                    ▼
        submission.csv  (object_id, STAR, GALAXY, QSO)
```

Three processes are forked from the notebook so the CPU GBDTs and the two GPU
models train **concurrently**; the wall clock is the slowest branch (the
fine-tuned TabICL) plus fusion.

---

## 4. Feature engineering

All transforms are row-wise and stateless w.r.t. test, except the L2 stellar-locus
polynomial, which is fit **only on train STAR rows**.

**Base features** (`base_feats`, used by TabICL):
- colours: `ug, gr, ri, iz, ur, gz, ui, rz`
- per-band signal-to-noise: `q_u … q_z`
- `log_size = log1p(size_r)`, `mag_mean`, `color_spread`
- galactic geometry: `pm_l`, `cos_b`, `sin_b_abs`, `cos_l`, `sin_l`

**L2 features** (`add_l2`, used by the GBDTs):
- quadratic stellar-locus residuals: `res_ug, res_ri, res_iz, res_ur, res_gr, res_ugri`
- colour S/N: `sug, sgr, sri, siz, sur, sgz`
- log S/N per band: `lsnr_u … lsnr_z`
- colour interactions: `ug_gr, gr_ri, ri_iz, ug_minus_gr, gr_minus_ri`
- size interactions: `size_mag, size_over_err, flux_ratio_ug, size_gr, size_ri`
- `log_err_u`, `log_err_z`

`redshift` is excluded everywhere.

---

## 5. Model members

### 5.1 Gradient-boosted trees (CPU, L2 features)

Five-fold stratified CV (seed 7), early stopping on the fold validation set, test
prediction averaged over folds:

- **LightGBM** ×2 seeds (`7`, `1337`): 1500 trees, lr 0.035, 80 leaves, feature/bagging 0.8, L2 1.0.
- **XGBoost** ×2 seeds: 1500 trees, lr 0.035, depth 8, `hist`, subsample/colsample 0.8.
- **CatBoost**: 1500 iterations, lr 0.05, depth 8, L2 2.0.
- **lgbrew7**: a logistic-regression density-ratio reweighting of train vs test
  (C = 1e-3, weights clipped 0.55–1.7), then LightGBM with those sample weights.

XGBoost members are optional: on some runtime images XGBoost probes CUDA inside
the CPU worker and raises `cudaErrorNoDevice`. The pipeline catches that, logs it,
and continues with the remaining members, so the run always completes.

### 5.2 Base TabICL (GPU0, base features)

Open-source TabICL v2 classifier, `n_estimators=4`, `batch_size=8`,
`offload_mode="auto"`, `kv_cache=False`. For each of the 5 folds: fit on the
fold-train split and predict the held-out fold; finally fit on all train rows and
predict test. Full-context inference is used, with disk offload to stay inside the
RAM budget.

### 5.3 Fine-tuned TabICL (GPU1, base features)

Per fold, the fold-train split is divided 90 % fit / 10 % validation, and the
pretrained checkpoint is fine-tuned with:

```
epochs=8, lr=5e-6, weight_decay=0.01, grad_clip=1.0, amp=True
n_estimators_finetune=2, n_estimators_validation=2
max_data_size=4096, finetune_ctx_query_ratio=0.2
early_stopping=True, patience=2, min_delta=1e-4, time_limit=480 s
freeze_col=True, freeze_row=True, freeze_icl=False
```

The best checkpoint is then used for full-context inference with
`n_estimators=8`, `batch_size=2`, disk offload, predicting the fold's held-out rows
and the test set. Test predictions are averaged across the 5 fold checkpoints.

---

## 6. Fusion and calibration

The fusion is deliberately **cross-fitted** so the fold-level estimate is honest
(a calibrator fitted on the same rows it scores would be optimistic):

For each fold `k`:
1. Fit geometric blend weights on the other 4 folds (`fit_geo`, SLSQP on the
   probability simplex, minimising row log loss over weighted log-probabilities).
2. Fit Dirichlet calibration on the 4-fold blend (`fit_dirichlet`: class weights,
   temperature, class bias, Nelder–Mead).
3. Fit a depth-3 decision tree on segment features
   `[log_size, ur, iz, sin_b_abs, err_u, gal_b]` and, per leaf, fit a
   temperature/bias correction; final leaf parameters are blended 70/30 with the
   global parameters (`fit_segment`).
4. Apply the fold's calibrator to the held-out fold.

After the cross-fitted pass, the same procedure is refit on **all** train rows and
applied to the test predictions.

Output is reindexed to `test['object_id']` and written with the exact
`sample_submission.csv` columns.

---

## 7. Runtime, memory and environment

- Free Kaggle notebook, **2× Tesla T4 (16 GB each)**, ~29 GB RAM, 4 vCPU.
- Runs comfortably under the 3 h code-check limit.
- The fine-tuned TabICL branch is the critical path; the CPU GBDTs and base
  TabICL run concurrently and finish earlier.
- Disk offload (`offload_mode="auto"`, separate directories per branch) keeps the
  full-context TabICL KV caches inside the RAM/GPU budget.
- Versions observed in the run: `numpy 2.1.3`, `pandas 2.3.3`, `torch 2.11.0+cu128`,
  `scipy 1.16.3`, `scikit-learn 1.6.1`, `tabicl 2.2.0`.

---

## 8. Reproducing

**Kaggle (official path):**
1. Open `cosmic_sort_v10.ipynb` as a Kaggle notebook.
2. Accelerator: **GPU T4 ×2**; enable **Internet**.
3. Add the competition data source `cosmic-sort2026`.
4. Run all. It writes `submission.csv` to `/kaggle/working` and a machine-readable
   `report.json` with the member breakdown and fusion weights.

**Local smoke test (optional, tiny/CPU):**

```bash
COSMIC_SMOKE=1 COSMIC_OUT=/tmp/v10out python cosmic_sort_v10.py
```

This subsamples 2,000 rows per class, uses 2 folds, 1 estimator and CPU, and is
only meant to check that the pipeline runs end to end.

---

## 9. Notes

- **Internet is required** for `pip install` and the Hugging Face checkpoint. If
  the code-check environment disables internet, the TabICL members cannot load;
  the pipeline falls back to the GBDT members only.
- The hidden/private 70 % is handled without changes: every model is trained on
  `train.csv` only, test rows are used solely for inference, and features are
  row-wise (the L2 polynomial is fit on train stars only), so any test set with
  the same schema is valid.
