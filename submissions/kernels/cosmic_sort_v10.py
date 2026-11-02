"""Cosmic Sort 2026 - v8: fully self-contained, competition-data-only.

Rule compliance (Cosmic Sort rulebook):
  Rule 1 / 8 : reads ONLY train.csv / test.csv / sample_submission.csv from the
               competition. No datasets, no kernel sources, no precomputed arrays.
  Rule 2     : `redshift` is never used as a feature (it is dropped explicitly).
  Rule 4     : open-source packages + pretrained TabICL checkpoint are allowed.
  Rule 8     : free 2xT4 Kaggle notebook, three workers run concurrently so the
               wall clock stays well under 3 hours:
                 CPU  : 6 gradient-boosted members on L2 features
                 GPU0 : TabICL base member (n_estimators=4, full context, disk offload)
                 GPU1 : fine-tuned TabICL member (n_estimators_inference=8, disk offload)
               Fusion = SLSQP geometric blend + Dirichlet + segment-adaptive delta.
               The calibrator is scored CROSS-FITTED (fit on 4 folds, score the
               5th) so the reported OOF is an honest estimate for the hidden 70%.
  Rule 5     : the notebook is the code that produced submission.csv.

Local smoke test: COSMIC_SMOKE=1 (tiny subset, CPU, 1 estimator).
"""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  TOKENIZERS_PARALLELISM='false')
import sys, time, json, gc, subprocess, traceback, importlib.util, importlib.metadata
from pathlib import Path


def ensure_pkgs():
    """Install missing packages BEFORE numpy/torch are imported, so pip can
    safely resolve versions. Never imports tabicl here (keeps fork()+CUDA clean)."""
    need = [p for p in ('lightgbm', 'xgboost', 'catboost')
            if importlib.util.find_spec(p) is None]
    if importlib.util.find_spec('tabicl') is None:
        need.append('tabicl')
    else:
        try:
            if importlib.metadata.version('tabicl') != '2.2.0':
                need.append('tabicl')
        except Exception:
            need.append('tabicl')
    if need:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'tabicl==2.2.0', 'lightgbm', 'xgboost', 'catboost'], check=False)


ensure_pkgs()

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from scipy.optimize import minimize

START = time.monotonic()
SMOKE = os.environ.get('COSMIC_SMOKE', '0') == '1'
OUT = Path(os.environ.get('COSMIC_OUT', '/kaggle/working'))
OUT.mkdir(parents=True, exist_ok=True)
MEM = OUT / 'members'; MEM.mkdir(exist_ok=True)
OFF = OUT / 'offload'; OFF.mkdir(exist_ok=True)
CKPT_DIR = OUT / 'ckpt'; CKPT_DIR.mkdir(exist_ok=True)


def log(*a):
    print(f'[{time.monotonic()-START:7.0f}s]', *a, flush=True)


def status(stage, **kw):
    try:
        (OUT / 'status.json').write_text(json.dumps(
            {'stage': stage, 'seconds': round(time.monotonic() - START, 1), **kw}, indent=2))
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# data + features (stateless, competition data only)
# --------------------------------------------------------------------------- #
def find_data():
    if os.path.isdir('/kaggle/input'):
        for r, _d, f in os.walk('/kaggle/input'):
            if 'train.csv' in f and 'test.csv' in f:
                return r
    return '/home/pixelnotfound/cosmic'


def base_feats(df):
    d = df.copy()
    d['ug'] = d.u - d.g; d['gr'] = d.g - d.r; d['ri'] = d.r - d.i; d['iz'] = d.i - d.z
    d['ur'] = d.u - d.r; d['gz'] = d.g - d.z; d['ui'] = d.u - d.i; d['rz'] = d.r - d.z
    for b in 'ugriz':
        d['q_' + b] = d[b] / d['err_' + b]
    d['log_size'] = np.log1p(d.size_r)
    d['mag_mean'] = d[list('ugriz')].mean(1)
    d['color_spread'] = d[['ug', 'gr', 'ri', 'iz']].std(1)
    d['pm_l'] = np.cos(np.deg2rad(d.gal_b)) * d.gal_l
    d['cos_b'] = np.cos(np.deg2rad(d.gal_b))
    d['sin_b_abs'] = np.abs(np.sin(np.deg2rad(d.gal_b)))
    d['cos_l'] = np.cos(np.deg2rad(d.gal_l))
    d['sin_l'] = np.sin(np.deg2rad(d.gal_l))
    return d


def add_l2(d, Y, n_tr):
    """L2 features validated in v5/v6: stellar-locus residuals, colour S/N,
    log-S/N, colour interactions and size interactions. Fitted only on train
    STAR rows, so it is stateless w.r.t. test."""
    star = np.where(Y == 0)[0]
    for ycol, xcol, nm in [('ug', 'gr', 'res_ug'), ('ri', 'gr', 'res_ri'),
                           ('iz', 'gr', 'res_iz'), ('ur', 'gr', 'res_ur'),
                           ('gr', 'ri', 'res_gr'), ('ug', 'ri', 'res_ugri')]:
        xv = d[xcol].values; yv = d[ycol].values
        cf = np.polyfit(xv[star], yv[star], 2)
        d[nm] = yv - np.polyval(cf, xv)
    for a, b, nm in [('u', 'g', 'sug'), ('g', 'r', 'sgr'), ('r', 'i', 'sri'),
                     ('i', 'z', 'siz'), ('u', 'r', 'sur'), ('g', 'z', 'sgz')]:
        d[nm] = (d[a] - d[b]) / np.sqrt(d['err_' + a] ** 2 + d['err_' + b] ** 2)
    for b in 'ugriz':
        d['lsnr_' + b] = np.log(d[b] / d['err_' + b] + 1e-6)
    d['log_err_u'] = np.log(d.err_u); d['log_err_z'] = np.log(d.err_z)
    d['ug_gr'] = d.ug * d.gr; d['gr_ri'] = d.gr * d.ri; d['ri_iz'] = d.ri * d.iz
    d['ug_minus_gr'] = d.ug - d.gr; d['gr_minus_ri'] = d.gr - d.ri
    d['size_mag'] = d.size_r * (d.mag_mean - 15)
    d['size_over_err'] = d.size_r / (d.err_r + 1e-3)
    d['flux_ratio_ug'] = d.ug / (d.size_r + 1e-3)
    d['size_gr'] = d.size_r * d.gr; d['size_ri'] = d.size_r * d.ri
    return d


def build_features(train, test, Y):
    n_tr = len(train)
    BT = pd.concat([train, test], ignore_index=True)
    dB = base_feats(BT)
    FB = [c for c in dB.columns if c not in ('redshift', 'class', 'object_id')]
    XB = dB.iloc[:n_tr][FB].to_numpy(np.float32)
    XTB = dB.iloc[n_tr:][FB].to_numpy(np.float32)
    dL = add_l2(base_feats(BT), Y, n_tr)
    FL = [c for c in dL.columns if c not in ('redshift', 'class', 'object_id')]
    XL = dL.iloc[:n_tr][FL].to_numpy(np.float32)
    XTL = dL.iloc[n_tr:][FL].to_numpy(np.float32)
    return XB, XTB, XL, XTL, dB, FB, FL


# --------------------------------------------------------------------------- #
# fusion helpers (v4/v5 post-processing, calibrated cross-fitted)
# --------------------------------------------------------------------------- #
def loss(y, p):
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)).mean())


def blend_geo(P, LP, w):
    z = np.tensordot(w, LP, axes=1)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_geo(P, LP, y):
    m = P.shape[0]
    r = minimize(lambda w: loss(y, blend_geo(P, LP, w)), np.ones(m) / m, method='SLSQP',
                 bounds=[(1e-4, 1)] * m,
                 constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},),
                 options={'maxiter': 600, 'ftol': 1e-10})
    w = np.maximum(r.x, 1e-4)
    return w / w.sum()


def dirichlet_cal(p, w, T, b):
    q = np.clip(p, 1e-9, 1) ** (1.0 / T)
    q = q * w * np.exp(b)
    return q / q.sum(1, keepdims=True)


def fit_dirichlet(p, y):
    def f(th):
        w = np.exp(th[0:3]); w = w / w.sum(); T = float(np.exp(th[3]))
        return loss(y, dirichlet_cal(p, w, T, th[4:7]))
    rng = np.random.RandomState(0); best = (f(np.zeros(7)), np.zeros(7))
    for _ in range(3):
        r = minimize(f, np.zeros(7) + rng.randn(7) * 0.05, method='Nelder-Mead',
                     options={'maxiter': 4000, 'fatol': 1e-10})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    th = best[1]; w = np.exp(th[0:3]); w = w / w.sum()
    return w, float(np.exp(th[3])), th[4:7]


def fit_segment(p, y, F, w0, T0, b0):
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6000, random_state=0)
    dt.fit(F, p.argmax(1) != y)
    seg = dt.apply(F); models = {}
    for s in np.unique(seg):
        m = seg == s
        if m.sum() < 8000:
            continue
        def g(th):
            T = float(np.exp(th[0])); pp = np.clip(p[m], 1e-9, 1) ** (1.0 / T)
            pp = pp * w0 * np.exp(th[1:4])
            return loss(y[m], pp / pp.sum(1, keepdims=True))
        r = minimize(g, [np.log(T0), 0, 0, 0], method='Nelder-Mead',
                     options={'maxiter': 800, 'fatol': 1e-9})
        Ts = 0.7 * np.exp(r.x[0]) + 0.3 * T0
        bs = 0.7 * r.x[1:4] + 0.3 * b0
        models[int(s)] = (Ts, bs)
    return dt, models


def apply_segment(p, F, dt, models, w0, T0, b0):
    q = dirichlet_cal(p, w0, T0, b0)
    seg = dt.apply(F)
    for s, (Ts, bs) in models.items():
        m = seg == s
        q[m] = dirichlet_cal(p[m], w0, Ts, bs)
    return q


# --------------------------------------------------------------------------- #
# workers (top-level so fork() can target them)
# --------------------------------------------------------------------------- #
def worker_gbts(XL, XTL, Y, folds, out, smoke=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    NC = 3
    ne = 40 if smoke else 1500
    nj = 2
    es = 5 if smoke else 100

    def reg(name, oof, te):
        np.save(out / f'oof_{name}.npy', oof)
        np.save(out / f'te_{name}.npy', te)
        log('GBT', name, 'OOF', round(log_loss(Y, oof), 5))

    def run_lgb(seed):
        oof = np.zeros((len(XL), NC)); te = np.zeros((len(XTL), NC))
        for k, (tr, va) in enumerate(folds):
            m = lgb.LGBMClassifier(objective='multiclass', num_class=NC, n_estimators=ne,
                                   learning_rate=0.035, num_leaves=80, feature_fraction=0.8,
                                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                                   min_data_in_leaf=40, verbosity=-1, seed=seed, n_jobs=nj)
            m.fit(XL[tr], Y[tr], eval_set=[(XL[va], Y[va])], eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(es, verbose=False)])
            oof[va] = m.predict_proba(XL[va]); te += m.predict_proba(XTL) / len(folds)
        return oof, te

    def run_xgb(seed):
        oof = np.zeros((len(XL), NC)); te = np.zeros((len(XTL), NC))
        for k, (tr, va) in enumerate(folds):
            m = xgb.XGBClassifier(objective='multi:softprob', num_class=NC, n_estimators=ne,
                                  learning_rate=0.035, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=3,
                                  tree_method='hist', device='cpu', early_stopping_rounds=es, verbosity=0,
                                  seed=seed, n_jobs=nj)
            m.fit(XL[tr], Y[tr], eval_set=[(XL[va], Y[va])])
            oof[va] = m.predict_proba(XL[va]); te += m.predict_proba(XTL) / len(folds)
        return oof, te

    def run_cb(seed):
        oof = np.zeros((len(XL), NC)); te = np.zeros((len(XTL), NC))
        for k, (tr, va) in enumerate(folds):
            m = CatBoostClassifier(iterations=ne, learning_rate=0.05, depth=8, l2_leaf_reg=2.0,
                                   loss_function='MultiClass', eval_metric='MultiClass',
                                   random_seed=seed, verbose=False, thread_count=nj,
                                   fold_permutation_block=64)
            m.fit(XL[tr], Y[tr], eval_set=(XL[va], Y[va]), early_stopping_rounds=es)
            oof[va] = m.predict_proba(XL[va]); te += m.predict_proba(XTL) / len(folds)
        return oof, te

    def run_lgbrew(seed):
        sc = StandardScaler().fit(np.vstack([XL, XTL]))
        Za, Zb = sc.transform(XL), sc.transform(XTL)
        Xp = np.vstack([Zb, Za]); yb = np.concatenate([np.ones(len(Zb)), np.zeros(len(Za))])
        lr = LogisticRegression(C=1e-3, max_iter=300).fit(Xp, yb)
        pr = lr.predict_proba(Za)[:, 1].clip(1e-4, 1 - 1e-4)
        w = pr / (1 - pr); w = w / w.mean(); w = np.clip(w, 0.55, 1.7)
        oof = np.zeros((len(XL), NC)); te = np.zeros((len(XTL), NC))
        for k, (tr, va) in enumerate(folds):
            m = lgb.LGBMClassifier(objective='multiclass', num_class=NC, n_estimators=ne,
                                   learning_rate=0.035, num_leaves=80, feature_fraction=0.8,
                                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                                   min_data_in_leaf=40, verbosity=-1, seed=seed, n_jobs=nj)
            m.fit(XL[tr], Y[tr], sample_weight=w[tr], eval_set=[(XL[va], Y[va])],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(es, verbose=False)])
            oof[va] = m.predict_proba(XL[va]); te += m.predict_proba(XTL) / len(folds)
        return oof, te

    for name, fn in [('lgb7', lambda: run_lgb(7)), ('xgb7', lambda: run_xgb(7)),
                     ('cb7', lambda: run_cb(7)), ('lgb1337', lambda: run_lgb(1337)),
                     ('xgb1337', lambda: run_xgb(1337)), ('lgbrew7', lambda: run_lgbrew(7))]:
        try:
            o, t = fn(); reg(name, o, t)
        except Exception as exc:
            log('GBT ERR', name, repr(exc)); traceback.print_exc()


def worker_tabicl(XB, XTB, Y, folds, out, ckpt, smoke=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    from tabicl import TabICLClassifier
    NC = 3
    ne = 1 if smoke else 4
    dev = 'cpu' if smoke else 'cuda'
    off = OFF / 'tabicl'; off.mkdir(exist_ok=True)
    oof = np.zeros((len(XB), NC)); te = np.zeros((len(XTB), NC))
    for k, (tr, va) in enumerate(folds):
        clf = TabICLClassifier(model_path=ckpt, allow_auto_download=False, device=dev,
                               n_estimators=ne, random_state=7, batch_size=8,
                               offload_mode='auto', disk_offload_dir=str(off),
                               kv_cache=False, verbose=not smoke)
        clf.fit(XB[tr], Y[tr]); oof[va] = clf.predict_proba(XB[va])
        log('tabicl fold', k)
        del clf; gc.collect()
        if not smoke:
            torch.cuda.empty_cache()
    clf = TabICLClassifier(model_path=ckpt, allow_auto_download=False, device=dev,
                           n_estimators=ne, random_state=7, batch_size=8,
                           offload_mode='auto', disk_offload_dir=str(off),
                           kv_cache=False, verbose=not smoke)
    clf.fit(XB, Y); te = clf.predict_proba(XTB)
    oof = np.clip(oof, 1e-9, 1); oof /= oof.sum(1, keepdims=True)
    te = np.clip(te, 1e-9, 1); te /= te.sum(1, keepdims=True)
    np.save(out / 'oof_tabicl.npy', oof); np.save(out / 'te_tabicl.npy', te)
    log('tabicl OOF', round(log_loss(Y, oof), 5))


def worker_finetune(XB, XTB, Y, folds, out, ckpt, smoke=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    import torch
    from tabicl import FinetunedTabICLClassifier, TabICLClassifier
    from sklearn.model_selection import StratifiedShuffleSplit
    NC = 3
    dev = 'cpu' if smoke else 'cuda'
    off = OFF / 'finetune'; off.mkdir(exist_ok=True)
    epochs = 1 if smoke else 8
    ne_inf = 1 if smoke else 8
    tl = 20 if smoke else 480
    max_data = 512 if smoke else 4096
    oof = np.zeros((len(XB), NC)); parts = []
    for k, (tr, va) in enumerate(folds):
        a, h = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=7)
                    .split(XB[tr], Y[tr]))
        fit_idx, inner_idx = tr[a], tr[h]
        ft = FinetunedTabICLClassifier(
            model_path=ckpt, allow_auto_download=False, device=dev, random_state=7,
            extra_classifier_kwargs={'batch_size': 2, 'offload_mode': 'auto',
                                     'disk_offload_dir': str(off), 'kv_cache': False},
            epochs=epochs, learning_rate=5e-6, weight_decay=0.01, grad_clip=1.0, amp=True,
            n_estimators_finetune=1 if smoke else 2,
            n_estimators_validation=1 if smoke else 2,
            n_estimators_inference=ne_inf, max_data_size=max_data,
            finetune_ctx_query_ratio=0.2, early_stopping=True, patience=2, min_delta=1e-4,
            time_limit=tl, save_interval=0, freeze_col=True, freeze_row=True,
            freeze_icl=False, eval_metric='log_loss', verbose=False)
        od = out / f'ft_{k}'
        ft.fit(XB[fit_idx], Y[fit_idx], X_val=XB[inner_idx], y_val=Y[inner_idx],
               output_dir=od)
        ck = od / 'best.ckpt'
        clf = TabICLClassifier(model_path=str(ck), allow_auto_download=False, device=dev,
                               n_estimators=ne_inf, random_state=7, batch_size=2,
                               offload_mode='auto', disk_offload_dir=str(off),
                               kv_cache=False, verbose=not smoke)
        clf.fit(XB[tr], Y[tr])
        oof[va] = clf.predict_proba(XB[va])
        parts.append(clf.predict_proba(XTB))
        log('finetuned fold', k)
        del ft, clf; gc.collect()
        if not smoke:
            torch.cuda.empty_cache()
    te = np.mean(parts, axis=0)
    oof = np.clip(oof, 1e-9, 1); oof /= oof.sum(1, keepdims=True)
    te = np.clip(te, 1e-9, 1); te /= te.sum(1, keepdims=True)
    np.save(out / 'oof_finetuned.npy', oof); np.save(out / 'te_finetuned.npy', te)
    log('finetuned OOF', round(log_loss(Y, oof), 5))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def get_checkpoint():
    from huggingface_hub import hf_hub_download
    return hf_hub_download('jingang/TabICL', 'tabicl-classifier-v2-20260212.ckpt',
                           revision='4dcd344ece2c00be9e831fdd35bed57b5ad83e19',
                           cache_dir=str(CKPT_DIR))


def fuse(Y, folds, F, FT, ss, CLASSES, n_test, out):
    members = {}; tests = {}
    for f in sorted(MEM.glob('oof_*.npy')):
        name = f.stem[4:]
        tf = MEM / f'te_{name}.npy'
        if not tf.exists():
            continue
        o = np.load(f); t = np.load(tf)
        if o.shape != (len(Y), 3) or t.shape != (n_test, 3):
            log('SKIP', name, 'shape', o.shape, t.shape); continue
        if not (np.isfinite(o).all() and np.isfinite(t).all()):
            log('SKIP', name, 'nonfinite'); continue
        members[name] = o; tests[name] = t
    log('members', list(members))
    if len(members) < 2:
        # Fallback so the kernel always emits a valid submission.
        if len(members) == 1:
            pt = np.asarray(list(tests.values())[0], dtype=float)
        else:
            prior = np.bincount(Y, minlength=3) / len(Y)
            pt = np.tile(prior, (n_test, 1))
        pt = np.clip(pt, 1e-9, 1); pt /= pt.sum(1, keepdims=True)
        sub = ss.copy()
        for i, c in enumerate(CLASSES):
            sub[c] = pt[:, i]
        sub = sub[['object_id'] + CLASSES]
        sub.to_csv(out / 'submission.csv', index=False)
        (out / 'report.json').write_text(json.dumps(
            {'members': list(members), 'fallback': True,
             'total_seconds': round(time.monotonic() - START, 1)}, indent=2))
        log('FALLBACK submission', sub.shape)
        return
    names = list(members)
    P = np.stack([members[n] for n in names])
    PT = np.stack([tests[n] for n in names])
    P = np.clip(P, 1e-9, 1); P /= P.sum(2, keepdims=True)
    PT = np.clip(PT, 1e-9, 1); PT /= PT.sum(2, keepdims=True)
    LP = np.log(P); LPT = np.log(PT)

    q = np.zeros((len(Y), 3))
    for k, (tr, va) in enumerate(folds):
        w = fit_geo(P[:, tr], LP[:, tr], Y[tr])
        raw_tr = blend_geo(P[:, tr], LP[:, tr], w)
        raw_va = blend_geo(P[:, va], LP[:, va], w)
        w0, T0, b0 = fit_dirichlet(raw_tr, Y[tr])
        dt, models = fit_segment(raw_tr, Y[tr], F[tr], w0, T0, b0)
        q[va] = apply_segment(raw_va, F[va], dt, models, w0, T0, b0)
    cf = loss(Y, q)
    log('cross-fitted OOF', round(cf, 5))

    w = fit_geo(P, LP, Y)
    raw = blend_geo(P, LP, w)
    w0, T0, b0 = fit_dirichlet(raw, Y)
    dt, models = fit_segment(raw, Y, F, w0, T0, b0)
    rawT = blend_geo(PT, LPT, w)
    pt = apply_segment(rawT, FT, dt, models, w0, T0, b0)
    pt = np.clip(pt, 1e-9, 1); pt /= pt.sum(1, keepdims=True)
    sub = ss.copy()
    for i, c in enumerate(CLASSES):
        sub[c] = pt[:, i]
    sub = sub[['object_id'] + CLASSES]
    sub.to_csv(out / 'submission.csv', index=False)
    log('wrote submission', sub.shape)
    report = {
        'members': names,
        'member_oof': {n: round(float(loss(Y, members[n])), 6) for n in names},
        'cross_fitted_oof': round(cf, 6),
        'full_oof': round(float(loss(Y, apply_segment(raw, F, dt, models, w0, T0, b0))), 6),
        'weights': {n: round(float(x), 5) for n, x in zip(names, w)},
        'total_seconds': round(time.monotonic() - START, 1),
    }
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    log('REPORT', json.dumps(report))


def main():
    d = find_data()
    log('data dir', d)
    train = pd.read_csv(f'{d}/train.csv'); test = pd.read_csv(f'{d}/test.csv')
    ss = pd.read_csv(f'{d}/sample_submission.csv')
    CLASSES = list(ss.columns[1:])
    assert CLASSES == ['STAR', 'GALAXY', 'QSO'], CLASSES
    ss = ss.set_index('object_id').reindex(test['object_id']).reset_index()
    Y = train['class'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    if SMOKE:
        train = train.groupby('class', group_keys=False).head(2000).reset_index(drop=True)
        test = test.head(1000).reset_index(drop=True)
        ss = ss.set_index('object_id').reindex(test['object_id']).reset_index()
        Y = train['class'].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    log('rows', len(train), 'test', len(test))
    status('data_loaded', rows=len(train), test=len(test))

    XB, XTB, XL, XTL, dB, FB, FL = build_features(train, test, Y)
    n_tr = len(train)
    seg_cols = ['log_size', 'ur', 'iz', 'sin_b_abs', 'err_u', 'gal_b']
    F = dB.iloc[:n_tr][seg_cols].to_numpy(np.float32)
    FT = dB.iloc[n_tr:][seg_cols].to_numpy(np.float32)
    n_folds = 2 if SMOKE else 5
    folds = list(StratifiedKFold(n_folds, shuffle=True, random_state=7).split(XB, Y))
    log('features base', XB.shape, 'L2', XL.shape, 'folds', n_folds)
    status('features_built', base=list(XB.shape), l2=list(XL.shape), folds=n_folds)

    ckpt = get_checkpoint()
    log('checkpoint', ckpt)
    status('checkpoint_ready')

    import multiprocessing as mp
    ctx = mp.get_context('fork')
    specs = [
        (worker_gbts, (XL, XTL, Y, folds, MEM, SMOKE), 600 if SMOKE else 9000, 'gbts'),
        (worker_tabicl, (XB, XTB, Y, folds, MEM, ckpt, SMOKE), 600 if SMOKE else 9000, 'tabicl'),
        (worker_finetune, (XB, XTB, Y, folds, MEM, ckpt, SMOKE), 900 if SMOKE else 10000, 'finetune'),
    ]
    procs = []
    for target, args, timeout, name in specs:
        p = ctx.Process(target=target, args=args, name=name)
        p.start(); procs.append((p, timeout, name))
    deadline = time.monotonic() + (400 if SMOKE else 10200)
    for p, _timeout, name in procs:
        p.join(max(0.0, deadline - time.monotonic()))
        if p.is_alive():
            log('TIMEOUT', name); p.terminate(); p.join(10)
        else:
            log('DONE', name, 'exit', p.exitcode)
        status('worker_done', name=name, exit=p.exitcode, alive=p.is_alive())
    status('fusing', members=[f.stem[4:] for f in MEM.glob('oof_*.npy')])
    fuse(Y, folds, F, FT, ss, CLASSES, len(test), OUT)
    status('done')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        (OUT / 'blocker.json').write_text(json.dumps(
            {'error': str(exc), 'traceback': traceback.format_exc(),
             'elapsed': time.monotonic() - START}, indent=2))
        raise
