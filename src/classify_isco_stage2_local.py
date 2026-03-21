#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISCO Classification from Topic Profiles — Stage 2 (本地版)
=========================================================

用岗位的50维 topic profile 预测 ISCO-08 职业分类，
比较 pre-period (2016-2021) 和 post-period (2022-2025) 的分类准确率。

Memory-efficient: 加载float16，立即subsample，释放全量数据。
"""

import gc
import json
import time
import numpy as np
from pathlib import Path

BASE = Path("/Users/yu/code/code2601/TY")
NPZ_PATH = BASE / "output" / "topic_vectors_with_isco.npz"
OUTPUT_JSON = BASE / "output" / "regression" / "isco_classification_results.json"
N_JOBS = 8
SEED = 42
N_FOLDS = 5
MAX_PER_CLASS = 3000  # 本地版降低一些


def stratified_subsample(X, y, max_per_class, rng):
    classes = np.unique(y)
    indices = []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        indices.append(idx)
    idx = np.concatenate(indices)
    rng.shuffle(idx)
    return X[idx], y[idx]


def load_and_split():
    """Load npz, split pre/post, subsample, release full data."""
    print("[1/5] Loading data (memory-efficient)...")
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    data = np.load(NPZ_PATH, allow_pickle=True)
    X_all = data['X']       # float16, keep as-is initially
    y_all = data['y']
    years = data['years']
    isco_labels = data['isco_labels']
    print(f"  Loaded: {X_all.shape}, {len(isco_labels)} classes [{time.time()-t0:.1f}s]")

    # Split and subsample immediately
    pre_mask = years < 2022
    post_mask = years >= 2022

    print(f"  Pre: {pre_mask.sum():,}, Post: {post_mask.sum():,}")

    # Subsample pre
    X_pre, y_pre = stratified_subsample(
        X_all[pre_mask].astype(np.float32),
        y_all[pre_mask],
        MAX_PER_CLASS, rng
    )
    print(f"  Pre subsampled: {len(X_pre):,}")

    # Subsample post
    X_post, y_post = stratified_subsample(
        X_all[post_mask].astype(np.float32),
        y_all[post_mask],
        MAX_PER_CLASS, rng
    )
    print(f"  Post subsampled: {len(X_post):,}")

    # Per-year data (subsample each year separately)
    unique_years = sorted(np.unique(years))
    year_data = {}
    for yr in unique_years:
        yr_mask = years == yr
        X_yr = X_all[yr_mask].astype(np.float32)
        y_yr = y_all[yr_mask]
        if len(X_yr) > MAX_PER_CLASS * 200:
            X_yr, y_yr = stratified_subsample(X_yr, y_yr, MAX_PER_CLASS, rng)
        year_data[int(yr)] = (X_yr, y_yr)
        print(f"    Year {yr}: {len(X_yr):,}")

    # Release full data
    del X_all, y_all, years, data
    gc.collect()

    return X_pre, y_pre, X_post, y_post, year_data, isco_labels


def cross_val_accuracy(X, y, n_folds, n_jobs, rng):
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    # Filter rare classes
    classes, counts = np.unique(y, return_counts=True)
    valid_classes = set(classes[counts >= n_folds])
    mask = np.array([yi in valid_classes for yi in y])
    X_f, y_f = X[mask], y[mask]
    # Re-encode labels to be contiguous (XGBoost requirement)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_f = le.fit_transform(y_f)
    print(f"    {len(X_f):,} samples, {len(valid_classes)} classes")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(1e9)))
    accuracies = []
    top3_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_f, y_f)):
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            n_jobs=n_jobs, tree_method='hist', random_state=SEED, verbosity=0,
        )
        clf.fit(X_f[train_idx], y_f[train_idx])
        pred = clf.predict(X_f[test_idx])
        acc = float((pred == y_f[test_idx]).mean())
        accuracies.append(acc)

        proba = clf.predict_proba(X_f[test_idx])
        top3 = np.argsort(proba, axis=1)[:, -3:]
        top3_hit = np.array([y_f[test_idx][i] in top3[i] for i in range(len(test_idx))])
        top3_accs.append(float(top3_hit.mean()))

        print(f"      Fold {fold+1}: top1={acc:.4f}, top3={top3_hit.mean():.4f}")
        del clf
        gc.collect()

    return {
        'top1_mean': float(np.mean(accuracies)),
        'top1_std': float(np.std(accuracies)),
        'top3_mean': float(np.mean(top3_accs)),
        'top3_std': float(np.std(top3_accs)),
        'n_samples': int(len(X_f)),
        'n_classes': int(len(valid_classes)),
    }


def cross_period_accuracy(X_train, y_train, X_test, y_test, n_jobs):
    import xgboost as xgb

    common = set(np.unique(y_train)) & set(np.unique(y_test))
    mask_tr = np.array([yi in common for yi in y_train])
    mask_te = np.array([yi in common for yi in y_test])
    X_tr, y_tr = X_train[mask_tr], y_train[mask_tr]
    X_te, y_te = X_test[mask_te], y_test[mask_te]
    # Re-encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(np.concatenate([y_tr, y_te]))
    y_tr = le.transform(y_tr)
    y_te = le.transform(y_te)
    print(f"    Common classes: {len(common)}, train={len(X_tr):,}, test={len(X_te):,}")

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        n_jobs=n_jobs, tree_method='hist', random_state=SEED, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = float((pred == y_te).mean())

    proba = clf.predict_proba(X_te)
    top3 = np.argsort(proba, axis=1)[:, -3:]
    top3_hit = np.array([y_te[i] in top3[i] for i in range(len(y_te))])

    del clf
    gc.collect()

    return {
        'top1': acc,
        'top3': float(top3_hit.mean()),
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'n_classes': int(len(common)),
    }


def per_year_accuracy(year_data, n_jobs, rng):
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    results = {}
    for yr in sorted(year_data.keys()):
        X_yr, y_yr = year_data[yr]

        classes, counts = np.unique(y_yr, return_counts=True)
        valid = set(classes[counts >= 5])
        m = np.array([yi in valid for yi in y_yr])
        X_yr, y_yr = X_yr[m], y_yr[m]

        if len(X_yr) < 100 or len(valid) < 10:
            continue

        # Re-encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_yr = le.fit_transform(y_yr)

        min_count = min(counts[counts >= 5])
        n_f = min(N_FOLDS, min_count)
        if n_f < 2:
            continue

        skf = StratifiedKFold(n_splits=n_f, shuffle=True,
                              random_state=int(rng.integers(1e9)))
        accs = []
        for train_idx, test_idx in skf.split(X_yr, y_yr):
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                n_jobs=n_jobs, tree_method='hist', random_state=SEED, verbosity=0,
            )
            clf.fit(X_yr[train_idx], y_yr[train_idx])
            acc = float((clf.predict(X_yr[test_idx]) == y_yr[test_idx]).mean())
            accs.append(acc)
            del clf

        gc.collect()
        results[yr] = {
            'top1_mean': float(np.mean(accs)),
            'top1_std': float(np.std(accs)),
            'n_samples': int(len(X_yr)),
            'n_classes': int(len(valid)),
        }
        print(f"    {yr}: acc={np.mean(accs):.4f} ±{np.std(accs):.4f}, "
              f"n={len(X_yr):,}, classes={len(valid)}")

    return results


def main():
    rng = np.random.default_rng(SEED)
    X_pre, y_pre, X_post, y_post, year_data, isco_labels = load_and_split()

    results = {'isco_labels': [str(x) for x in isco_labels]}

    # ── 1. Pre-period CV ──
    print("\n[2/5] Pre-period (2016-2021) cross-validation...")
    results['pre_cv'] = cross_val_accuracy(X_pre, y_pre, N_FOLDS, N_JOBS, rng)
    print(f"  → Top-1: {results['pre_cv']['top1_mean']:.4f} ± {results['pre_cv']['top1_std']:.4f}")
    print(f"  → Top-3: {results['pre_cv']['top3_mean']:.4f}")

    # ── 2. Post-period CV ──
    print("\n[3/5] Post-period (2022-2025) cross-validation...")
    results['post_cv'] = cross_val_accuracy(X_post, y_post, N_FOLDS, N_JOBS, rng)
    print(f"  → Top-1: {results['post_cv']['top1_mean']:.4f} ± {results['post_cv']['top1_std']:.4f}")
    print(f"  → Top-3: {results['post_cv']['top3_mean']:.4f}")

    # ── 3. Cross-period ──
    print("\n[4/5] Cross-period tests...")
    print("  Train pre → predict post:")
    results['cross_pre2post'] = cross_period_accuracy(X_pre, y_pre, X_post, y_post, N_JOBS)
    print(f"  → Top-1: {results['cross_pre2post']['top1']:.4f}")

    print("  Train post → predict pre:")
    results['cross_post2pre'] = cross_period_accuracy(X_post, y_post, X_pre, y_pre, N_JOBS)
    print(f"  → Top-1: {results['cross_post2pre']['top1']:.4f}")

    # ── 4. Per-year trend ──
    print("\n[5/5] Per-year accuracy trend...")
    results['per_year'] = per_year_accuracy(year_data, N_JOBS, rng)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ISCO Classification Results")
    print("=" * 60)
    pre_acc = results['pre_cv']['top1_mean']
    post_acc = results['post_cv']['top1_mean']
    delta = post_acc - pre_acc
    print(f"  Pre-period  CV: {pre_acc:.4f} ± {results['pre_cv']['top1_std']:.4f}")
    print(f"  Post-period CV: {post_acc:.4f} ± {results['post_cv']['top1_std']:.4f}")
    print(f"  Δ (post-pre):   {delta:+.4f}")
    print(f"  Cross (pre→post): {results['cross_pre2post']['top1']:.4f}")
    print(f"  Cross (post→pre): {results['cross_post2pre']['top1']:.4f}")
    print()

    if abs(delta) < 0.01:
        print("  结论: 职业分类可预测性基本不变")
        print("  → 职业边界在技能空间上没有明显模糊化")
    elif delta < -0.01:
        print("  结论: 职业分类可预测性下降")
        print("  → 职业边界可能在变模糊")
    else:
        print("  结论: 职业分类可预测性上升")

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
