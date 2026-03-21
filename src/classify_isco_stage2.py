#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISCO Classification from Topic Profiles — Stage 2 (远程)
=========================================================

用岗位的50维 topic profile 预测 ISCO-08 职业分类，
比较 pre-period (2016-2021) 和 post-period (2022-2025) 的分类准确率。

如果准确率下降 → 职业边界在技能空间上变模糊。
如果准确率不变 → 职业分类仍然有效。

方法:
  1. Pre-period: 5-fold CV accuracy
  2. Post-period: 5-fold CV accuracy
  3. Cross-period: train on pre, predict post (distribution shift test)
  4. Per-year accuracy trend

使用 XGBoost (多分类) + 128 核并行

Input: /root/autodl-tmp/topic_vectors_with_isco.npz
Output: /root/autodl-tmp/isco_classification_results.json
"""

import json
import time
import numpy as np
from pathlib import Path

# ── Config ──
NPZ_PATH = Path("/root/autodl-tmp/topic_vectors_with_isco.npz")
OUTPUT_JSON = Path("/root/autodl-tmp/isco_classification_results.json")
N_JOBS = 64  # 用一半核心，避免内存压力
SEED = 42
N_FOLDS = 5

# 每个ISCO-year最多取多少样本用于训练（控制计算量）
MAX_PER_CLASS = 5000


def load_data():
    print("[1/5] Loading data...")
    t0 = time.time()
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data['X'].astype(np.float32)  # (N, 50) float16 → float32
    y = data['y']                      # (N,) int16 ISCO encoded
    years = data['years']              # (N,) int16
    isco_labels = data['isco_labels']  # (n_classes,) str
    print(f"  Loaded: X={X.shape}, classes={len(isco_labels)}, [{time.time()-t0:.1f}s]")
    print(f"  Years: {years.min()}-{years.max()}")
    return X, y, years, isco_labels


def stratified_subsample(X, y, max_per_class, rng):
    """Subsample to at most max_per_class per class."""
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


def cross_val_accuracy(X, y, n_folds, n_jobs, rng):
    """Stratified K-fold cross-validation with XGBoost."""
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    # Filter classes with < n_folds samples
    classes, counts = np.unique(y, return_counts=True)
    valid_classes = set(classes[counts >= n_folds])
    mask = np.array([yi in valid_classes for yi in y])
    X_f, y_f = X[mask], y[mask]
    print(f"    After filtering rare classes: {len(X_f):,} samples, {len(valid_classes)} classes")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.integers(1e9))
    accuracies = []
    top3_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_f, y_f)):
        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=n_jobs,
            tree_method='hist',
            random_state=SEED,
            verbosity=0,
        )
        clf.fit(X_f[train_idx], y_f[train_idx])

        # Top-1 accuracy
        pred = clf.predict(X_f[test_idx])
        acc = (pred == y_f[test_idx]).mean()
        accuracies.append(acc)

        # Top-3 accuracy
        proba = clf.predict_proba(X_f[test_idx])
        top3 = np.argsort(proba, axis=1)[:, -3:]
        top3_hit = np.array([y_f[test_idx][i] in top3[i] for i in range(len(test_idx))])
        top3_accs.append(top3_hit.mean())

        print(f"      Fold {fold+1}: top1={acc:.4f}, top3={top3_hit.mean():.4f}")

    return {
        'top1_mean': float(np.mean(accuracies)),
        'top1_std': float(np.std(accuracies)),
        'top3_mean': float(np.mean(top3_accs)),
        'top3_std': float(np.std(top3_accs)),
        'n_samples': int(len(X_f)),
        'n_classes': int(len(valid_classes)),
    }


def cross_period_accuracy(X_train, y_train, X_test, y_test, n_jobs):
    """Train on one period, evaluate on another."""
    import xgboost as xgb

    # Only keep classes present in both sets
    common = set(np.unique(y_train)) & set(np.unique(y_test))
    mask_tr = np.array([yi in common for yi in y_train])
    mask_te = np.array([yi in common for yi in y_test])
    X_tr, y_tr = X_train[mask_tr], y_train[mask_tr]
    X_te, y_te = X_test[mask_te], y_test[mask_te]
    print(f"    Common classes: {len(common)}, train={len(X_tr):,}, test={len(X_te):,}")

    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=n_jobs,
        tree_method='hist',
        random_state=SEED,
        verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = (pred == y_te).mean()

    proba = clf.predict_proba(X_te)
    top3 = np.argsort(proba, axis=1)[:, -3:]
    top3_hit = np.array([y_te[i] in top3[i] for i in range(len(y_te))])

    return {
        'top1': float(acc),
        'top3': float(top3_hit.mean()),
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'n_classes': int(len(common)),
    }


def per_year_accuracy(X, y, years, n_jobs, rng):
    """Train within each year separately, report CV accuracy."""
    from sklearn.model_selection import StratifiedKFold
    import xgboost as xgb

    unique_years = sorted(np.unique(years))
    results = {}

    for yr in unique_years:
        mask = years == yr
        X_yr, y_yr = X[mask], y[mask]

        # Subsample if too large
        if len(X_yr) > MAX_PER_CLASS * 300:
            X_yr, y_yr = stratified_subsample(X_yr, y_yr, MAX_PER_CLASS, rng)

        # Filter rare classes
        classes, counts = np.unique(y_yr, return_counts=True)
        valid = set(classes[counts >= 5])
        m = np.array([yi in valid for yi in y_yr])
        X_yr, y_yr = X_yr[m], y_yr[m]

        if len(X_yr) < 100:
            continue

        n_f = min(N_FOLDS, min(np.unique(y_yr, return_counts=True)[1]))
        if n_f < 2:
            continue

        skf = StratifiedKFold(n_splits=min(N_FOLDS, n_f), shuffle=True,
                              random_state=rng.integers(1e9))
        accs = []
        for train_idx, test_idx in skf.split(X_yr, y_yr):
            clf = xgb.XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                n_jobs=n_jobs, tree_method='hist', random_state=SEED, verbosity=0,
            )
            clf.fit(X_yr[train_idx], y_yr[train_idx])
            acc = (clf.predict(X_yr[test_idx]) == y_yr[test_idx]).mean()
            accs.append(acc)

        results[int(yr)] = {
            'top1_mean': float(np.mean(accs)),
            'top1_std': float(np.std(accs)),
            'n_samples': int(len(X_yr)),
            'n_classes': int(len(valid)),
        }
        print(f"    {yr}: acc={np.mean(accs):.4f} (±{np.std(accs):.4f}), "
              f"n={len(X_yr):,}, classes={len(valid)}")

    return results


def main():
    rng = np.random.default_rng(SEED)
    X, y, years, isco_labels = load_data()

    results = {}

    # ── 1. Pre-period CV ──
    print("\n[2/5] Pre-period (2016-2021) cross-validation...")
    pre_mask = years < 2022
    X_pre, y_pre = stratified_subsample(X[pre_mask], y[pre_mask], MAX_PER_CLASS, rng)
    print(f"  Subsampled: {len(X_pre):,}")
    results['pre_cv'] = cross_val_accuracy(X_pre, y_pre, N_FOLDS, N_JOBS, rng)
    print(f"  → Top-1: {results['pre_cv']['top1_mean']:.4f} ± {results['pre_cv']['top1_std']:.4f}")
    print(f"  → Top-3: {results['pre_cv']['top3_mean']:.4f}")

    # ── 2. Post-period CV ──
    print("\n[3/5] Post-period (2022-2025) cross-validation...")
    post_mask = years >= 2022
    X_post, y_post = stratified_subsample(X[post_mask], y[post_mask], MAX_PER_CLASS, rng)
    print(f"  Subsampled: {len(X_post):,}")
    results['post_cv'] = cross_val_accuracy(X_post, y_post, N_FOLDS, N_JOBS, rng)
    print(f"  → Top-1: {results['post_cv']['top1_mean']:.4f} ± {results['post_cv']['top1_std']:.4f}")
    print(f"  → Top-3: {results['post_cv']['top3_mean']:.4f}")

    # ── 3. Cross-period ──
    print("\n[4/5] Cross-period: train pre → predict post...")
    results['cross_pre2post'] = cross_period_accuracy(X_pre, y_pre, X_post, y_post, N_JOBS)
    print(f"  → Top-1: {results['cross_pre2post']['top1']:.4f}")
    print(f"  → Top-3: {results['cross_pre2post']['top3']:.4f}")

    print("\n  Cross-period: train post → predict pre...")
    results['cross_post2pre'] = cross_period_accuracy(X_post, y_post, X_pre, y_pre, N_JOBS)
    print(f"  → Top-1: {results['cross_post2pre']['top1']:.4f}")
    print(f"  → Top-3: {results['cross_post2pre']['top3']:.4f}")

    # ── 4. Per-year trend ──
    print("\n[5/5] Per-year accuracy trend...")
    results['per_year'] = per_year_accuracy(X, y, years, N_JOBS, rng)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  ISCO Classification Results Summary")
    print("=" * 60)
    pre_acc = results['pre_cv']['top1_mean']
    post_acc = results['post_cv']['top1_mean']
    delta = post_acc - pre_acc
    print(f"  Pre-period  CV accuracy: {pre_acc:.4f}")
    print(f"  Post-period CV accuracy: {post_acc:.4f}")
    print(f"  Δ (post - pre):          {delta:+.4f}")
    print()
    if abs(delta) < 0.01:
        print("  结论: 职业分类可预测性基本不变")
        print("  → 职业边界在技能空间上没有明显模糊化")
        print("  → 用职业级AI暴露度作为处理变量是合理的")
    elif delta < -0.01:
        print("  结论: 职业分类可预测性下降")
        print("  → 职业边界可能在变模糊，需要谨慎解读DID结果")
    else:
        print("  结论: 职业分类可预测性上升")
        print("  → 职业内部可能更加同质化")

    # Save
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
