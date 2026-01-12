"""
Analyze shallow tree with GT anchors that must include at least one model anchor feature.

This script:
1. Extracts the model anchor (decision path) from the shallow tree
2. Builds GT anchors that MUST include at least one feature from the model anchor
3. Allows one additional feature to find high-precision positive regions
4. This ensures meaningful overlap comparison
"""

import pickle
import json
import numpy as np
import os
from datetime import datetime
from itertools import combinations, product

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')

DELTA = 0.05  # Require 95% precision for GT anchor
MIN_SAMPLES = 5  # Minimum samples in GT anchor


def load_data():
    """Load model and data."""
    with open(os.path.join(MODEL_DIR, 'shallow_tree.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'data_splits.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'false_negatives.json'), 'r') as f:
        fn_indices = json.load(f)

    return model, data, fn_indices


def get_decision_path_rules(tree, instance, feature_names):
    """Extract the actual decision rules and feature indices used for this instance."""
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    node_indicator = tree.decision_path(instance.reshape(1, -1))
    node_indices = node_indicator.indices

    rules = []
    feature_indices = set()

    for node_id in node_indices[:-1]:
        if feature[node_id] != -2:
            feat_idx = feature[node_id]
            feat_name = feature_names[feat_idx]
            thresh = threshold[node_id]
            feature_indices.add(feat_idx)

            if instance[feat_idx] <= thresh:
                rules.append({
                    'feature': feat_name,
                    'feature_idx': feat_idx,
                    'op': '<=',
                    'threshold': thresh
                })
            else:
                rules.append({
                    'feature': feat_name,
                    'feature_idx': feat_idx,
                    'op': '>',
                    'threshold': thresh
                })

    return rules, list(feature_indices)


def apply_model_rules(X, rules):
    """Apply model rules to get mask."""
    mask = np.ones(len(X), dtype=bool)
    for rule in rules:
        feat_idx = rule['feature_idx']
        thresh = rule['threshold']
        if rule['op'] == '<=':
            mask &= (X[:, feat_idx] <= thresh)
        else:
            mask &= (X[:, feat_idx] > thresh)
    return mask


def find_gt_anchor_with_model_feature(X, y, feature_names, model_features, all_features,
                                       model_mask, delta=0.05, min_samples=5):
    """
    Find GT positive anchors that:
    1. Include at least one feature from the model anchor
    2. Overlap with the model anchor region
    3. Have high positive precision (>= 1-delta)

    Args:
        X: Training data
        y: Training labels
        feature_names: List of feature names
        model_features: Feature indices used by model anchor
        all_features: All feature indices
        model_mask: Boolean mask for model anchor region
        delta: Precision threshold
        min_samples: Minimum samples in anchor

    Returns:
        List of valid GT anchors with overlap info
    """
    n_samples = len(X)
    other_features = [f for f in all_features if f not in model_features]

    valid_anchors = []

    # Get feature ranges
    feat_mins = {f: X[:, f].min() for f in all_features}
    feat_maxs = {f: X[:, f].max() for f in all_features}

    # Strategy: For each model feature, combine with other features
    # to find high-precision positive regions that overlap with model anchor

    for model_feat in model_features:
        # Try model feature alone
        feat_combos = [(model_feat,)]

        # Try model feature + one other feature
        for other_feat in other_features:
            feat_combos.append((model_feat, other_feat))

        for feat_combo in feat_combos:
            # Generate candidate bounds
            for bounds in generate_smart_bounds(X, y, feat_combo, model_mask):
                # Create mask
                mask = np.ones(n_samples, dtype=bool)
                for feat_idx, (lower, upper) in bounds.items():
                    mask &= (X[:, feat_idx] >= lower) & (X[:, feat_idx] <= upper)

                n_in_box = np.sum(mask)
                if n_in_box < min_samples:
                    continue

                n_positive = np.sum(y[mask] == 1)
                precision = n_positive / n_in_box

                if precision < 1 - delta:
                    continue

                # Check overlap with model anchor
                overlap = np.sum(mask & model_mask)
                if overlap == 0:
                    continue

                coverage = n_in_box / n_samples
                overlap_pct = overlap / n_in_box * 100

                # Generate rules
                rules = []
                for feat_idx, (lower, upper) in bounds.items():
                    feat_name = feature_names[feat_idx]
                    feat_min, feat_max = feat_mins[feat_idx], feat_maxs[feat_idx]

                    if lower > feat_min and upper < feat_max:
                        rules.append(f"{lower:.2f} <= {feat_name} <= {upper:.2f}")
                    elif lower > feat_min:
                        rules.append(f"{feat_name} >= {lower:.2f}")
                    elif upper < feat_max:
                        rules.append(f"{feat_name} <= {upper:.2f}")

                valid_anchors.append({
                    'rules': rules,
                    'bounds': {str(feat_idx): (float(l), float(u)) for feat_idx, (l, u) in bounds.items()},
                    'precision': float(precision),
                    'coverage': float(coverage),
                    'n_samples': int(n_in_box),
                    'n_positive': int(n_positive),
                    'overlap': int(overlap),
                    'overlap_pct': float(overlap_pct),
                    'features_used': [feature_names[f] for f in feat_combo],
                    'includes_model_feature': feature_names[model_feat]
                })

    # Sort by overlap (descending)
    valid_anchors.sort(key=lambda x: x['overlap'], reverse=True)

    return valid_anchors


def generate_smart_bounds(X, y, feat_combo, model_mask):
    """Generate candidate bounds focused on positive regions overlapping with model anchor."""
    n_samples = len(X)

    # Focus on samples that are:
    # 1. In the model anchor region (model_mask)
    # 2. Actually positive (y == 1)
    target_mask = model_mask & (y == 1)

    if np.sum(target_mask) == 0:
        return

    X_target = X[target_mask]

    for feat_idx in feat_combo:
        vals = X_target[:, feat_idx]
        all_vals = np.sort(np.unique(X[:, feat_idx]))

    # Generate bounds that cover target samples
    candidates_per_feature = {}

    for feat_idx in feat_combo:
        target_vals = X_target[:, feat_idx]
        all_vals = np.sort(np.unique(X[:, feat_idx]))

        candidates = []

        # Bounds covering different portions of target samples
        percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
        for lower_p in percentiles:
            for upper_p in percentiles:
                if lower_p < upper_p:
                    lower = np.percentile(target_vals, lower_p)
                    upper = np.percentile(target_vals, upper_p)
                    candidates.append((lower, upper))

        # Also try specific value bounds
        for lower in all_vals[::max(1, len(all_vals)//10)]:
            for upper in all_vals[::max(1, len(all_vals)//10)]:
                if lower <= upper:
                    candidates.append((lower, upper))

        candidates = list(set(candidates))
        candidates_per_feature[feat_idx] = candidates

    # Generate combinations
    feat_list = list(feat_combo)
    for combo in product(*[candidates_per_feature[f] for f in feat_list]):
        yield {feat_list[i]: combo[i] for i in range(len(feat_list))}


def main():
    print("=" * 70)
    print("SHALLOW TREE - GT ANCHORS WITH MODEL FEATURE CONSTRAINT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GT anchor precision threshold: {1-DELTA:.0%}")
    print(f"Constraint: GT anchor must include at least one model feature")
    print()

    # Load data
    model, data, fn_indices = load_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
    n_features = len(feature_names)

    print(f"Model: Decision Tree with max_depth=3")
    print(f"Tree depth: {model.get_depth()}")
    print()

    # Get false negative indices for shallow tree
    fn_test_indices = fn_indices.get('shallow_tree', [])
    if not fn_test_indices:
        print("No false negatives found for shallow tree")
        return

    # Get the decision path for first false negative
    first_instance = X_test[fn_test_indices[0]]
    model_rules, model_features = get_decision_path_rules(model, first_instance, feature_names)
    unique_model_features = list(set(model_features))

    print("MODEL ANCHOR (decision path):")
    for rule in model_rules:
        print(f"  - {rule['feature']} {rule['op']} {rule['threshold']:.2f}")
    print(f"  Features: {[feature_names[f] for f in unique_model_features]}")
    print()

    # Get model mask and stats
    model_mask = apply_model_rules(X_train, model_rules)
    model_support = np.sum(model_mask)
    model_preds = model.predict(X_train[model_mask])
    model_precision = np.mean(model_preds == 0)
    true_pos_rate = np.mean(y_train[model_mask] == 1)
    threshold = model_support * (1 - model_precision)

    print(f"  Support: {model_support}")
    print(f"  Model predicts negative: {model_precision*100:.1f}%")
    print(f"  TRUE positive rate in model region: {true_pos_rate*100:.1f}%")
    print(f"  (This means {int(true_pos_rate * model_support)} of {model_support} samples are actually positive!)")
    print(f"  Error budget (threshold): {threshold:.2f}")
    print()

    # Find GT anchors that overlap with model anchor
    print("=" * 70)
    print("SEARCHING FOR GT ANCHORS WITH OVERLAP...")
    print("=" * 70)

    all_features = list(range(n_features))
    gt_anchors = find_gt_anchor_with_model_feature(
        X_train, y_train, feature_names,
        unique_model_features, all_features,
        model_mask, delta=DELTA, min_samples=MIN_SAMPLES
    )

    if not gt_anchors:
        print("\nNo GT anchors found with overlap!")
        print("This means there are no high-precision positive regions")
        print("that overlap with the model's negative anchor region.")
        return

    print(f"\nFound {len(gt_anchors)} GT anchors with overlap")
    print()

    # Show top anchors
    print("TOP GT ANCHORS (by overlap):")
    print("-" * 60)

    for i, anchor in enumerate(gt_anchors[:10]):
        print(f"\n[{i+1}] GT ANCHOR:")
        for rule in anchor['rules']:
            print(f"    - {rule}")
        print(f"    Precision: {anchor['precision']*100:.1f}%")
        print(f"    Support: {anchor['n_samples']}")
        print(f"    Overlap with model anchor: {anchor['overlap']} samples ({anchor['overlap_pct']:.1f}%)")
        print(f"    Model feature used: {anchor['includes_model_feature']}")

        violation = anchor['overlap'] > threshold
        if violation:
            print(f"    >>> VIOLATION: overlap ({anchor['overlap']}) > threshold ({threshold:.2f}) <<<")

    # Best example
    best = gt_anchors[0]
    print("\n" + "=" * 70)
    print("BEST EXAMPLE FOR PARAGRAPH")
    print("=" * 70)
    print(f"\nMODEL ANCHOR:")
    print(f"  Rules: checking_status <= 1.50 AND duration > 31.50")
    print(f"  Support: {model_support}, Precision: {model_precision*100:.1f}%")
    print(f"  Error budget: {threshold:.2f}")
    print(f"\nGT ANCHOR:")
    for rule in best['rules']:
        print(f"  - {rule}")
    print(f"  Precision: {best['precision']*100:.1f}%")
    print(f"  Support: {best['n_samples']}")
    print(f"\nOVERLAP: {best['overlap']} samples")
    print(f"VIOLATION: overlap ({best['overlap']}) > threshold ({threshold:.2f})")

    # Save results
    output = {
        'model_anchor': {
            'rules': [f"{r['feature']} {r['op']} {r['threshold']:.2f}" for r in model_rules],
            'features': [feature_names[f] for f in unique_model_features],
            'support': int(model_support),
            'precision': float(model_precision),
            'true_positive_rate': float(true_pos_rate),
            'threshold': float(threshold)
        },
        'gt_anchors': gt_anchors[:20],
        'best_example': best
    }

    output_path = os.path.join(SCRIPT_DIR, 'shallow_tree_constrained_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
