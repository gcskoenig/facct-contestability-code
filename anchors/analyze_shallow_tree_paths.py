"""
Analyze actual decision paths from shallow tree and compare with GT anchors.

This script extracts the true decision path from the shallow decision tree
(max_depth=3) rather than using alibi's perturbation-based anchors.
This gives simpler, more interpretable model anchors (2-3 rules).
"""

import pickle
import json
import numpy as np
import re
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')


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
    """Extract the actual decision rules used for this instance."""
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    node_indicator = tree.decision_path(instance.reshape(1, -1))
    node_indices = node_indicator.indices

    rules = []
    for node_id in node_indices[:-1]:
        if feature[node_id] != -2:  # -2 means leaf
            feat_name = feature_names[feature[node_id]]
            thresh = threshold[node_id]
            if instance[feature[node_id]] <= thresh:
                rules.append((feat_name, '<=', thresh))
            else:
                rules.append((feat_name, '>', thresh))

    return rules


def apply_model_rules(X, rules, feature_names):
    """Apply model rules to get mask."""
    mask = np.ones(len(X), dtype=bool)
    for feat_name, op, thresh in rules:
        feat_idx = feature_names.index(feat_name)
        if op == '<=':
            mask &= (X[:, feat_idx] <= thresh)
        else:
            mask &= (X[:, feat_idx] > thresh)
    return mask


def parse_gt_rule(rule, feature_names):
    """Parse a GT rule and return (feature_idx, lower, upper)."""
    rule = rule.strip()

    # Range pattern: "3972.00 <= credit_amount <= 4042.00"
    range_match = re.match(r'([\d.]+)\s*<=\s*(\w+)\s*<=\s*([\d.]+)', rule)
    if range_match:
        lower = float(range_match.group(1))
        feat_name = range_match.group(2)
        upper = float(range_match.group(3))
        if feat_name in feature_names:
            return (feature_names.index(feat_name), lower, upper)
        return None

    # Simple >= pattern
    ge_match = re.match(r'(\w+)\s*>=\s*([\d.]+)', rule)
    if ge_match:
        feat_name = ge_match.group(1)
        thresh = float(ge_match.group(2))
        if feat_name in feature_names:
            return (feature_names.index(feat_name), thresh, np.inf)
        return None

    # Simple <= pattern (not range)
    le_match = re.match(r'(\w+)\s*<=\s*([\d.]+)', rule)
    if le_match:
        feat_name = le_match.group(1)
        thresh = float(le_match.group(2))
        if feat_name in feature_names:
            return (feature_names.index(feat_name), -np.inf, thresh)
        return None

    return None


def apply_gt_rules(X, rules, feature_names):
    """Apply GT rules to get mask."""
    mask = np.ones(len(X), dtype=bool)
    for rule in rules:
        parsed = parse_gt_rule(rule, feature_names)
        if parsed:
            feat_idx, lower, upper = parsed
            mask &= (X[:, feat_idx] >= lower) & (X[:, feat_idx] <= upper)
    return mask


def main():
    print("=" * 70)
    print("SHALLOW TREE - DECISION PATH vs GT ANCHOR ANALYSIS")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    model, data, fn_indices = load_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']

    print(f"Model: Decision Tree with max_depth=3")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Tree leaves: {model.get_n_leaves()}")
    print()

    # Load GT anchors from alibi run
    gt_anchor_path = os.path.join(SCRIPT_DIR, 'anchor_results_shallow_tree.json')
    if not os.path.exists(gt_anchor_path):
        print("ERROR: Run run_anchors.py first to generate GT anchors")
        return

    with open(gt_anchor_path, 'r') as f:
        st_results = json.load(f)

    # Get the decision path for first false negative (all share same path)
    test_idx = st_results[0]['test_idx']
    instance = X_test[test_idx]
    rules = get_decision_path_rules(model, instance, feature_names)

    # Get model mask
    model_mask = apply_model_rules(X_train, rules, feature_names)
    model_support = np.sum(model_mask)

    # Model precision
    preds_in_region = model.predict(X_train[model_mask])
    model_precision = np.mean(preds_in_region == 0)

    # True positive rate in model region
    true_pos_rate = np.mean(y_train[model_mask] == 1)
    threshold = model_support * (1 - model_precision)

    print("MODEL ANCHOR (decision path):")
    for feat, op, thresh in rules:
        print(f"  - {feat} {op} {thresh:.2f}")
    print(f"  Support: {model_support}")
    print(f"  Model predicts negative: {model_precision*100:.1f}%")
    print(f"  True positive rate in region: {true_pos_rate*100:.1f}%")
    print(f"  Threshold = {model_support} x (1 - {model_precision:.4f}) = {threshold:.2f}")
    print()

    # Analyze each false negative
    results = []
    best_example = None
    best_overlap_pct = 0

    for r in st_results:
        test_idx = r['test_idx']
        gt_rules = r['ground_truth_anchor']['rules']
        gt_precision = r['ground_truth_anchor']['precision']

        # Apply GT rules
        gt_mask = apply_gt_rules(X_train, gt_rules, feature_names)
        gt_support = np.sum(gt_mask)

        # Overlap
        overlap = np.sum(model_mask & gt_mask)
        overlap_pct = overlap / gt_support * 100 if gt_support > 0 else 0

        # True positive rate in GT region
        gt_true_pos = np.mean(y_train[gt_mask] == 1) if gt_support > 0 else 0

        violation = overlap > threshold

        print(f"Test {test_idx}: GT rules = {gt_rules}")
        print(f"  GT support: {gt_support}, GT precision: {gt_precision:.2f}")
        print(f"  GT true positive rate: {gt_true_pos*100:.1f}%")
        print(f"  Overlap: {overlap} ({overlap_pct:.1f}% of GT)")
        if violation:
            print(f"  >>> VIOLATION: overlap ({overlap}) > threshold ({threshold:.2f})")
        print()

        results.append({
            'test_idx': test_idx,
            'gt_rules': gt_rules,
            'gt_support': gt_support,
            'gt_precision': gt_precision,
            'gt_true_pos': gt_true_pos,
            'overlap': overlap,
            'overlap_pct': overlap_pct,
            'violation': violation
        })

        if overlap_pct > best_overlap_pct and gt_support > 5:
            best_overlap_pct = overlap_pct
            best_example = results[-1]

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    violations = [r for r in results if r['violation']]
    print(f"Total false negatives: {len(results)}")
    print(f"Violations found: {len(violations)}")
    for v in violations:
        print(f"  - Test {v['test_idx']}: overlap={v['overlap']}, threshold={threshold:.2f}")

    if best_example:
        print()
        print("=" * 70)
        print("BEST EXAMPLE FOR PARAGRAPH:")
        print("=" * 70)
        print(f"Test index: {best_example['test_idx']}")
        print(f"Model anchor: checking_status <= 1.50 AND duration > 31.50")
        print(f"  Support: {model_support}, Precision: {model_precision*100:.1f}%")
        print(f"GT anchor: {best_example['gt_rules']}")
        print(f"  Support: {best_example['gt_support']}, Precision: {best_example['gt_precision']*100:.1f}%")
        print(f"Overlap: {best_example['overlap']} samples ({best_example['overlap_pct']:.1f}% of GT anchor)")
        print(f"Threshold: {threshold:.2f}")
        print(f"Violation: overlap ({best_example['overlap']}) > threshold ({threshold:.2f})")

    # Save results
    output = {
        'model_anchor': {
            'rules': [f"{feat} {op} {thresh:.2f}" for feat, op, thresh in rules],
            'support': int(model_support),
            'precision': float(model_precision),
            'true_positive_rate': float(true_pos_rate),
            'threshold': float(threshold)
        },
        'results': results,
        'best_example': best_example
    }

    output_path = os.path.join(SCRIPT_DIR, 'shallow_tree_path_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
