"""
Full Anchor Analysis for All False Negatives

For each false negative across all models:
1. Extract model anchor (from alibi for RF/DT, from decision path for shallow tree)
2. Find GT anchors that include at least one model anchor feature
3. Report overlap and violations
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


def load_all_data():
    """Load all models and data."""
    models = {}

    with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'rb') as f:
        models['random_forest'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'decision_tree.pkl'), 'rb') as f:
        models['decision_tree'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'shallow_tree.pkl'), 'rb') as f:
        models['shallow_tree'] = pickle.load(f)

    with open(os.path.join(MODEL_DIR, 'data_splits.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'false_negatives.json'), 'r') as f:
        fn_indices = json.load(f)

    return models, data, fn_indices


def get_tree_decision_path(tree, instance, feature_names):
    """Extract decision path rules from a decision tree."""
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
                rules.append(f"{feat_name} <= {thresh:.2f}")
            else:
                rules.append(f"{feat_name} > {thresh:.2f}")

    return rules, list(feature_indices)


def parse_alibi_anchor_features(anchor_rules, feature_names):
    """Extract feature indices from alibi anchor rules."""
    feature_indices = set()

    for rule in anchor_rules:
        for i, feat_name in enumerate(feature_names):
            if feat_name in rule:
                feature_indices.add(i)
                break

    return list(feature_indices)


def apply_rules_mask(X, rules, feature_names):
    """Apply text rules to get a mask. Handles various rule formats."""
    mask = np.ones(len(X), dtype=bool)

    for rule in rules:
        rule = rule.strip()
        applied = False

        for feat_idx, feat_name in enumerate(feature_names):
            if feat_name not in rule:
                continue

            # Parse the rule
            if ' <= ' in rule and ' < ' in rule:
                # Range like "12.00 < duration <= 24.00"
                parts = rule.replace(feat_name, '|').split('|')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    if left.endswith('<'):
                        lower = float(left[:-1].strip())
                        if right.startswith('<='):
                            upper = float(right[2:].strip())
                            mask &= (X[:, feat_idx] > lower) & (X[:, feat_idx] <= upper)
                            applied = True
            elif '<=' in rule and '>=' not in rule:
                # Could be "feat <= val" or "val <= feat <= val"
                if rule.startswith(feat_name):
                    # "feat <= val"
                    val = float(rule.split('<=')[1].strip())
                    mask &= (X[:, feat_idx] <= val)
                    applied = True
                else:
                    # Try range "val <= feat <= val"
                    parts = rule.split(feat_name)
                    if len(parts) == 2:
                        left = parts[0].strip().rstrip('<=').strip()
                        right = parts[1].strip().lstrip('<=').strip()
                        if left and right:
                            lower = float(left)
                            upper = float(right)
                            mask &= (X[:, feat_idx] >= lower) & (X[:, feat_idx] <= upper)
                            applied = True
            elif '>=' in rule:
                val = float(rule.split('>=')[1].strip())
                mask &= (X[:, feat_idx] >= val)
                applied = True
            elif '>' in rule and '<' not in rule:
                val = float(rule.split('>')[1].strip())
                mask &= (X[:, feat_idx] > val)
                applied = True
            elif '<' in rule and '>' not in rule:
                val = float(rule.split('<')[1].strip())
                mask &= (X[:, feat_idx] < val)
                applied = True

            if applied:
                break

    return mask


def find_gt_anchors_with_overlap(X, y, feature_names, model_features, model_mask,
                                  delta=0.05, min_samples=5, max_anchors=5):
    """Find GT positive anchors that overlap with model anchor region.

    GT anchors are built using credit_amount + one model feature.
    """
    n_samples = len(X)
    n_features = len(feature_names)
    all_features = list(range(n_features))

    # Get credit_amount index
    credit_amount_idx = feature_names.index('credit_amount') if 'credit_amount' in feature_names else None

    # Focus on positive samples in model region
    target_mask = model_mask & (y == 1)
    if np.sum(target_mask) == 0:
        return []

    X_target = X[target_mask]

    valid_anchors = []
    feat_mins = {f: X[:, f].min() for f in all_features}
    feat_maxs = {f: X[:, f].max() for f in all_features}

    # Build GT anchors using credit_amount + one model feature
    feat_combos = []

    if credit_amount_idx is not None:
        # credit_amount alone
        feat_combos.append((credit_amount_idx,))

        # credit_amount + each model feature
        for model_feat in model_features:
            if model_feat != credit_amount_idx:
                feat_combos.append((credit_amount_idx, model_feat))

    # Also try model features alone as fallback
    for model_feat in model_features:
        feat_combos.append((model_feat,))

    for feat_combo in feat_combos:
        # Generate bounds from target samples
        for bounds in generate_bounds(X, y, X_target, feat_combo):
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

            overlap = np.sum(mask & model_mask)
            if overlap == 0:
                continue

            # Generate rule strings
            rules = []
            for feat_idx, (lower, upper) in bounds.items():
                feat_name = feature_names[feat_idx]
                if lower > feat_mins[feat_idx] and upper < feat_maxs[feat_idx]:
                    rules.append(f"{lower:.2f} <= {feat_name} <= {upper:.2f}")
                elif lower > feat_mins[feat_idx]:
                    rules.append(f"{feat_name} >= {lower:.2f}")
                elif upper < feat_maxs[feat_idx]:
                    rules.append(f"{feat_name} <= {upper:.2f}")

            if not rules:
                continue

            # Identify which model feature is used (if any)
            model_feat_used = None
            for f in feat_combo:
                if f in model_features:
                    model_feat_used = feature_names[f]
                    break
            if model_feat_used is None and credit_amount_idx in feat_combo:
                model_feat_used = "credit_amount"

            valid_anchors.append({
                'rules': rules,
                'precision': precision,
                'support': n_in_box,
                'overlap': overlap,
                'overlap_pct': overlap / n_in_box * 100,
                'model_feature': model_feat_used
            })

    # Remove duplicates and sort by overlap
    seen = set()
    unique_anchors = []
    for a in valid_anchors:
        key = tuple(sorted(a['rules']))
        if key not in seen:
            seen.add(key)
            unique_anchors.append(a)

    unique_anchors.sort(key=lambda x: x['overlap'], reverse=True)
    return unique_anchors[:max_anchors]


def generate_bounds(X, y, X_target, feat_combo):
    """Generate candidate bounds for features."""
    candidates_per_feature = {}

    for feat_idx in feat_combo:
        target_vals = X_target[:, feat_idx]
        all_vals = np.sort(np.unique(X[:, feat_idx]))

        candidates = []

        # Percentile-based bounds on target
        for lower_p in [0, 10, 25, 50]:
            for upper_p in [50, 75, 90, 100]:
                if lower_p < upper_p:
                    lower = np.percentile(target_vals, lower_p)
                    upper = np.percentile(target_vals, upper_p)
                    candidates.append((lower, upper))

        # Value-based bounds
        step = max(1, len(all_vals) // 8)
        for lower in all_vals[::step]:
            for upper in all_vals[::step]:
                if lower <= upper:
                    candidates.append((lower, upper))

        candidates = list(set(candidates))
        candidates_per_feature[feat_idx] = candidates

    feat_list = list(feat_combo)
    for combo in product(*[candidates_per_feature[f] for f in feat_list]):
        yield {feat_list[i]: combo[i] for i in range(len(feat_list))}


def analyze_model(model_name, model, X_train, X_test, y_train, y_test,
                  feature_names, fn_indices, output_lines):
    """Analyze all false negatives for a model."""
    output_lines.append("=" * 80)
    output_lines.append(f" {model_name.upper()} ")
    output_lines.append("=" * 80)
    output_lines.append("")

    is_tree = model_name in ['decision_tree', 'shallow_tree']

    # Load alibi results if available
    alibi_results = {}
    alibi_path = os.path.join(SCRIPT_DIR, f'anchor_results_{model_name}.json')
    if os.path.exists(alibi_path):
        with open(alibi_path, 'r') as f:
            for r in json.load(f):
                alibi_results[r['test_idx']] = r

    results = []

    for i, test_idx in enumerate(fn_indices):
        instance = X_test[test_idx]

        output_lines.append("-" * 80)
        output_lines.append(f"FALSE NEGATIVE #{i+1} (Test Index: {test_idx})")
        output_lines.append("-" * 80)

        # Get model anchor
        if is_tree:
            model_rules, model_features = get_tree_decision_path(model, instance, feature_names)
            model_features = list(set(model_features))
        else:
            # Use alibi results
            if test_idx in alibi_results:
                model_rules = alibi_results[test_idx]['model_anchor']['rules']
                model_features = parse_alibi_anchor_features(model_rules, feature_names)
            else:
                model_rules = ['(no anchor available)']
                model_features = []

        output_lines.append("")
        output_lines.append("MODEL ANCHOR:")
        for rule in model_rules:
            output_lines.append(f"  - {rule}")
        output_lines.append(f"  Features: {[feature_names[f] for f in model_features]}")

        # Get model anchor mask and stats
        model_mask = apply_rules_mask(X_train, model_rules, feature_names)
        model_support = np.sum(model_mask)

        if model_support > 0:
            model_preds = model.predict(X_train[model_mask])
            model_precision = np.mean(model_preds == 0)
            true_pos_rate = np.mean(y_train[model_mask] == 1)
        else:
            model_precision = 0
            true_pos_rate = 0

        threshold = model_support * (1 - model_precision)

        output_lines.append(f"  Support: {model_support}")
        output_lines.append(f"  Model predicts negative: {model_precision*100:.1f}%")
        output_lines.append(f"  True positive rate: {true_pos_rate*100:.1f}%")
        output_lines.append(f"  Error budget (threshold): {threshold:.2f}")

        # Find GT anchors
        if model_features and model_support > 0:
            gt_anchors = find_gt_anchors_with_overlap(
                X_train, y_train, feature_names, model_features, model_mask,
                delta=DELTA, min_samples=MIN_SAMPLES, max_anchors=3
            )
        else:
            gt_anchors = []

        output_lines.append("")
        output_lines.append("GT ANCHORS (constrained to model features):")

        if not gt_anchors:
            output_lines.append("  (no valid GT anchors found with overlap)")
        else:
            for j, gt in enumerate(gt_anchors):
                output_lines.append(f"  [{j+1}] Rules: {', '.join(gt['rules'])}")
                output_lines.append(f"      Precision: {gt['precision']*100:.1f}%, Support: {gt['support']}")
                output_lines.append(f"      Overlap: {gt['overlap']} samples ({gt['overlap_pct']:.1f}%)")
                output_lines.append(f"      Model feature: {gt['model_feature']}")

                if gt['overlap'] > threshold:
                    output_lines.append(f"      >>> VIOLATION: {gt['overlap']} > {threshold:.2f} <<<")

        output_lines.append("")

        # Store result
        results.append({
            'test_idx': test_idx,
            'model_rules': model_rules,
            'model_features': [feature_names[f] for f in model_features],
            'model_support': model_support,
            'model_precision': model_precision,
            'true_pos_rate': true_pos_rate,
            'threshold': threshold,
            'gt_anchors': gt_anchors,
            'has_violation': any(gt['overlap'] > threshold for gt in gt_anchors)
        })

    # Summary for this model
    violations = [r for r in results if r['has_violation']]
    output_lines.append("=" * 80)
    output_lines.append(f"SUMMARY FOR {model_name.upper()}")
    output_lines.append("=" * 80)
    output_lines.append(f"Total false negatives: {len(results)}")
    output_lines.append(f"With violations: {len(violations)}")

    if violations:
        output_lines.append("")
        output_lines.append("Best examples (highest overlap):")
        # Sort by max overlap
        violations.sort(key=lambda x: max((gt['overlap'] for gt in x['gt_anchors']), default=0), reverse=True)
        for v in violations[:3]:
            best_gt = max(v['gt_anchors'], key=lambda x: x['overlap'])
            output_lines.append(f"  Test {v['test_idx']}:")
            output_lines.append(f"    Model: {v['model_rules'][:2]}...")
            output_lines.append(f"    GT: {best_gt['rules']}")
            output_lines.append(f"    Overlap: {best_gt['overlap']}, Threshold: {v['threshold']:.2f}")

    output_lines.append("")
    output_lines.append("")

    return results


def main():
    output_lines = []

    output_lines.append("=" * 80)
    output_lines.append(" FULL ANCHOR ANALYSIS FOR ALL FALSE NEGATIVES")
    output_lines.append("=" * 80)
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"GT anchor precision threshold: {1-DELTA:.0%}")
    output_lines.append(f"Minimum GT anchor support: {MIN_SAMPLES}")
    output_lines.append("")
    output_lines.append("For each false negative, we:")
    output_lines.append("1. Extract the model anchor (decision rules)")
    output_lines.append("2. Find GT anchors that use at least one model feature")
    output_lines.append("3. Check if overlap exceeds error budget (support * (1-precision))")
    output_lines.append("")
    output_lines.append("")

    # Load data
    models, data, fn_indices = load_all_data()
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']

    all_results = {}

    # Analyze each model
    for model_name in ['shallow_tree', 'random_forest', 'decision_tree']:
        if model_name not in fn_indices:
            continue

        results = analyze_model(
            model_name, models[model_name],
            X_train, X_test, y_train, y_test,
            feature_names, fn_indices[model_name],
            output_lines
        )
        all_results[model_name] = results

    # Overall summary
    output_lines.append("=" * 80)
    output_lines.append(" OVERALL SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append("")

    for model_name, results in all_results.items():
        violations = [r for r in results if r['has_violation']]
        output_lines.append(f"{model_name}: {len(violations)}/{len(results)} false negatives have violations")

    output_lines.append("")
    output_lines.append("=" * 80)
    output_lines.append(" BEST EXAMPLES FOR PAPER")
    output_lines.append("=" * 80)

    # Find best examples across all models
    all_violations = []
    for model_name, results in all_results.items():
        for r in results:
            if r['has_violation']:
                best_gt = max(r['gt_anchors'], key=lambda x: x['overlap'])
                all_violations.append({
                    'model': model_name,
                    'test_idx': r['test_idx'],
                    'model_rules': r['model_rules'],
                    'gt_rules': best_gt['rules'],
                    'gt_precision': best_gt['precision'],
                    'overlap': best_gt['overlap'],
                    'threshold': r['threshold'],
                    'model_feature': best_gt['model_feature']
                })

    all_violations.sort(key=lambda x: x['overlap'], reverse=True)

    for i, v in enumerate(all_violations[:5]):
        output_lines.append("")
        output_lines.append(f"[{i+1}] {v['model']} - Test {v['test_idx']}")
        output_lines.append(f"    Model anchor: {v['model_rules'][:3]}...")
        output_lines.append(f"    GT anchor: {v['gt_rules']}")
        output_lines.append(f"    GT precision: {v['gt_precision']*100:.1f}%")
        output_lines.append(f"    Overlap: {v['overlap']} samples")
        output_lines.append(f"    Threshold: {v['threshold']:.2f}")
        output_lines.append(f"    Shared feature: {v['model_feature']}")

    # Save to file
    output_path = os.path.join(SCRIPT_DIR, 'full_anchor_analysis_report.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Report saved to: {output_path}")
    print(f"\nTotal violations found:")
    for model_name, results in all_results.items():
        violations = [r for r in results if r['has_violation']]
        print(f"  {model_name}: {len(violations)}/{len(results)}")


if __name__ == "__main__":
    main()
