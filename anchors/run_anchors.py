"""
Anchor Explanations for False Negatives

Goal: Detect violations of ground truth anchors/rules.

For each false negative (predicted Bad, actually Good), we:
1. Compute model's negative anchor (via alibi) - why model predicts Bad
2. Compute ground truth positive anchor - box where training data is actually Good
3. Compare overlap and detect where model contradicts ground truth

This reveals where the model's decision boundary differs from the true data distribution.
Violations occur when the model has high confidence (strong anchor) in regions where
the ground truth shows the opposite.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import gc
import json
import pickle
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# Add parent directory to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')

# Import alibi
try:
    from alibi.explainers import AnchorTabular
except ImportError:
    print("Error: alibi package not found. Install with: pip install alibi")
    sys.exit(1)

RANDOM_SEED = 42
DELTA = 0.05  # Precision threshold: require 95% positive in the box
MIN_SAMPLES_IN_BOX = 5
MAX_GT_RULES = 2  # Maximum number of rules (constrained features) in GT anchor
MODEL_ANCHOR_THRESHOLD = 0.95  # Precision threshold for model anchor


def load_models_and_data():
    """Load pre-trained models and data splits."""
    print("Loading models and data...")

    with open(os.path.join(MODEL_DIR, 'random_forest.pkl'), 'rb') as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'decision_tree.pkl'), 'rb') as f:
        dt_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'data_splits.pkl'), 'rb') as f:
        data = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'false_negatives.json'), 'r') as f:
        fn_indices = json.load(f)

    models = {
        'random_forest': rf_model,
        'decision_tree': dt_model
    }

    # Load shallow tree if it exists
    shallow_tree_path = os.path.join(MODEL_DIR, 'shallow_tree.pkl')
    if os.path.exists(shallow_tree_path):
        with open(shallow_tree_path, 'rb') as f:
            models['shallow_tree'] = pickle.load(f)

    return models, data, fn_indices


def count_in_box(X, y, lower_bounds, upper_bounds):
    """Count samples and positive rate within a box."""
    mask = np.ones(len(X), dtype=bool)
    for i in range(X.shape[1]):
        mask &= (X[:, i] >= lower_bounds[i]) & (X[:, i] <= upper_bounds[i])

    n_in_box = np.sum(mask)
    if n_in_box == 0:
        return 0, 0, 0.0

    n_positive = np.sum(y[mask] == 1)
    precision = n_positive / n_in_box
    return n_in_box, n_positive, precision


def grow_positive_anchor(instance, X_all, y_all, feature_names, delta=0.1, max_rules=2):
    """
    Grow a box around the instance that maximizes coverage while maintaining
    precision >= 1 - delta based on true labels in ALL data (train + test).

    Limits the anchor to at most max_rules constrained features.
    """
    n_features = len(instance)

    # Find positive samples in all data
    positive_mask = y_all == 1
    X_positive = X_all[positive_mask]

    if len(X_positive) == 0:
        return {
            'lower_bounds': instance.copy(),
            'upper_bounds': instance.copy(),
            'rules': ['(no positive samples in data)'],
            'precision': 0.0,
            'coverage': 0.0,
            'n_samples': 0,
            'n_positive': 0
        }

    # Get data ranges for each feature
    feat_mins = np.array([X_all[:, i].min() for i in range(n_features)])
    feat_maxs = np.array([X_all[:, i].max() for i in range(n_features)])

    # Try different combinations of features to constrain (at most max_rules)
    from itertools import combinations

    best_anchor = None
    best_coverage = 0

    # Try all combinations of 1 to max_rules features
    for n_constrained in range(1, max_rules + 1):
        for feat_combo in combinations(range(n_features), n_constrained):
            # Start with full range (no constraints)
            lower_bounds = feat_mins.copy()
            upper_bounds = feat_maxs.copy()

            # For selected features, constrain around instance's nearest positive neighbors
            k = min(10, len(X_positive))
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X_positive)
            distances, indices = nn.kneighbors(instance.reshape(1, -1))
            nearest_positives = X_positive[indices[0]]

            # Set initial bounds only for selected features
            for feat_idx in feat_combo:
                all_vals = np.concatenate([[instance[feat_idx]], nearest_positives[:, feat_idx]])
                lower_bounds[feat_idx] = np.min(all_vals)
                upper_bounds[feat_idx] = np.max(all_vals)

            # Check precision
            n_in_box, n_pos, precision = count_in_box(X_all, y_all, lower_bounds, upper_bounds)

            # If precision not met, shrink by using fewer neighbors
            if precision < 1 - delta:
                for shrink_k in range(k-1, 0, -1):
                    nearest_subset = X_positive[indices[0][:shrink_k]]
                    for feat_idx in feat_combo:
                        all_vals = np.concatenate([[instance[feat_idx]], nearest_subset[:, feat_idx]])
                        lower_bounds[feat_idx] = np.min(all_vals)
                        upper_bounds[feat_idx] = np.max(all_vals)
                    # Reset non-selected features to full range
                    for feat_idx in range(n_features):
                        if feat_idx not in feat_combo:
                            lower_bounds[feat_idx] = feat_mins[feat_idx]
                            upper_bounds[feat_idx] = feat_maxs[feat_idx]
                    n_in_box, n_pos, precision = count_in_box(X_all, y_all, lower_bounds, upper_bounds)
                    if precision >= 1 - delta:
                        break

            # Try to expand bounds for selected features only
            feature_values = [np.unique(X_all[:, i]) for i in range(n_features)]
            improved = True
            max_iterations = 50
            iteration = 0

            while improved and iteration < max_iterations:
                improved = False
                iteration += 1

                for feat_idx in feat_combo:  # Only expand selected features
                    feat_vals = feature_values[feat_idx]

                    # Try expanding lower bound
                    lower_candidates = feat_vals[feat_vals < lower_bounds[feat_idx]]
                    for new_lower in reversed(lower_candidates):
                        test_lower = lower_bounds.copy()
                        test_lower[feat_idx] = new_lower
                        n_in, n_pos, prec = count_in_box(X_all, y_all, test_lower, upper_bounds)
                        if n_in >= MIN_SAMPLES_IN_BOX and prec >= 1 - delta:
                            lower_bounds[feat_idx] = new_lower
                            improved = True
                            break

                    # Try expanding upper bound
                    upper_candidates = feat_vals[feat_vals > upper_bounds[feat_idx]]
                    for new_upper in upper_candidates:
                        test_upper = upper_bounds.copy()
                        test_upper[feat_idx] = new_upper
                        n_in, n_pos, prec = count_in_box(X_all, y_all, lower_bounds, test_upper)
                        if n_in >= MIN_SAMPLES_IN_BOX and prec >= 1 - delta:
                            upper_bounds[feat_idx] = new_upper
                            improved = True
                            break

            # Check final precision and coverage
            n_in_box, n_pos, precision = count_in_box(X_all, y_all, lower_bounds, upper_bounds)
            coverage = n_in_box / len(X_all)

            # Keep best anchor (highest coverage with valid precision)
            if precision >= 1 - delta and n_in_box >= MIN_SAMPLES_IN_BOX:
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_anchor = {
                        'lower_bounds': lower_bounds.copy(),
                        'upper_bounds': upper_bounds.copy(),
                        'feat_combo': feat_combo,
                        'n_samples': n_in_box,
                        'n_positive': n_pos,
                        'precision': precision,
                        'coverage': coverage
                    }

    # If no valid anchor found, return empty
    if best_anchor is None:
        return {
            'lower_bounds': instance.copy(),
            'upper_bounds': instance.copy(),
            'rules': ['(no valid anchor found)'],
            'precision': 0.0,
            'coverage': 0.0,
            'n_samples': 0,
            'n_positive': 0
        }

    # Generate rules from best anchor
    lower_bounds = best_anchor['lower_bounds']
    upper_bounds = best_anchor['upper_bounds']
    rules = []
    for i in best_anchor['feat_combo']:
        feat = feature_names[i]
        if lower_bounds[i] > feat_mins[i] and upper_bounds[i] < feat_maxs[i]:
            rules.append(f"{lower_bounds[i]:.2f} <= {feat} <= {upper_bounds[i]:.2f}")
        elif lower_bounds[i] > feat_mins[i]:
            rules.append(f"{feat} >= {lower_bounds[i]:.2f}")
        elif upper_bounds[i] < feat_maxs[i]:
            rules.append(f"{feat} <= {upper_bounds[i]:.2f}")

    n_in_box = best_anchor['n_samples']
    n_positive = best_anchor['n_positive']
    precision = best_anchor['precision']
    coverage = best_anchor['coverage']

    return {
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'rules': rules if rules else ['(entire feature space - no constraints needed)'],
        'precision': precision,
        'coverage': coverage,
        'n_samples': n_in_box,
        'n_positive': n_positive
    }


def parse_alibi_anchor_to_bounds(anchor_rules, instance, feature_names, X_train):
    """Parse alibi anchor rules into lower/upper bounds."""
    n_features = len(feature_names)
    # Use ±inf to represent "no constraint" (semantically cleaner)
    lower_bounds = np.full(n_features, -np.inf)
    upper_bounds = np.full(n_features, +np.inf)

    for rule in anchor_rules:
        rule = rule.strip()

        feat_idx = None
        feat_name = None
        for i, fn in enumerate(feature_names):
            if fn in rule:
                feat_idx = i
                feat_name = fn
                break

        if feat_idx is None:
            continue

        rule_clean = rule.replace(feat_name, "X")

        try:
            # Handle compound rules like "1.0 < X <= 2.0" first
            if '<' in rule_clean and '<=' in rule_clean:
                parts = rule_clean.split('<=')
                if len(parts) == 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    # Set upper bound from right side
                    if 'X' not in right:
                        upper_bounds[feat_idx] = min(upper_bounds[feat_idx], float(right))
                    # Extract lower bound from left side (e.g., "1.0 < X" -> 1.0)
                    if '<' in left and 'X' in left:
                        left_val = left.replace('<', '').replace('X', '').strip()
                        if left_val:
                            lower_bounds[feat_idx] = max(lower_bounds[feat_idx], float(left_val))
            elif '<=' in rule_clean:
                parts = rule_clean.split('<=')
                if 'X' in parts[0]:
                    upper_bounds[feat_idx] = min(upper_bounds[feat_idx], float(parts[1].strip()))
                else:
                    lower_bounds[feat_idx] = max(lower_bounds[feat_idx], float(parts[0].strip()))
            elif '>' in rule_clean:
                parts = rule_clean.split('>')
                if 'X' in parts[0]:
                    lower_bounds[feat_idx] = max(lower_bounds[feat_idx], float(parts[1].strip()))
                else:
                    upper_bounds[feat_idx] = min(upper_bounds[feat_idx], float(parts[0].strip()))
            elif '<' in rule_clean:
                parts = rule_clean.split('<')
                if 'X' in parts[0]:
                    upper_bounds[feat_idx] = min(upper_bounds[feat_idx], float(parts[1].strip()))
                else:
                    lower_bounds[feat_idx] = max(lower_bounds[feat_idx], float(parts[0].strip()))
        except (ValueError, IndexError):
            continue

    return lower_bounds, upper_bounds


def compute_overlap(bounds1_lower, bounds1_upper, bounds2_lower, bounds2_upper, X_train, y_train, model):
    """Compute overlap between two boxes and analyze."""
    mask1 = np.ones(len(X_train), dtype=bool)
    for i in range(X_train.shape[1]):
        mask1 &= (X_train[:, i] >= bounds1_lower[i]) & (X_train[:, i] <= bounds1_upper[i])

    mask2 = np.ones(len(X_train), dtype=bool)
    for i in range(X_train.shape[1]):
        mask2 &= (X_train[:, i] >= bounds2_lower[i]) & (X_train[:, i] <= bounds2_upper[i])

    overlap_mask = mask1 & mask2

    n_box1 = np.sum(mask1)
    n_box2 = np.sum(mask2)
    n_overlap = np.sum(overlap_mask)

    # Analyze overlap region
    if n_overlap > 0:
        overlap_true_positive_rate = np.mean(y_train[overlap_mask])
        overlap_pred_positive_rate = np.mean(model.predict(X_train[overlap_mask]))
    else:
        overlap_true_positive_rate = 0.0
        overlap_pred_positive_rate = 0.0

    return {
        'n_in_positive_anchor': n_box1,
        'n_in_negative_anchor': n_box2,
        'n_overlap': n_overlap,
        'overlap_fraction_of_positive': n_overlap / n_box1 if n_box1 > 0 else 0,
        'overlap_fraction_of_negative': n_overlap / n_box2 if n_box2 > 0 else 0,
        'overlap_true_positive_rate': overlap_true_positive_rate,
        'overlap_pred_positive_rate': overlap_pred_positive_rate
    }


def run_anchor_analysis(model, model_name, X_train, X_test, y_train, y_test,
                        feature_names, false_negative_indices, output_dir):
    """Run anchor analysis for a single model on all false negatives."""

    print(f"\n{'='*70}")
    print(f"ANCHOR ANALYSIS FOR {model_name.upper()}")
    print(f"{'='*70}")

    # Combine train and test data for GT anchor computation (uses true labels only)
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    # Setup alibi explainer
    print("Setting up alibi explainer...")
    explainer = AnchorTabular(
        predictor=lambda x: model.predict(x),
        feature_names=feature_names,
        categorical_names={},
        seed=RANDOM_SEED
    )
    explainer.fit(X_train, disc_perc=(25, 50, 75))

    results = []
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"\nAnalyzing {len(false_negative_indices)} false negatives...")

    for i, test_idx in enumerate(false_negative_indices):
        instance = X_test[test_idx]

        log(f"\n{'─'*70}")
        log(f"FALSE NEGATIVE #{i+1} (Test Index: {test_idx})")
        log(f"{'─'*70}")

        # Get model's negative anchor
        log("\n[1] MODEL'S NEGATIVE ANCHOR:")
        model_explanation = explainer.explain(instance.reshape(1, -1), threshold=MODEL_ANCHOR_THRESHOLD)
        model_anchor_rules = model_explanation.anchor if model_explanation.anchor else ['(no anchor found)']
        model_precision = model_explanation.precision
        model_coverage = model_explanation.coverage

        log(f"    Rules:")
        for rule in model_anchor_rules:
            log(f"      - {rule}")
        log(f"    Precision: {model_precision:.4f}")
        log(f"    Coverage: {model_coverage:.4f}")

        # Parse model anchor to bounds
        model_lower, model_upper = parse_alibi_anchor_to_bounds(
            model_anchor_rules, instance, feature_names, X_train
        )

        # DEBUG: Print bounds for Instance 77
        if test_idx == 77:
            log("\n[DEBUG] Model anchor bounds for Instance 77:")
            for i, fn in enumerate(feature_names):
                feat_min = X_train[:, i].min()
                feat_max = X_train[:, i].max()
                if model_lower[i] > feat_min or model_upper[i] < feat_max:
                    log(f"    {fn}: [{model_lower[i]:.2f}, {model_upper[i]:.2f}] (data range: [{feat_min:.2f}, {feat_max:.2f}])")
            # Count samples matching model anchor
            mask = np.ones(len(X_train), dtype=bool)
            for i in range(X_train.shape[1]):
                mask &= (X_train[:, i] >= model_lower[i]) & (X_train[:, i] <= model_upper[i])
            log(f"    Samples matching model anchor: {np.sum(mask)}")

        # Grow ground truth positive anchor (using ALL data with true labels)
        log("\n[2] GROUND TRUTH POSITIVE ANCHOR:")
        gt_anchor = grow_positive_anchor(instance, X_all, y_all, feature_names, delta=DELTA, max_rules=MAX_GT_RULES)

        log(f"    Rules:")
        for rule in gt_anchor['rules'][:10]:
            log(f"      - {rule}")
        if len(gt_anchor['rules']) > 10:
            log(f"      ... and {len(gt_anchor['rules']) - 10} more rules")
        log(f"    Precision: {gt_anchor['precision']:.4f} ({gt_anchor['n_positive']}/{gt_anchor['n_samples']} positive)")
        log(f"    Coverage: {gt_anchor['coverage']:.4f}")

        # Compute overlap
        log("\n[3] OVERLAP ANALYSIS:")
        overlap = compute_overlap(
            gt_anchor['lower_bounds'], gt_anchor['upper_bounds'],
            model_lower, model_upper,
            X_train, y_train, model
        )

        log(f"    Samples in positive anchor (GT): {overlap['n_in_positive_anchor']}")
        log(f"    Samples in negative anchor (model): {overlap['n_in_negative_anchor']}")
        log(f"    Samples in overlap: {overlap['n_overlap']}")
        log(f"    Overlap as % of GT anchor: {100*overlap['overlap_fraction_of_positive']:.1f}%")

        if overlap['n_overlap'] > 0:
            log(f"    In overlap: True positive rate: {100*overlap['overlap_true_positive_rate']:.1f}%")
            log(f"    In overlap: Model predicts positive: {100*overlap['overlap_pred_positive_rate']:.1f}%")
            violation_strength = overlap['overlap_true_positive_rate'] - overlap['overlap_pred_positive_rate']
            log(f"    VIOLATION STRENGTH: {100*violation_strength:.1f}% (GT - Model)")

        # Store result
        results.append({
            'test_idx': int(test_idx),
            'instance': instance.tolist(),
            'model_anchor': {
                'rules': model_anchor_rules,
                'precision': float(model_precision),
                'coverage': float(model_coverage)
            },
            'ground_truth_anchor': {
                'rules': gt_anchor['rules'],
                'precision': float(gt_anchor['precision']),
                'coverage': float(gt_anchor['coverage']),
                'n_samples': int(gt_anchor['n_samples'])
            },
            'overlap': {
                'n_in_positive': int(overlap['n_in_positive_anchor']),
                'n_in_negative': int(overlap['n_in_negative_anchor']),
                'n_overlap': int(overlap['n_overlap']),
                'overlap_pct_of_positive': float(overlap['overlap_fraction_of_positive']),
                'violation_strength': float(overlap['overlap_true_positive_rate'] - overlap['overlap_pred_positive_rate']) if overlap['n_overlap'] > 0 else 0.0
            }
        })

        del model_explanation
        gc.collect()

    # Save results
    results_file = os.path.join(output_dir, f'anchor_results_{model_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {results_file}")

    report_file = os.path.join(output_dir, f'anchor_report_{model_name}.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(output_lines))
    log(f"Report saved to: {report_file}")

    return results


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("ANCHOR EXPLANATIONS FOR FALSE NEGATIVES")
    print("Goal: Detect violations of ground truth anchors/rules")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load models and data
    models, data, fn_indices = load_models_and_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # Output directory
    output_dir = SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    # Run for each model
    for model_name, model in models.items():
        false_negatives = fn_indices[model_name]
        print(f"\n{model_name}: {len(false_negatives)} false negatives")

        results = run_anchor_analysis(
            model, model_name, X_train, X_test, y_train, y_test,
            feature_names, false_negatives, output_dir
        )
        all_results[model_name] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name, results in all_results.items():
        violations_with_overlap = [r for r in results if r['overlap']['n_overlap'] > 0]
        strong_violations = [r for r in results if r['overlap']['violation_strength'] > 0.5]

        print(f"\n{model_name}:")
        print(f"  False negatives analyzed: {len(results)}")
        print(f"  Cases with anchor overlap: {len(violations_with_overlap)}")
        print(f"  Strong violations (>50% GT-Model): {len(strong_violations)}")
        if violations_with_overlap:
            avg_violation = np.mean([r['overlap']['violation_strength'] for r in violations_with_overlap])
            print(f"  Avg violation strength: {100*avg_violation:.1f}%")

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for model_name in models.keys():
        print(f"  - anchor_results_{model_name}.json")
        print(f"  - anchor_report_{model_name}.txt")

    return all_results


if __name__ == "__main__":
    all_results = main()
