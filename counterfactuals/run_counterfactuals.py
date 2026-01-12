"""
Counterfactual Explanations for False Negatives

Goal: Detect continuity violations.

A continuity violation occurs when:
- An instance is predicted negative with high confidence
- But a counterfactual (point with opposite prediction) is very close

This indicates the model's decision boundary is unstable - it's confident but
the boundary is extremely close, suggesting overfitting or poor generalization.

For each false negative, we find the minimal counterfactual and report:
- Distance to counterfactual (normalized by feature std)
- Number of features changed
- Which features changed and by how much
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


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

    return {
        'random_forest': rf_model,
        'decision_tree': dt_model
    }, data, fn_indices


def find_sparse_counterfactual(instance, model, X_train, feature_names, feature_stds, max_features=2):
    """
    Find the sparsest counterfactual by trying 1-feature and 2-feature changes.

    Returns the counterfactual with minimum distance that flips the prediction.
    """
    original_pred = model.predict(instance.reshape(1, -1))[0]
    n_features = len(feature_names)

    best_cf = None
    best_distance = float('inf')
    best_n_changed = float('inf')
    changed_features = []

    # Try single feature changes first
    for i in range(n_features):
        unique_vals = np.unique(X_train[:, i])
        for val in unique_vals:
            if val == instance[i]:
                continue
            cf = instance.copy()
            cf[i] = val
            pred = model.predict(cf.reshape(1, -1))[0]
            if pred != original_pred:
                dist = np.linalg.norm((cf - instance) / feature_stds)
                if dist < best_distance:
                    best_distance = dist
                    best_cf = cf.copy()
                    best_n_changed = 1
                    changed_features = [(feature_names[i], instance[i], val)]

    # If we found a 1-feature counterfactual, return it
    if best_n_changed == 1:
        return best_cf, best_distance, best_n_changed, changed_features

    # Try 2-feature changes
    if max_features >= 2:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                unique_vals_i = np.unique(X_train[:, i])
                unique_vals_j = np.unique(X_train[:, j])

                # Sample to avoid combinatorial explosion
                if len(unique_vals_i) > 10:
                    unique_vals_i = np.random.choice(unique_vals_i, 10, replace=False)
                if len(unique_vals_j) > 10:
                    unique_vals_j = np.random.choice(unique_vals_j, 10, replace=False)

                for val_i in unique_vals_i:
                    for val_j in unique_vals_j:
                        if val_i == instance[i] and val_j == instance[j]:
                            continue
                        cf = instance.copy()
                        cf[i] = val_i
                        cf[j] = val_j
                        pred = model.predict(cf.reshape(1, -1))[0]
                        if pred != original_pred:
                            dist = np.linalg.norm((cf - instance) / feature_stds)
                            if dist < best_distance:
                                best_distance = dist
                                best_cf = cf.copy()
                                best_n_changed = 2
                                changed_features = []
                                if val_i != instance[i]:
                                    changed_features.append((feature_names[i], instance[i], val_i))
                                if val_j != instance[j]:
                                    changed_features.append((feature_names[j], instance[j], val_j))

    if best_cf is None:
        return None, float('inf'), 0, []

    return best_cf, best_distance, best_n_changed, changed_features


def run_counterfactual_analysis(model, model_name, X_train, X_test, y_train, y_test,
                                 feature_names, false_negative_indices, output_dir):
    """Run counterfactual analysis for a single model on all false negatives."""

    print(f"\n{'='*70}")
    print(f"COUNTERFACTUAL ANALYSIS FOR {model_name.upper()}")
    print(f"{'='*70}")

    # Compute feature standard deviations for normalized distance
    feature_stds = np.std(X_train, axis=0)
    feature_stds[feature_stds == 0] = 1.0  # Avoid division by zero

    results = []
    counterfactuals_data = []
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"\nAnalyzing {len(false_negative_indices)} false negatives...")
    log(f"Distance metric: Normalized Euclidean (features divided by std)")

    for i, test_idx in enumerate(false_negative_indices):
        if (i + 1) % 10 == 0:
            log(f"  Processing {i+1}/{len(false_negative_indices)}...")

        instance = X_test[test_idx]
        prob_positive = model.predict_proba(instance.reshape(1, -1))[0, 1]

        # Find counterfactual
        cf_values, cf_distance, n_changed, changed_features = find_sparse_counterfactual(
            instance, model, X_train, feature_names, feature_stds, max_features=2
        )

        if cf_values is not None:
            cf_prob_positive = model.predict_proba(cf_values.reshape(1, -1))[0, 1]

            # Violation score: low prob_positive AND low distance = strong violation
            violation_score = cf_distance * (prob_positive + 0.01)

            result = {
                'test_idx': int(test_idx),
                'prob_positive': float(prob_positive),
                'cf_prob_positive': float(cf_prob_positive),
                'cf_distance': float(cf_distance),
                'n_changed': int(n_changed),
                'changed_features': [(f, float(old), float(new)) for f, old, new in changed_features],
                'violation_score': float(violation_score),
                'true_label': int(y_test[test_idx])
            }
            results.append(result)

            # Store counterfactual data
            cf_row = {'datapoint_idx': test_idx}
            for j, feat in enumerate(feature_names):
                cf_row[feat] = cf_values[j]
            counterfactuals_data.append(cf_row)

    # Sort by violation score (lower = stronger violation)
    results.sort(key=lambda x: x['violation_score'])

    # Log top violations
    log("\n" + "=" * 70)
    log("TOP CONTINUITY VIOLATIONS")
    log("(Low distance + low confidence = strong violation)")
    log("=" * 70)

    for rank, r in enumerate(results[:10]):
        log(f"\nRank {rank + 1}: Test instance {r['test_idx']}")
        log(f"  P(Good Credit) = {r['prob_positive']:.4f}")
        log(f"  Distance to counterfactual: {r['cf_distance']:.4f}")
        log(f"  Features changed: {r['n_changed']}")
        log(f"  Violation score: {r['violation_score']:.6f}")
        log(f"  True label: {'Good' if r['true_label'] == 1 else 'Bad'} Credit")
        log(f"  Changes:")
        for feat, old, new in r['changed_features']:
            log(f"    {feat}: {old:.2f} -> {new:.2f}")

    # Save results
    results_file = os.path.join(output_dir, f'counterfactual_results_{model_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to: {results_file}")

    # Save counterfactuals as CSV
    cf_df = pd.DataFrame(counterfactuals_data)
    cf_file = os.path.join(output_dir, f'counterfactuals_{model_name}.csv')
    cf_df.to_csv(cf_file, index=False)
    log(f"Counterfactuals saved to: {cf_file}")

    # Save violations summary
    violations_df = pd.DataFrame([{
        'datapoint_idx': r['test_idx'],
        'pred_prob_datapoint': r['prob_positive'],
        'pred_prob_counterfactual': r['cf_prob_positive'],
        'distance': r['cf_distance'],
        'n_changed': r['n_changed'],
        'violation_score': r['violation_score']
    } for r in results])
    violations_file = os.path.join(output_dir, f'violations_{model_name}.csv')
    violations_df.to_csv(violations_file, index=False)
    log(f"Violations saved to: {violations_file}")

    # Save report
    report_file = os.path.join(output_dir, f'counterfactual_report_{model_name}.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(output_lines))
    log(f"Report saved to: {report_file}")

    return results


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("COUNTERFACTUAL EXPLANATIONS FOR FALSE NEGATIVES")
    print("Goal: Detect continuity violations")
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

        results = run_counterfactual_analysis(
            model, model_name, X_train, X_test, y_train, y_test,
            feature_names, false_negatives, output_dir
        )
        all_results[model_name] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name, results in all_results.items():
        if not results:
            print(f"\n{model_name}: No counterfactuals found")
            continue

        distances = [r['cf_distance'] for r in results]
        violation_scores = [r['violation_score'] for r in results]

        print(f"\n{model_name}:")
        print(f"  False negatives with counterfactuals: {len(results)}")
        print(f"  Average distance to counterfactual: {np.mean(distances):.4f}")
        print(f"  Min distance (strongest violation): {np.min(distances):.4f}")
        print(f"  Average violation score: {np.mean(violation_scores):.6f}")

        # Count very close counterfactuals (strong violations)
        close_cfs = sum(1 for d in distances if d < 1.0)
        print(f"  Counterfactuals within 1.0 distance: {close_cfs}")

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for model_name in models.keys():
        print(f"  - counterfactual_results_{model_name}.json")
        print(f"  - counterfactuals_{model_name}.csv")
        print(f"  - violations_{model_name}.csv")
        print(f"  - counterfactual_report_{model_name}.txt")

    return all_results


if __name__ == "__main__":
    all_results = main()
