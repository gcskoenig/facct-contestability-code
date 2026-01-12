"""
LIME Explanations for False Negatives

Goal: Detect monotonicity violations.

A monotonicity violation occurs when LIME shows a feature with unexpected sign:
- Features that should increase P(Good Credit) have negative weights
- Features that should decrease P(Good Credit) have positive weights

For credit scoring, there are domain expectations:
- Higher income should generally improve credit score (positive weight expected)
- Higher credit amount requested might decrease approval (could go either way)
- Longer employment duration should improve credit (positive weight expected)

We analyze LIME weights across all false negatives to find:
1. Features with inconsistent signs across instances
2. Features with weights contradicting domain expectations
3. High variance in feature attributions
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

# Import LIME
try:
    import lime
    import lime.lime_tabular
except ImportError:
    print("Error: lime package not found. Install with: pip install lime")
    sys.exit(1)

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


def run_lime_analysis(model, model_name, X_train, X_test, y_train, y_test,
                      feature_names, false_negative_indices, output_dir):
    """Run LIME analysis for a single model on all false negatives."""

    print(f"\n{'='*70}")
    print(f"LIME ANALYSIS FOR {model_name.upper()}")
    print(f"{'='*70}")

    # Create LIME explainer
    # kernel_width=0.5 with Chebyshev (L∞) distance focuses on local neighborhood
    # L∞ ensures features stay more independent conditional on weight
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Bad Credit', 'Good Credit'],
        mode='classification',
        discretize_continuous=False,
        kernel_width=0.5,
        random_state=RANDOM_SEED
    )

    all_lime_data = []
    instance_data = []
    output_lines = []

    def log(msg=""):
        print(msg)
        output_lines.append(msg)

    log(f"\nAnalyzing {len(false_negative_indices)} false negatives...")

    for i, test_idx in enumerate(false_negative_indices):
        if (i + 1) % 10 == 0:
            log(f"  Processing {i+1}/{len(false_negative_indices)}...")

        instance = X_test[test_idx]
        proba = model.predict_proba(instance.reshape(1, -1))[0]

        # Generate LIME explanation with Chebyshev (L∞) distance
        explanation = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=len(feature_names),
            distance_metric='chebyshev'
        )

        # Extract weights as dict
        lime_weights = dict(explanation.as_list())

        # Store data for each feature
        for feat in feature_names:
            weight = lime_weights.get(feat, 0.0)
            all_lime_data.append({
                'instance_idx': test_idx,
                'feature': feat,
                'feature_value': instance[feature_names.index(feat)],
                'lime_weight': weight,
                'abs_weight': abs(weight)
            })

        # Store instance info
        instance_data.append({
            'instance_idx': test_idx,
            'pred_prob_bad': proba[0],
            'pred_prob_good': proba[1],
            'true_label': y_test[test_idx],
            **{feat: instance[j] for j, feat in enumerate(feature_names)}
        })

    # Create DataFrames
    lime_df = pd.DataFrame(all_lime_data)
    instances_df = pd.DataFrame(instance_data)

    # Analyze monotonicity violations
    log("\n" + "=" * 70)
    log("MONOTONICITY ANALYSIS")
    log("=" * 70)

    # Feature summary
    summary_df = lime_df.groupby('feature').agg({
        'lime_weight': ['mean', 'std', 'min', 'max'],
        'abs_weight': 'mean'
    }).round(6)
    summary_df.columns = ['mean_weight', 'std_weight', 'min_weight', 'max_weight', 'mean_abs_weight']
    summary_df = summary_df.sort_values('mean_abs_weight', ascending=False)

    log("\nFeature summary (sorted by importance):")
    log("-" * 70)
    log(f"{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    log("-" * 70)
    for feat, row in summary_df.iterrows():
        log(f"{feat:<25} {row['mean_weight']:>10.4f} {row['std_weight']:>10.4f} "
            f"{row['min_weight']:>10.4f} {row['max_weight']:>10.4f}")

    # Detect sign inconsistencies (both positive and negative weights for same feature)
    log("\n" + "=" * 70)
    log("SIGN INCONSISTENCIES")
    log("(Features with both positive and negative weights across instances)")
    log("=" * 70)

    sign_inconsistent = []
    for feat in feature_names:
        feat_weights = lime_df[lime_df['feature'] == feat]['lime_weight']
        n_positive = int((feat_weights > 0.01).sum())
        n_negative = int((feat_weights < -0.01).sum())
        n_total = len(feat_weights)

        if n_positive > 0 and n_negative > 0:
            inconsistency_ratio = min(n_positive, n_negative) / n_total
            sign_inconsistent.append({
                'feature': feat,
                'n_positive': n_positive,
                'n_negative': n_negative,
                'inconsistency_ratio': float(inconsistency_ratio),
                'mean_weight': float(feat_weights.mean()),
                'std_weight': float(feat_weights.std())
            })

    sign_inconsistent.sort(key=lambda x: -x['inconsistency_ratio'])

    if sign_inconsistent:
        log(f"\n{'Feature':<25} {'+ Count':>10} {'- Count':>10} {'Incon.Ratio':>12}")
        log("-" * 60)
        for item in sign_inconsistent[:10]:
            log(f"{item['feature']:<25} {item['n_positive']:>10} {item['n_negative']:>10} "
                f"{item['inconsistency_ratio']:>12.2%}")
    else:
        log("\nNo sign inconsistencies found.")

    # Find features with high variance (unstable attributions)
    log("\n" + "=" * 70)
    log("HIGH VARIANCE FEATURES")
    log("(Features with unstable attributions - high std relative to mean)")
    log("=" * 70)

    high_variance = []
    for feat in feature_names:
        feat_data = summary_df.loc[feat]
        if feat_data['mean_abs_weight'] > 0.01:
            cv = feat_data['std_weight'] / (abs(feat_data['mean_weight']) + 0.001)
            if cv > 0.5:  # Coefficient of variation > 0.5
                high_variance.append({
                    'feature': feat,
                    'mean_weight': float(feat_data['mean_weight']),
                    'std_weight': float(feat_data['std_weight']),
                    'cv': float(cv)
                })

    high_variance.sort(key=lambda x: -x['cv'])

    if high_variance:
        log(f"\n{'Feature':<25} {'Mean':>10} {'Std':>10} {'CV':>10}")
        log("-" * 60)
        for item in high_variance[:10]:
            log(f"{item['feature']:<25} {item['mean_weight']:>10.4f} {item['std_weight']:>10.4f} "
                f"{item['cv']:>10.2f}")
    else:
        log("\nNo high-variance features found.")

    # Analyze credit_amount specifically (common domain feature)
    log("\n" + "=" * 70)
    log("CREDIT AMOUNT ANALYSIS")
    log("(Higher credit amounts might decrease approval - domain expectation)")
    log("=" * 70)

    if 'credit_amount' in feature_names:
        credit_weights = lime_df[lime_df['feature'] == 'credit_amount']
        positive_credit = credit_weights[credit_weights['lime_weight'] > 0.01]
        negative_credit = credit_weights[credit_weights['lime_weight'] < -0.01]

        log(f"\nInstances with POSITIVE credit_amount weight: {len(positive_credit)}")
        log(f"  (credit_amount increases P(Good Credit))")
        log(f"Instances with NEGATIVE credit_amount weight: {len(negative_credit)}")
        log(f"  (credit_amount decreases P(Good Credit))")

        if len(positive_credit) > 0:
            log(f"\nTop instances where credit_amount has positive effect:")
            top_positive = positive_credit.nlargest(5, 'lime_weight')
            for _, row in top_positive.iterrows():
                log(f"  Instance {int(row['instance_idx'])}: "
                    f"credit={row['feature_value']:.0f}, weight={row['lime_weight']:.4f}")

    # Save results
    lime_weights_file = os.path.join(output_dir, f'lime_weights_{model_name}.csv')
    lime_df.to_csv(lime_weights_file, index=False)
    log(f"\nLIME weights saved to: {lime_weights_file}")

    instances_file = os.path.join(output_dir, f'instances_{model_name}.csv')
    instances_df.to_csv(instances_file, index=False)
    log(f"Instance data saved to: {instances_file}")

    summary_file = os.path.join(output_dir, f'lime_summary_{model_name}.csv')
    summary_df.to_csv(summary_file)
    log(f"Feature summary saved to: {summary_file}")

    # Save monotonicity violations
    violations = {
        'sign_inconsistencies': sign_inconsistent,
        'high_variance_features': high_variance
    }
    violations_file = os.path.join(output_dir, f'monotonicity_violations_{model_name}.json')
    with open(violations_file, 'w') as f:
        json.dump(violations, f, indent=2)
    log(f"Violations saved to: {violations_file}")

    # Save report
    report_file = os.path.join(output_dir, f'lime_report_{model_name}.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(output_lines))
    log(f"Report saved to: {report_file}")

    return {
        'lime_df': lime_df,
        'instances_df': instances_df,
        'summary_df': summary_df,
        'sign_inconsistencies': sign_inconsistent,
        'high_variance_features': high_variance
    }


def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("LIME EXPLANATIONS FOR FALSE NEGATIVES")
    print("Goal: Detect monotonicity violations")
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

        results = run_lime_analysis(
            model, model_name, X_train, X_test, y_train, y_test,
            feature_names, false_negatives, output_dir
        )
        all_results[model_name] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name, results in all_results.items():
        sign_incon = results['sign_inconsistencies']
        high_var = results['high_variance_features']

        print(f"\n{model_name}:")
        print(f"  Features with sign inconsistencies: {len(sign_incon)}")
        print(f"  Features with high variance: {len(high_var)}")

        if sign_incon:
            top_incon = sign_incon[0]
            print(f"  Most inconsistent feature: {top_incon['feature']} "
                  f"({top_incon['inconsistency_ratio']:.1%} mixed signs)")

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    for model_name in models.keys():
        print(f"  - lime_weights_{model_name}.csv")
        print(f"  - instances_{model_name}.csv")
        print(f"  - lime_summary_{model_name}.csv")
        print(f"  - monotonicity_violations_{model_name}.json")
        print(f"  - lime_report_{model_name}.txt")

    return all_results


if __name__ == "__main__":
    all_results = main()
