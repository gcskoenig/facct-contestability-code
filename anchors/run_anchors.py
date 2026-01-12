"""
Model Anchor Computation for Paper Example

Extracts the model anchor directly from the shallow decision tree's structure
for a specific false negative instance, and compares it with a prior knowledge rule.
"""

import os
import pickle
import numpy as np

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, 'model')

# Configuration
TEST_INDEX = 1  # The false negative instance to explain

# Prior knowledge rule (given, not computed)
PRIOR_RULE_DURATION_THRESHOLD = 31
PRIOR_RULE_SAVINGS_THRESHOLD = 3  # savings_status >= 3


def load_model_and_data():
    """Load shallow tree model and data splits."""
    with open(os.path.join(MODEL_DIR, 'shallow_tree.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'data_splits.pkl'), 'rb') as f:
        data = pickle.load(f)
    return model, data


def extract_anchor_from_tree(model, instance, feature_names):
    """
    Extract the decision path (anchor) from the decision tree for a given instance.
    Returns the rules that lead to the prediction.
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold

    # Get the decision path
    node_indicator = model.decision_path(instance.reshape(1, -1))
    node_indices = node_indicator.indices

    rules = []
    for node_id in node_indices:
        # Skip leaf nodes (feature[node_id] == -2 for leaves)
        if feature[node_id] != -2:
            feat_idx = feature[node_id]
            feat_name = feature_names[feat_idx]
            thresh = threshold[node_id]

            # Check which branch was taken
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

    return rules


def apply_rules_to_data(rules, X):
    """Apply anchor rules to data and return boolean mask."""
    mask = np.ones(len(X), dtype=bool)
    for rule in rules:
        feat_idx = rule['feature_idx']
        thresh = rule['threshold']
        if rule['op'] == '<=':
            mask &= X[:, feat_idx] <= thresh
        else:
            mask &= X[:, feat_idx] > thresh
    return mask


def main():
    # Load model and data
    model, data = load_model_and_data()
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feature_names = data['feature_names']

    # Get the instance to explain
    instance = X_test[TEST_INDEX]
    true_label = y_test[TEST_INDEX]
    prediction = model.predict(instance.reshape(1, -1))[0]

    # Extract anchor from decision tree
    print("Extracting model anchor from decision tree...")
    rules = extract_anchor_from_tree(model, instance, feature_names)

    # Compute model anchor support and true positive rate
    model_anchor_mask = apply_rules_to_data(rules, X_train)
    model_anchor_support = np.sum(model_anchor_mask)
    model_anchor_tpr = np.mean(y_train[model_anchor_mask]) if model_anchor_support > 0 else 0.0
    model_anchor_n_positive = int(np.sum(y_train[model_anchor_mask])) if model_anchor_support > 0 else 0

    # Model precision: fraction of samples in anchor where model predicts negative
    model_preds_in_anchor = model.predict(X_train[model_anchor_mask])
    model_precision = np.mean(model_preds_in_anchor == 0) if model_anchor_support > 0 else 0.0

    # Prior knowledge rule
    duration_idx = feature_names.index('duration')
    savings_idx = feature_names.index('savings_status')

    prior_rule_mask = (
        (X_train[:, duration_idx] > PRIOR_RULE_DURATION_THRESHOLD) &
        (X_train[:, savings_idx] >= PRIOR_RULE_SAVINGS_THRESHOLD)
    )
    prior_rule_support = np.sum(prior_rule_mask)
    prior_rule_n_positive = int(np.sum(y_train[prior_rule_mask])) if prior_rule_support > 0 else 0
    prior_rule_precision = np.mean(y_train[prior_rule_mask]) if prior_rule_support > 0 else 0.0

    # Compute overlap
    overlap_mask = model_anchor_mask & prior_rule_mask
    overlap_count = np.sum(overlap_mask)
    overlap_n_positive = int(np.sum(y_train[overlap_mask])) if overlap_count > 0 else 0
    overlap_precision = np.mean(y_train[overlap_mask]) if overlap_count > 0 else 0.0

    # Generate output
    output_lines = [
        "=" * 80,
        "MODEL ANCHOR EXAMPLE FOR PAPER (Test Index 1)",
        "=" * 80,
        "",
        "METHODOLOGY",
        "-----------",
        "Anchor extraction: Decision path from sklearn DecisionTreeClassifier",
        "Dataset: German Credit (OpenML ID=31, N=1000)",
        f"Training set: {len(X_train)} samples",
        f"Test set: {len(X_test)} samples",
        "",
        f"EXPLAINED INSTANCE (Test Index {TEST_INDEX})",
        "---------------------------------",
    ]

    for i, fname in enumerate(feature_names):
        output_lines.append(f"{fname}: {instance[i]}")

    output_lines.extend([
        "",
        f"True label: {true_label} ({'good credit' if true_label == 1 else 'bad credit'})",
        f"Model prediction: {prediction} ({'good credit' if prediction == 1 else 'bad credit'})",
        f"Classification: {'False negative' if true_label == 1 and prediction == 0 else 'Other'}",
        "",
        "MODEL ANCHOR (Shallow Decision Tree, max_depth=3)",
        "-------------------------------------------------",
        "Rules (decision path to prediction):",
    ])

    for rule in rules:
        output_lines.append(f"  - {rule['feature']} {rule['op']} {rule['threshold']:.2f}")

    output_lines.extend([
        "",
        f"Support (training): {model_anchor_support}",
        f"Model precision: {model_precision:.4f} (predicts negative for {model_precision*100:.1f}% of samples)",
        f"True positive rate in anchor region: {model_anchor_tpr:.4f} ({model_anchor_n_positive}/{model_anchor_support})",
        "",
        "PRIOR KNOWLEDGE RULE",
        "--------------------",
        "Rules:",
        f"  - duration > {PRIOR_RULE_DURATION_THRESHOLD}",
        f"  - savings_status >= {PRIOR_RULE_SAVINGS_THRESHOLD}",
        "",
        f"Support (training): {prior_rule_support}",
        f"Precision: {prior_rule_precision:.4f} ({prior_rule_n_positive}/{prior_rule_support})",
        "",
        "OVERLAP",
        "-------",
        f"Overlap (training): {overlap_count}",
        f"Precision in overlap: {overlap_precision:.4f} ({overlap_n_positive}/{overlap_count})" if overlap_count > 0 else "Precision in overlap: N/A (no overlap)",
        "",
        "FEATURE ENCODINGS",
        "-----------------",
        "savings_status:",
        "  0: 100<=X<500",
        "  1: 500<=X<1000",
        "  2: <100",
        "  3: >=1000 (rich)",
        "  4: no known savings",
    ])

    # Write output file
    output_path = os.path.join(SCRIPT_DIR, 'anchor_paper_example.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Output written to: {output_path}")
    print("\n" + "\n".join(output_lines))


if __name__ == "__main__":
    main()
