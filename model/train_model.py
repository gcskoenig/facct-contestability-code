"""
Model Training Script for Explainability Experiments

This script trains and saves models that will be used consistently across all
explanation methods (LIME, counterfactuals, anchors).

Models trained:
1. Random Forest with GridSearchCV - main model for production-like predictions
2. Decision Tree (default) - unstable model to detect overfitting/violations

Both models are trained on the German Credit dataset and saved for reuse.
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
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(MODEL_DIR), 'data')


def load_german_credit_data():
    """Load German Credit dataset from OpenML."""
    print("Loading German Credit dataset from OpenML (data_id=31)...")
    german = fetch_openml(data_id=31, as_frame=True)
    df = german.frame
    target_col = german.target_names[0] if german.target_names else 'class'
    feature_names = [col for col in df.columns if col != target_col]

    X_df = df[feature_names].copy()
    label_encoders = {}

    for col in X_df.columns:
        if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            label_encoders[col] = le

    X = X_df.values.astype(float)
    y = (df[target_col] == 'good').astype(int).values

    print(f"  Dataset shape: {X.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Class distribution: Good={np.sum(y)}, Bad={np.sum(1-y)}")

    return X, y, list(feature_names), label_encoders


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with GridSearchCV."""
    print("\nTraining Random Forest with GridSearchCV...")

    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED),
        rf_params, cv=3, scoring='accuracy', n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    model = rf_grid.best_estimator_

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"  Best params: {rf_grid.best_params_}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    return model, rf_grid.best_params_, train_acc, test_acc


def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train Decision Tree with default (no constraints) parameters."""
    print("\nTraining Decision Tree (default parameters - no constraints)...")

    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"  Tree depth: {model.get_depth()}")
    print(f"  Number of leaves: {model.get_n_leaves()}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    return model, train_acc, test_acc


def train_shallow_tree(X_train, y_train, X_test, y_test, max_depth=3):
    """Train a shallow Decision Tree with limited depth."""
    print(f"\nTraining Shallow Decision Tree (max_depth={max_depth})...")

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"  Tree depth: {model.get_depth()}")
    print(f"  Number of leaves: {model.get_n_leaves()}")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    return model, train_acc, test_acc


def find_false_negatives(model, X_test, y_test):
    """Find false negatives: predicted Bad (0) but actually Good (1)."""
    y_pred = model.predict(X_test)
    false_negative_mask = (y_pred == 0) & (y_test == 1)
    false_negative_indices = np.where(false_negative_mask)[0]
    return false_negative_indices, y_pred


def main():
    print("=" * 70)
    print("MODEL TRAINING FOR EXPLAINABILITY EXPERIMENTS")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")

    # Load data
    X, y, feature_names, label_encoders = load_german_credit_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train Random Forest
    rf_model, rf_params, rf_train_acc, rf_test_acc = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # Train Decision Tree
    dt_model, dt_train_acc, dt_test_acc = train_decision_tree(
        X_train, y_train, X_test, y_test
    )

    # Train Shallow Decision Tree
    st_model, st_train_acc, st_test_acc = train_shallow_tree(
        X_train, y_train, X_test, y_test, max_depth=3
    )

    # Find false negatives for all models
    rf_fn_indices, rf_y_pred = find_false_negatives(rf_model, X_test, y_test)
    dt_fn_indices, dt_y_pred = find_false_negatives(dt_model, X_test, y_test)
    st_fn_indices, st_y_pred = find_false_negatives(st_model, X_test, y_test)

    print(f"\nRandom Forest false negatives: {len(rf_fn_indices)}")
    print(f"Decision Tree false negatives: {len(dt_fn_indices)}")
    print(f"Shallow Tree false negatives: {len(st_fn_indices)}")

    # Save models
    print("\nSaving models...")

    rf_path = os.path.join(MODEL_DIR, 'random_forest.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"  Saved: {rf_path}")

    dt_path = os.path.join(MODEL_DIR, 'decision_tree.pkl')
    with open(dt_path, 'wb') as f:
        pickle.dump(dt_model, f)
    print(f"  Saved: {dt_path}")

    st_path = os.path.join(MODEL_DIR, 'shallow_tree.pkl')
    with open(st_path, 'wb') as f:
        pickle.dump(st_model, f)
    print(f"  Saved: {st_path}")

    # Save data splits
    data_bundle = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }
    data_path = os.path.join(MODEL_DIR, 'data_splits.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data_bundle, f)
    print(f"  Saved: {data_path}")

    # Save false negative indices
    fn_bundle = {
        'random_forest': rf_fn_indices.tolist(),
        'decision_tree': dt_fn_indices.tolist(),
        'shallow_tree': st_fn_indices.tolist()
    }
    fn_path = os.path.join(MODEL_DIR, 'false_negatives.json')
    with open(fn_path, 'w') as f:
        json.dump(fn_bundle, f, indent=2)
    print(f"  Saved: {fn_path}")

    # Generate model description file
    desc_path = os.path.join(MODEL_DIR, 'model_description.txt')
    with open(desc_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL DESCRIPTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n\n")

        f.write("DATASET\n")
        f.write("-" * 70 + "\n")
        f.write("German Credit dataset (OpenML data_id=31)\n")
        f.write(f"  - Total samples: {len(X)}\n")
        f.write(f"  - Features: {len(feature_names)}\n")
        f.write(f"  - Train/Test split: {len(X_train)}/{len(X_test)} (80/20)\n")
        f.write(f"  - Target: Good Credit (1) vs Bad Credit (0)\n")
        f.write(f"  - Class distribution: Good={np.sum(y)}, Bad={np.sum(1-y)}\n\n")

        f.write("MODEL 1: RANDOM FOREST\n")
        f.write("-" * 70 + "\n")
        f.write("Purpose: Main model for production-like predictions\n")
        f.write("Training: GridSearchCV with 3-fold cross-validation\n")
        f.write(f"Best hyperparameters:\n")
        for k, v in rf_params.items():
            f.write(f"  - {k}: {v}\n")
        f.write(f"Performance:\n")
        f.write(f"  - Train accuracy: {rf_train_acc:.4f}\n")
        f.write(f"  - Test accuracy: {rf_test_acc:.4f}\n")
        f.write(f"  - False negatives (test): {len(rf_fn_indices)}\n")
        f.write(f"  - False negative indices: {rf_fn_indices.tolist()}\n\n")

        f.write("MODEL 2: DECISION TREE (default)\n")
        f.write("-" * 70 + "\n")
        f.write("Purpose: Unstable model to detect overfitting/violations\n")
        f.write("Training: Default parameters (no constraints)\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  - max_depth: None (unlimited)\n")
        f.write(f"  - min_samples_split: 2\n")
        f.write(f"  - min_samples_leaf: 1\n")
        f.write(f"  - random_state: {RANDOM_SEED}\n")
        f.write(f"Tree statistics:\n")
        f.write(f"  - Depth: {dt_model.get_depth()}\n")
        f.write(f"  - Number of leaves: {dt_model.get_n_leaves()}\n")
        f.write(f"Performance:\n")
        f.write(f"  - Train accuracy: {dt_train_acc:.4f}\n")
        f.write(f"  - Test accuracy: {dt_test_acc:.4f}\n")
        f.write(f"  - False negatives (test): {len(dt_fn_indices)}\n")
        f.write(f"  - False negative indices: {dt_fn_indices.tolist()}\n\n")

        f.write("USAGE\n")
        f.write("-" * 70 + "\n")
        f.write("To load models in other scripts:\n\n")
        f.write("```python\n")
        f.write("import pickle\n")
        f.write("import json\n\n")
        f.write("# Load models\n")
        f.write("with open('model/random_forest.pkl', 'rb') as f:\n")
        f.write("    rf_model = pickle.load(f)\n")
        f.write("with open('model/decision_tree.pkl', 'rb') as f:\n")
        f.write("    dt_model = pickle.load(f)\n\n")
        f.write("# Load data splits\n")
        f.write("with open('model/data_splits.pkl', 'rb') as f:\n")
        f.write("    data = pickle.load(f)\n")
        f.write("X_train, X_test = data['X_train'], data['X_test']\n")
        f.write("y_train, y_test = data['y_train'], data['y_test']\n")
        f.write("feature_names = data['feature_names']\n\n")
        f.write("# Load false negative indices\n")
        f.write("with open('model/false_negatives.json', 'r') as f:\n")
        f.write("    fn_indices = json.load(f)\n")
        f.write("```\n\n")

        f.write("=" * 70 + "\n")
        f.write("END OF MODEL DESCRIPTION\n")
        f.write("=" * 70 + "\n")

    print(f"  Saved: {desc_path}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFiles created in {MODEL_DIR}/:")
    print("  - random_forest.pkl (Random Forest model)")
    print("  - decision_tree.pkl (Decision Tree model)")
    print("  - data_splits.pkl (train/test data)")
    print("  - false_negatives.json (false negative indices)")
    print("  - model_description.txt (this description)")

    return rf_model, dt_model, data_bundle, fn_bundle


if __name__ == "__main__":
    rf_model, dt_model, data_bundle, fn_bundle = main()
