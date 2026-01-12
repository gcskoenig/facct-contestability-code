================================================================================
LIME EXPLANATIONS
================================================================================

GOAL: Detect monotonicity violations.

WHAT ARE LIME EXPLANATIONS?
---------------------------
LIME (Local Interpretable Model-agnostic Explanations) creates a local linear
approximation around a specific instance to explain the prediction.

LIME assigns weights to each feature indicating how much that feature
contributes to pushing the prediction toward one class or the other.

For example:
- credit_amount = +0.15 means higher credit amount increases P(Good Credit)
- duration = -0.08 means longer duration decreases P(Good Credit)

WHAT WE'RE LOOKING FOR
----------------------
A MONOTONICITY VIOLATION occurs when:
1. A feature has an unexpected sign in its LIME weight
2. The same feature has INCONSISTENT signs across different instances
3. Features have high variance in their attributions

Examples:
- If "duration" sometimes increases P(Good Credit) and sometimes decreases it,
  that's a sign inconsistency
- If a feature that should logically have a positive effect (like income)
  shows negative weights, that may indicate a problem

VIOLATION METRICS
-----------------
- Sign inconsistencies: Features with both positive and negative weights
  across different instances (inconsistency_ratio = min(+,-) / total)
- High variance features: Features where std >> |mean|, indicating unstable
  attributions (coefficient of variation > 0.5)

WHY THIS MATTERS
----------------
- Monotonicity is a fundamental expectation in many domains
- In credit scoring, we expect: higher income = better, longer employment = better
- If LIME shows violations of these expectations, it indicates:
  * Model has learned spurious correlations
  * Feature interactions are confusing the model
  * Potential fairness or bias issues
- Useful for model debugging and validation

FILES IN THIS FOLDER
--------------------
- run_lime.py: Main script to run LIME analysis
- lime_weights_{model_name}.csv: LIME weights for all instances and features
- instances_{model_name}.csv: Instance data with predictions
- lime_summary_{model_name}.csv: Feature-level statistics
- monotonicity_violations_{model_name}.json: Detected violations
- lime_report_{model_name}.txt: Human-readable report

USAGE
-----
From the project root:
  python lime/run_lime.py

Requirements: lime package (pip install lime)

================================================================================
