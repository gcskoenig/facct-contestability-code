================================================================================
COUNTERFACTUAL EXPLANATIONS
================================================================================

GOAL: Detect continuity violations.

WHAT ARE COUNTERFACTUAL EXPLANATIONS?
-------------------------------------
Counterfactuals answer: "What minimal change would flip the prediction?"

For example, if a person is denied credit, a counterfactual might show:
"If your credit amount were 3000 instead of 5000, you would be approved."

This is a sparse counterfactual - changing as few features as possible.

WHAT WE'RE LOOKING FOR
----------------------
A CONTINUITY VIOLATION occurs when:
1. The model predicts Bad Credit with high confidence (low P(Good Credit))
2. BUT a counterfactual (point with opposite prediction) is very close

This indicates the decision boundary is UNSTABLE:
- The model is confident, but the boundary is extremely close
- Small perturbations flip the prediction
- This suggests overfitting or poor generalization

VIOLATION METRICS
-----------------
- cf_distance: Normalized Euclidean distance to counterfactual (features / std)
- n_changed: Number of features that needed to change
- violation_score: cf_distance * (prob_positive + 0.01)
  Lower = stronger violation (very close counterfactual for confident prediction)

WHY THIS MATTERS
----------------
- Continuity violations reveal unstable decision boundaries
- If a "confident" prediction can be flipped with tiny changes, the model is
  unreliable in that region
- This is a form of adversarial vulnerability
- Important for trust and reliability of ML systems

FILES IN THIS FOLDER
--------------------
- run_counterfactuals.py: Main script to run counterfactual analysis
- counterfactual_results_{model_name}.json: Detailed results for each instance
- counterfactuals_{model_name}.csv: Counterfactual instances
- violations_{model_name}.csv: Violation metrics
- counterfactual_report_{model_name}.txt: Human-readable report

USAGE
-----
From the project root:
  python counterfactuals/run_counterfactuals.py

================================================================================
