================================================================================
ANCHORS FOLDER
================================================================================

This folder contains the anchor-based analysis for detecting model errors
by comparing model anchors with ground truth anchors.

METHOD
------
1. For each false negative (model predicts Bad, actually Good):
   - Compute MODEL ANCHOR: Region where model predicts Bad Credit (via alibi)
   - Compute GT ANCHOR: Region where ground truth is Good Credit (90%+ precision)

2. Check for violations:
   - If overlap between anchors exceeds (1 - model_precision) of model anchor
   - Then model's precision claim is violated, proving systematic errors

CONFIGURATION
-------------
Current settings in run_anchors.py:
  DELTA = 0.1           # GT anchor requires 90% precision
  MAX_GT_RULES = 2      # GT anchor limited to 2 rules (for interpretability)
  MIN_SAMPLES_IN_BOX = 5

FILES
-----
  run_anchors.py                    Main analysis script

  anchor_results_decision_tree.json Full results for Decision Tree (JSON)
  anchor_results_random_forest.json Full results for Random Forest (JSON)

  anchor_report_decision_tree.txt   Human-readable report for Decision Tree
  anchor_report_random_forest.txt   Human-readable report for Random Forest

  VIOLATION_EXAMPLE.txt             Detailed example of a violation with
                                    plain-language interpretation

RESULTS SUMMARY
---------------
Decision Tree: 6 violations found out of 37 false negatives
Random Forest: Results available in anchor_results_random_forest.json

Best example: Test Index 91 (Decision Tree)
  - Model anchor: 48 samples, 75% precision
  - GT anchor: 311 samples, 90% precision
  - Overlap: 15 samples > threshold of 11.9
  - See VIOLATION_EXAMPLE.txt for full details

USAGE
-----
To run the analysis:
  cd /path/to/project
  source venv311/bin/activate
  python anchors/run_anchors.py

Output files are saved to this folder.

DEPENDENCIES
------------
  - alibi (for AnchorTabular explainer)
  - scikit-learn
  - numpy

================================================================================
