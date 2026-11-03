# Scientific Machine Learning Portfolio

Two course projects applying machine learning and scientific computing to astronomical classification and experimental fluid measurement.

| Project | Objective | Methods and tools | Source |
| --- | --- | --- | --- |
| Pulsar Detection | Classify pulsar and non-pulsar sources using 3FGL features | R/caret, Python, classical classifiers, MLP, cross-validation | [Explore project](pulsar-detection/) |
| Pendant-Drop Surface Tension | Estimate physical parameters and surface tension from droplet contours | OpenCV, SciPy, Young-Laplace equation, PyTorch, bidirectional GRU | [Explore project](pendant-drop/) |

## Pulsar Detection

**Workflow:** data checks and cleaning → training-only preprocessing → classical baselines → MLP candidate selection → multi-seed ensemble → held-out evaluation.

- Part A compares LDA with PCA, CART, kNN, linear SVM and random forest, exporting predictions, metrics and figures.
- Part B retains all 35 predictors, selects candidates using five-fold stratified cross-validation average precision, and learns an F1 threshold from training out-of-fold predictions.
- Evaluation includes PR/ROC curves, confusion matrices, calibration and paired uncertainty analysis. Tuning budgets differ across methods, so the comparison is not an equal-budget benchmark.

[Setup and running instructions](pulsar-detection/README.md)

## Pendant-Drop Surface Tension Estimation

**Workflow:** image edge and contour detection → contour sampling → Young-Laplace numerical simulation → data augmentation → neural parameter estimation → physical calibration and surface tension calculation.

- OpenCV extracts experimental droplet contours with intermediate visualizations.
- SciPy solves the droplet shape equation; synthetic curves receive sampling, rotation, noise and scale perturbations.
- PyTorch bidirectional GRU models estimate curve parameters, with needle-diameter calibration converting pixel measurements to physical units.

[Setup and current limitations](pendant-drop/README.md)

## Scope and Verification

This repository presents the supplied course project source code with organized folders and dependency instructions. Historical notebook outputs have been cleared. Pulsar CSV datasets and trained pendant-drop weights are not included. Full training and end-to-end reproduction have not been performed during repository preparation, so no unverified accuracy, error or superiority claims are reported. Teaching context and existing references are retained; the boundary between course examples and individual contributions should be explained using the original course records.
