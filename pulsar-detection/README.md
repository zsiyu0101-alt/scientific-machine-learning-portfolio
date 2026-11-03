# Pulsar Detection

## Setup and Execution

Requires Python and R. Install Python dependencies with `pip install -r requirements.txt`. The R script attempts to install its required packages: caret, MASS, rpart, kernlab, randomForest, ggplot2, pROC and e1071.

1. Place `train_agnpsr_3fgl.csv` and `test_agnpsr_3fgl.csv` in this project's `data/` folder, or set `PULSAR_DATA_DIR` to their directory. Both files must contain 35 numeric predictors and a `pulsarness` label (Pulsar / Non-Pulsar). These CSV files are not included.
2. From this directory, run `Rscript Pulsar_Detection_Part_A.R` to generate `Part_A_results/`.
3. Run `python Pulsar_Detection_Part_B.py` to generate `Part_B_results/`. Its comparison stage requires the prediction and metric files from Part A.

Outputs include metric tables, test predictions, fitted models and figures. Dependency versions are not pinned; record the actual environment used. The original fixed-sample-size assertions remain in the code; review these assumptions when changing datasets.

References are linked within the code. The 3FGL catalog itself is not equivalent to the course-supplied training/test split used in this experiment.
