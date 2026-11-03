# Pulsar Detection Project - Part A
# Author: ZHOU Siyu
# This script trains five basic caret models on the supplied 3FGL split.
# Put the two CSV files beside this script, in a data subfolder, or set
# PULSAR_DATA_DIR to the folder containing both files.

set.seed(123)

# -----------------------------------------------------------------------------
# 1. Install and load packages
# -----------------------------------------------------------------------------
packages <- c("caret", "MASS", "rpart", "kernlab", "randomForest",
              "ggplot2", "pROC", "e1071")
missing_packages <- packages[!vapply(packages, requireNamespace,
                                    quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_packages) > 0L) {
  options(timeout = max(300, getOption("timeout")))
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}
unavailable <- packages[!vapply(packages, requireNamespace,
                               quietly = TRUE, FUN.VALUE = logical(1))]
if (length(unavailable) > 0L) {
  stop("These packages could not be loaded: ", paste(unavailable, collapse = ", "))
}
suppressPackageStartupMessages(library(caret))
suppressPackageStartupMessages(library(ggplot2))

# -----------------------------------------------------------------------------
# 2. Find and load the supplied training and test files
# -----------------------------------------------------------------------------
required_files <- c("train_agnpsr_3fgl.csv", "test_agnpsr_3fgl.csv")

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)))
  }
  normalizePath(getwd(), mustWork = FALSE)
}

script_dir <- get_script_dir()
configured_dir <- Sys.getenv("PULSAR_DATA_DIR", unset = "")
candidate_dirs <- unique(c(
  configured_dir,
  script_dir,
  file.path(script_dir, "data"),
  getwd(),
  file.path(getwd(), "data"),
  file.path(Sys.getenv("USERPROFILE", unset = ""), "Desktop"),
  file.path(Sys.getenv("USERPROFILE", unset = ""), "Desktop", "Part_B_Pulsar_Detection", "data")
))
candidate_dirs <- candidate_dirs[nzchar(candidate_dirs)]
valid_dirs <- candidate_dirs[vapply(candidate_dirs, function(folder) {
  all(file.exists(file.path(folder, required_files)))
}, logical(1))]

if (length(valid_dirs) == 0L) {
  stop(
    "Cannot find train_agnpsr_3fgl.csv and test_agnpsr_3fgl.csv. ",
    "Put both files beside this script, inside a data subfolder, or set PULSAR_DATA_DIR."
  )
}

data_dir <- normalizePath(valid_dirs[1], winslash = "/", mustWork = TRUE)
output_dir <- file.path(script_dir, "Part_A_results")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

train_data <- read.csv(file.path(data_dir, required_files[1]), stringsAsFactors = FALSE)
test_data <- read.csv(file.path(data_dir, required_files[2]), stringsAsFactors = FALSE)

stopifnot(
  "pulsarness" %in% names(train_data),
  !anyDuplicated(names(train_data)),
  !anyDuplicated(names(test_data)),
  setequal(names(train_data), names(test_data)),
  nrow(train_data) > 0L,
  nrow(test_data) > 0L
)
test_data <- test_data[, names(train_data), drop = FALSE]
predictors <- setdiff(names(train_data), "pulsarness")
stopifnot(length(predictors) == 35L)

cat("R version:", R.version.string, "\n")
cat("Data folder:", data_dir, "\n")
cat("Training observations:", nrow(train_data), "\n")
cat("Test observations:", nrow(test_data), "\n")
cat("Predictors retained:", length(predictors), "\n")

# -----------------------------------------------------------------------------
# 3. Clean labels and numeric values without deleting the first 34 features
# -----------------------------------------------------------------------------
prepare_labels <- function(x) {
  x <- trimws(x)
  stopifnot(!anyNA(x), all(x %in% c("Pulsar", "Non-Pulsar")))
  # caret requires valid factor names for class probabilities.
  factor(ifelse(x == "Non-Pulsar", "NonPulsar", x),
         levels = c("Pulsar", "NonPulsar"))
}

replace_nonfinite <- function(x) {
  x[!is.finite(x)] <- NA_real_
  x
}

train_data$pulsarness <- prepare_labels(train_data$pulsarness)
test_data$pulsarness <- prepare_labels(test_data$pulsarness)
stopifnot(
  all(vapply(train_data[predictors], is.numeric, logical(1))),
  all(vapply(test_data[predictors], is.numeric, logical(1)))
)
train_data[predictors] <- lapply(train_data[predictors], replace_nonfinite)
test_data[predictors] <- lapply(test_data[predictors], replace_nonfinite)
stopifnot(!any(vapply(train_data[predictors], function(x) all(is.na(x)), logical(1))))

cat("\nTraining class counts:\n")
print(table(train_data$pulsarness))
cat("\nTest class counts:\n")
print(table(test_data$pulsarness))
cat("\nFirst six rows:\n")
print(head(train_data))
cat("\nData summary:\n")
print(summary(train_data))

# -----------------------------------------------------------------------------
# 4. Visualize the training data
# -----------------------------------------------------------------------------
class_plot <- ggplot(train_data, aes(x = pulsarness, fill = pulsarness)) +
  geom_bar(width = 0.6) +
  labs(title = "Training Class Distribution", x = "Class", y = "Number of Sources") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave(file.path(output_dir, "01_class_distribution.png"), class_plot,
       width = 8, height = 5, dpi = 160)

feature_data <- stack(train_data[predictors])
names(feature_data) <- c("Value", "Feature")
feature_data$Class <- rep(train_data$pulsarness, times = length(predictors))

distribution_plot <- ggplot(feature_data, aes(x = Value)) +
  geom_histogram(bins = 25, fill = "steelblue", color = "white", na.rm = TRUE) +
  facet_wrap(~ Feature, scales = "free", ncol = 5) +
  labs(title = "Distributions of All Training Predictors", y = "Count") +
  theme_minimal(base_size = 10)
ggsave(file.path(output_dir, "02_feature_distributions.png"), distribution_plot,
       width = 14, height = 16, dpi = 160)

box_plot <- ggplot(feature_data, aes(x = Class, y = Value, fill = Class)) +
  geom_boxplot(outlier.size = 0.4, na.rm = TRUE) +
  facet_wrap(~ Feature, scales = "free_y", ncol = 5) +
  labs(title = "Training Predictors by Class") +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(output_dir, "03_features_by_class.png"), box_plot,
       width = 14, height = 16, dpi = 160)

# -----------------------------------------------------------------------------
# 5. Train LDA, CART, kNN, linear SVM and Random Forest
# -----------------------------------------------------------------------------
x_train <- train_data[predictors]
y_train <- train_data$pulsarness
x_test <- test_data[predictors]
y_test <- test_data$pulsarness

control <- trainControl(method = "none", classProbs = TRUE, allowParallel = FALSE)
control_lda <- trainControl(
  method = "none", classProbs = TRUE, allowParallel = FALSE,
  preProcOptions = list(thresh = 0.99)
)
preprocess_steps <- c("medianImpute", "zv", "center", "scale")

cat("\nTraining LDA with PCA...\n")
set.seed(123)
time_lda <- system.time({
  model_lda <- train(
    x = x_train, y = y_train, method = "lda", trControl = control_lda,
    preProcess = c(preprocess_steps, "pca")
  )
})
cat("LDA PCA components:", model_lda$preProcess$numComp, "\n")

cat("Training CART...\n")
set.seed(123)
time_cart <- system.time({
  model_cart <- train(
    x = x_train, y = y_train, method = "rpart", trControl = control,
    preProcess = preprocess_steps, tuneGrid = data.frame(cp = 0.01)
  )
})

cat("Training kNN...\n")
set.seed(123)
time_knn <- system.time({
  model_knn <- train(
    x = x_train, y = y_train, method = "knn", trControl = control,
    preProcess = preprocess_steps, tuneGrid = data.frame(k = 5)
  )
})

cat("Training linear SVM...\n")
set.seed(123)
time_svm <- system.time({
  model_svm <- train(
    x = x_train, y = y_train, method = "svmLinear", trControl = control,
    preProcess = preprocess_steps, tuneGrid = data.frame(C = 1), scaled = FALSE
  )
})

cat("Training Random Forest...\n")
set.seed(123)
time_rf <- system.time({
  model_rf <- train(
    x = x_train, y = y_train, method = "rf", trControl = control,
    preProcess = preprocess_steps,
    tuneGrid = data.frame(mtry = floor(sqrt(ncol(x_train)))),
    ntree = 500, importance = TRUE
  )
})

# -----------------------------------------------------------------------------
# 6. Evaluate predictions on the supplied held-out test set
# -----------------------------------------------------------------------------
evaluate_model <- function(model, new_data, truth, model_name, dataset) {
  probability <- predict(model, newdata = new_data, type = "prob")$Pulsar
  predicted <- factor(
    ifelse(probability >= 0.5, "Pulsar", "NonPulsar"),
    levels = levels(truth)
  )
  confusion <- confusionMatrix(predicted, truth, positive = "Pulsar")
  tp <- sum(predicted == "Pulsar" & truth == "Pulsar")
  fp <- sum(predicted == "Pulsar" & truth == "NonPulsar")
  fn <- sum(predicted == "NonPulsar" & truth == "Pulsar")
  roc_object <- pROC::roc(
    truth, probability, levels = c("NonPulsar", "Pulsar"),
    direction = "<", quiet = TRUE
  )
  metrics <- data.frame(
    Model = model_name,
    Dataset = dataset,
    Accuracy = unname(confusion$overall["Accuracy"]),
    Recall = unname(confusion$byClass["Sensitivity"]),
    Specificity = unname(confusion$byClass["Specificity"]),
    Precision = if (tp + fp == 0L) NA_real_ else tp / (tp + fp),
    F1 = if (2L * tp + fp + fn == 0L) NA_real_ else 2L * tp / (2L * tp + fp + fn),
    Balanced_Accuracy = unname(confusion$byClass["Balanced Accuracy"]),
    ROC_AUC = as.numeric(pROC::auc(roc_object)),
    TN = sum(predicted == "NonPulsar" & truth == "NonPulsar"),
    FP = fp,
    FN = fn,
    TP = tp
  )
  list(metrics = metrics, confusion = confusion, roc = roc_object,
       probability = probability, predicted = predicted)
}

models <- list(LDA = model_lda, CART = model_cart, kNN = model_knn,
               SVM = model_svm, RF = model_rf)
train_results <- Map(
  function(model, name) evaluate_model(model, x_train, y_train, name, "Training"),
  models, names(models)
)
test_results <- Map(
  function(model, name) evaluate_model(model, x_test, y_test, name, "Test"),
  models, names(models)
)
training_metrics <- do.call(rbind, lapply(train_results, `[[`, "metrics"))
test_metrics <- do.call(rbind, lapply(test_results, `[[`, "metrics"))
rownames(training_metrics) <- NULL
rownames(test_metrics) <- NULL

cat("\nHeld-out test results:\n")
print(test_metrics, digits = 4)
cat("Majority-class test accuracy:", mean(y_test == "NonPulsar"), "\n")

# -----------------------------------------------------------------------------
# 7. Save tables, predictions, models and comparison figures
# -----------------------------------------------------------------------------
metric_names <- c("Accuracy", "Recall", "Specificity", "Precision", "F1",
                  "Balanced_Accuracy", "ROC_AUC")
all_metrics <- rbind(training_metrics, test_metrics)
metric_long <- cbind(
  all_metrics[rep(seq_len(nrow(all_metrics)), times = length(metric_names)),
              c("Model", "Dataset")],
  stack(all_metrics[metric_names])
)
names(metric_long)[3:4] <- c("Score", "Metric")

score_plot <- ggplot(metric_long, aes(x = Model, y = Score, fill = Dataset)) +
  geom_col(position = "dodge", width = 0.75, na.rm = TRUE) +
  facet_wrap(~ Metric, ncol = 3) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Five-Model Comparison: Training and Test Performance",
    subtitle = "Training scores are in-sample; test scores use the held-out set"
  ) +
  theme_minimal()
ggsave(file.path(output_dir, "04_model_metric_comparison.png"), score_plot,
       width = 12, height = 10, dpi = 160)

roc_data <- do.call(rbind, lapply(names(test_results), function(name) {
  curve <- test_results[[name]]$roc
  data.frame(
    Model = name,
    False_Positive_Rate = 1 - curve$specificities,
    True_Positive_Rate = curve$sensitivities
  )
}))
roc_plot <- ggplot(roc_data, aes(False_Positive_Rate, True_Positive_Rate, color = Model)) +
  geom_path(linewidth = 0.8) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey60") +
  labs(title = "Test ROC Curves", x = "False Positive Rate", y = "True Positive Rate") +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  theme_minimal()
ggsave(file.path(output_dir, "05_test_roc_curves.png"), roc_plot,
       width = 9, height = 7, dpi = 160)

confusion_data <- do.call(rbind, lapply(names(test_results), function(name) {
  frame <- as.data.frame(test_results[[name]]$confusion$table)
  frame$Model <- name
  frame
}))
confusion_plot <- ggplot(confusion_data, aes(Reference, Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  facet_wrap(~ Model, ncol = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Test Confusion Matrices", x = "Actual Class", y = "Predicted Class") +
  theme_minimal()
ggsave(file.path(output_dir, "06_test_confusion_matrices.png"), confusion_plot,
       width = 12, height = 7, dpi = 160)

timing <- data.frame(
  Model = names(models),
  Seconds = c(time_lda[["elapsed"]], time_cart[["elapsed"]],
              time_knn[["elapsed"]], time_svm[["elapsed"]], time_rf[["elapsed"]])
)
timing_plot <- ggplot(timing, aes(Model, Seconds, fill = Model)) +
  geom_col(width = 0.65) +
  labs(title = "Model Fitting Time", y = "Elapsed Seconds") +
  theme_minimal() +
  theme(legend.position = "none")
ggsave(file.path(output_dir, "07_fitting_times.png"), timing_plot,
       width = 9, height = 5, dpi = 160)

test_predictions <- data.frame(Row = seq_len(nrow(test_data)), Actual = y_test)
for (name in names(test_results)) {
  test_predictions[[paste0(name, "_Probability")]] <- test_results[[name]]$probability
  test_predictions[[paste0(name, "_Prediction")]] <- test_results[[name]]$predicted
}

write.csv(training_metrics, file.path(output_dir, "training_metrics.csv"), row.names = FALSE)
write.csv(test_metrics, file.path(output_dir, "test_metrics.csv"), row.names = FALSE)
write.csv(test_predictions, file.path(output_dir, "test_predictions.csv"), row.names = FALSE)
write.csv(timing, file.path(output_dir, "fitting_times.csv"), row.names = FALSE)
saveRDS(models, file.path(output_dir, "part_a_models.rds"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))

stopifnot(nrow(test_metrics) == 5L, nrow(test_predictions) == nrow(test_data))
cat("\nAll five models completed successfully.\n")
cat("Results were saved to:", normalizePath(output_dir, winslash = "/"), "\n")
