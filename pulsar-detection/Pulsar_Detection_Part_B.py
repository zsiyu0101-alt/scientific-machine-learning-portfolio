# %% [markdown]
# # Part B — Pulsar Detection with a Regularized Deep MLP
# ## Research question
# Can a multilayer perceptron improve pulsar detection over the five **R/caret** models in Part A on the **same supplied split**?
# 
# This notebook follows the instructor's workflow: environment setup, loading, summary, visualization, algorithm evaluation, and predictions. All 35 original predictors enter the model. We specify the hidden layers ourselves using scikit-learn's MLP implementation (Python is permitted in Part B).
# 
# **Protocol fixed before test evaluation:** six preprocessing/architecture candidates; five stratified training folds; average precision (AP) for model selection; inner validation for early stopping; a three-seed probability ensemble; an F1 threshold learned from training out-of-fold (OOF) predictions. Report both 0.5 and the frozen F1 threshold. No test-driven retraining or architecture selection is performed. OOF estimates used to select candidates are development results, not unbiased final performance estimates.
# 
# **References:** [3FGL catalog](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4yr_catalog/), [Fermi-LAT Collaboration (2015)](https://arxiv.org/abs/1501.02003), [Deep Learning with R](https://srdas.github.io/DLBook/DeepLearningWithR.html), [MLPClassifier documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html). The catalog paper describes the astronomical context; the supplied CSVs define this experiment. Non-pulsar predictions apply to the supplied comparison population, not every possible astronomical source class.

# %% [markdown]
# ## Step 1. Prepare the Python environment
# Run the cells in order. The submitted copy already contains executed results. If packages are missing, run `%pip install -r requirements.txt` in a separate notebook cell, restart the kernel, then run all cells. No TensorFlow, PyTorch, GPU, or R installation is needed to reproduce Part B; the archived Part A outputs were produced in R.

# %%
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('MPLBACKEND', 'Agg')
import importlib.util
import subprocess
import sys

# Install missing packages when the script is used in a fresh Python environment.
REQUIRED_PACKAGES = {
    'numpy': 'numpy', 'pandas': 'pandas', 'matplotlib': 'matplotlib',
    'seaborn': 'seaborn', 'sklearn': 'scikit-learn', 'scipy': 'scipy',
    'joblib': 'joblib', 'threadpoolctl': 'threadpoolctl'
}
missing = [pip_name for module, pip_name in REQUIRED_PACKAGES.items()
           if importlib.util.find_spec(module) is None]
if missing:
    print('Installing missing packages:', ', '.join(missing), flush=True)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

from pathlib import Path
from copy import deepcopy
import hashlib, json, platform, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import joblib
try:
    from IPython.display import display, Markdown
except ImportError:
    def display(*objects):
        for item in objects:
            print(item)
    class Markdown(str):
        pass
from threadpoolctl import threadpool_limits
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    log_loss, brier_score_loss, roc_curve, precision_recall_curve,
    classification_report, matthews_corrcoef)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from scipy.stats import binomtest

SEED = 2026
np.random.seed(SEED)
threadpool_limits(limits=1)
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def find_data_directory():
    """Find both supplied CSV files without requiring the working directory."""
    required = {'train_agnpsr_3fgl.csv', 'test_agnpsr_3fgl.csv'}
    configured = os.environ.get('PULSAR_DATA_DIR', '')
    candidates = [
        Path(configured) if configured else None,
        SCRIPT_DIR,
        SCRIPT_DIR / 'data',
        Path.cwd(),
        Path.cwd() / 'data',
        Path.home() / 'Desktop',
        Path.home() / 'Desktop' / 'Part_B_Pulsar_Detection' / 'data',
    ]
    for folder in candidates:
        if folder and all((folder / name).is_file() for name in required):
            return folder.resolve()
    raise FileNotFoundError(
        'Cannot find both CSV files. Put them beside this script, in a data '
        'subfolder, or set the PULSAR_DATA_DIR environment variable.'
    )

BASE = SCRIPT_DIR
DATA = find_data_directory()
OUT = BASE / 'Part_B_results'
FIG = OUT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid', context='notebook', palette='colorblind')
plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 180,
                     'axes.spines.top': False, 'axes.spines.right': False})
def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / f'{name}.png', bbox_inches='tight')
    plt.show()
    plt.close()

print('Python:', platform.python_version(), '| scikit-learn:', sklearn.__version__)

# %% [markdown]
# ## Step 2. Load and audit the supplied data
# The original split is preserved. Only training data are explored before model freezing. Missing/infinite values are imputed using medians fitted within each training partition. The supplied values are retained: some flux-related columns already contain negative transformed values, so blindly logging every feature is inappropriate.

# %%
train = pd.read_csv(DATA / 'train_agnpsr_3fgl.csv')
TARGET = 'pulsarness'
features = train.columns.drop(TARGET).tolist()
assert len(features) == 35
assert set(train[TARGET].unique()) == {'Pulsar', 'Non-Pulsar'}
X = train[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
y = train[TARGET].eq('Pulsar').astype(int).to_numpy()
assert not X.isna().all().any()
assert not X.duplicated().any(), 'Investigate duplicate sources before ordinary CV.'
hashes = {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
          for name in ['train_agnpsr_3fgl.csv', 'test_agnpsr_3fgl.csv']}
audit = pd.DataFrame({'dtype': X.dtypes.astype(str), 'missing': X.isna().sum(),
    'unique': X.nunique(), 'min': X.min(), 'median': X.median(), 'max': X.max()})
audit.to_csv(OUT / 'feature_audit.csv')
print('Training shape:', train.shape, '| Positive class: Pulsar = 1')
print(train[TARGET].value_counts().to_string())
print('Missing/nonfinite training entries:', int(X.isna().sum().sum()))
print('All predictors retained:', len(features))
display(train.head(), audit)

# %% [markdown]
# ## Step 3. Summarize and visualize the training data
# Class imbalance makes accuracy alone insufficient. AP measures ranking quality for the rare pulsars; ROC-AUC, precision, recall, F1, balanced accuracy, specificity, MCC and calibration are complementary. The correlation heatmap uses Spearman correlation because feature scales vary substantially. Correlation is not a reason to drop the first 34 columns.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = train[TARGET].value_counts().reindex(['Non-Pulsar', 'Pulsar'])
axes[0].bar(counts.index, counts.values, color=['#5470a8', '#e88b49'])
axes[0].bar_label(axes[0].containers[0])
axes[0].set(title='Training class counts', ylabel='Sources')
axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            colors=['#5470a8', '#e88b49'], startangle=90)
axes[1].set_title('Training class proportions')
savefig('01_class_distribution')

plt.figure(figsize=(14, 11))
sns.heatmap(X.corr(method='spearman'), cmap='vlag', center=0, vmin=-1, vmax=1,
            xticklabels=True, yticklabels=True, cbar_kws={'label': 'Spearman correlation'})
plt.xticks(fontsize=7); plt.yticks(fontsize=7)
plt.title('Training predictors: correlated measurements are retained')
savefig('02_feature_correlation')

# %%
selected_features = ['Spectral_Index', 'Variability_Index', 'Signif_Curve',
                     'GLAT', 'hr23', 'hr45']
long = train[selected_features + [TARGET]].melt(id_vars=TARGET,
    var_name='Feature', value_name='Value')
g = sns.displot(data=long, x='Value', hue=TARGET, col='Feature', col_wrap=3,
    kind='kde', common_norm=False, facet_kws={'sharex': False, 'sharey': False},
    height=2.7, fill=True, alpha=.3)
g.fig.suptitle('Training feature distributions by class', y=1.03)
sns.move_legend(g, 'upper left', bbox_to_anchor=(1.01, 1), title='Source class')
savefig('03_class_conditional_distributions')

# %% [markdown]
# ## Step 4. Specify the MLP architecture and training procedure
# The deeper candidate is **35 → 128 → 64 → 32 → 1**, with ReLU hidden activations and a sigmoid output. Other candidates test smaller networks, tanh, L2 regularization and preprocessing. These are fully connected feedforward neural networks trained with mini-batch Adam and binary cross-entropy. L2 regularization, early stopping, and seed averaging control overfitting on this small dataset; dropout and batch normalization are not claimed.
# 
# Quantile-to-normal transformation is compared with standard scaling. It supports negative inputs and reduces the influence of extreme values without removing columns. It is fitted only on the applicable training partition. No oversampling is used, preserving the observed class prior. Rare-class performance is addressed by AP selection and an OOF-selected classification threshold.
# 
# For each outer training fold, an **inner 80:20 stratified split** determines the epoch count using validation cross-entropy (patience 30; maximum 300 epochs). Preprocessing is fitted on the inner fitting subset. A fresh preprocessor and network then train on the whole outer training fold for that many epochs. The outer validation fold is untouched until scoring. The only substantive loops are needed for training, folds, seed averaging and repeated uncertainty/importance calculations.

# %%
CANDIDATES = [
    dict(name='Standard_64_32', transform='standard', layers=(64, 32), activation='relu', alpha=.01),
    dict(name='Quantile_64_32', transform='quantile', layers=(64, 32), activation='relu', alpha=.01),
    dict(name='Quantile_128_64_32', transform='quantile', layers=(128, 64, 32), activation='relu', alpha=.01),
    dict(name='Quantile_64_32_L2', transform='quantile', layers=(64, 32), activation='relu', alpha=.1),
    dict(name='Quantile_Tanh_64_32', transform='quantile', layers=(64, 32), activation='tanh', alpha=.01),
    dict(name='Quantile_32_16', transform='quantile', layers=(32, 16), activation='relu', alpha=.1),
]
MAX_EPOCHS, PATIENCE = 300, 30
FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))

def make_preprocessor(config, n):
    scaler = (StandardScaler() if config['transform'] == 'standard' else
        QuantileTransformer(n_quantiles=min(200, n), output_distribution='normal',
                            subsample=100000, random_state=SEED))
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                     ('scale', scaler)])

def make_network(config, seed):
    return MLPClassifier(hidden_layer_sizes=config['layers'], activation=config['activation'],
        solver='adam', alpha=config['alpha'], learning_rate_init=.001,
        batch_size=64, shuffle=True, random_state=seed, early_stopping=False)

def train_with_inner_stopping(config, X_fit, y_fit, seed, keep_history=False):
    inner_fit, inner_valid = train_test_split(np.arange(len(y_fit)), test_size=.2,
        stratify=y_fit, random_state=seed)
    prep = make_preprocessor(config, len(inner_fit))
    a = prep.fit_transform(X_fit.iloc[inner_fit])
    b = prep.transform(X_fit.iloc[inner_valid])
    net = make_network(config, seed)
    best_loss, best_epoch, stale, history = np.inf, 1, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        net.partial_fit(a, y_fit[inner_fit], classes=np.array([0, 1]))
        pa, pb = net.predict_proba(a)[:, 1], net.predict_proba(b)[:, 1]
        loss = log_loss(y_fit[inner_valid], pb, labels=[0, 1])
        if keep_history:
            history.append(dict(epoch=epoch, train_loss=log_loss(y_fit[inner_fit], pa),
                validation_loss=loss, train_ap=average_precision_score(y_fit[inner_fit], pa),
                validation_ap=average_precision_score(y_fit[inner_valid], pb)))
        if loss < best_loss - .0001:
            best_loss, best_epoch, stale = loss, epoch, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    # Refit from scratch on the complete fitting partition, using the frozen epoch count.
    prep = make_preprocessor(config, len(y_fit))
    a = prep.fit_transform(X_fit)
    net = make_network(config, seed)
    for _ in range(best_epoch):
        net.partial_fit(a, y_fit, classes=np.array([0, 1]))
    return Pipeline([('preprocess', prep), ('mlp', net)]), best_epoch, pd.DataFrame(history)

def evaluate(y_true, probability, threshold=.5):
    pred = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y_true, pred), Recall=recall_score(y_true, pred),
        Specificity=tn/(tn+fp), Precision=precision_score(y_true, pred, zero_division=0),
        F1=f1_score(y_true, pred, zero_division=0),
        Balanced_Accuracy=balanced_accuracy_score(y_true, pred),
        ROC_AUC=roc_auc_score(y_true, probability),
        Average_Precision=average_precision_score(y_true, probability),
        MCC=matthews_corrcoef(y_true, pred), Brier=brier_score_loss(y_true, probability),
        Log_Loss=log_loss(y_true, probability), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

display(pd.DataFrame(CANDIDATES))

# %% [markdown]
# ## Step 5. Evaluate candidate networks using training data only
# The candidate with the highest mean outer-fold AP is selected; ties use ROC-AUC. Each candidate receives the same folds, stopping rule, and seed schedule. This is a modest, declared search rather than repeated test-set experimentation.

# %%
start = time.perf_counter()
cv_rows, candidate_oof = [], {}
for config in CANDIDATES:
    oof = np.zeros(len(y))
    for fold, (fit_idx, valid_idx) in enumerate(FOLDS, 1):
        model, epochs, _ = train_with_inner_stopping(config, X.iloc[fit_idx],
            y[fit_idx], SEED + fold)
        p = model.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = p
        cv_rows.append(dict(Candidate=config['name'], Fold=fold, Epochs=epochs,
            **evaluate(y[valid_idx], p)))
    candidate_oof[config['name']] = oof
    print(config['name'], '| OOF AP:', round(average_precision_score(y, oof), 4), flush=True)
cv = pd.DataFrame(cv_rows)
ranking = cv.groupby('Candidate').agg(AP_mean=('Average_Precision', 'mean'),
    AP_sd=('Average_Precision', 'std'), AUC_mean=('ROC_AUC', 'mean'),
    F1_mean=('F1', 'mean'), Epochs_median=('Epochs', 'median')).sort_values(
        ['AP_mean', 'AUC_mean'], ascending=False)
best = next(c for c in CANDIDATES if c['name'] == ranking.index[0])
cv.to_csv(OUT / 'cross_validation_folds.csv', index=False)
ranking.to_csv(OUT / 'candidate_ranking.csv')
print('Selected using training data:', best)
print('Candidate-search seconds:', round(time.perf_counter() - start, 1))
display(ranking.round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=cv, y='Candidate', x='Average_Precision', order=ranking.index,
            ax=axes[0], color='#9dc9df')
sns.stripplot(data=cv, y='Candidate', x='Average_Precision', order=ranking.index,
              ax=axes[0], color='#243b53', size=5)
axes[0].set_title('Five-fold development AP (not test performance)')
sns.barplot(data=cv, y='Candidate', x='Epochs', order=ranking.index,
            ax=axes[1], errorbar='sd', color='#e5af77')
axes[1].set_title('Inner-selected epochs: mean and standard deviation')
savefig('04_candidate_selection')

# %% [markdown]
# ## Step 6. Stabilize the selected network and freeze the decision threshold
# Three independently initialized MLPs are averaged, not stacked with any baseline model. OOF probabilities are generated by the same three-seed procedure used for the final ensemble. The threshold maximizes OOF F1 on the prespecified grid 0.05–0.95 (step 0.01); ties favor the value nearest 0.5. This operating point balances precision and recall rather than maximizing accuracy. Also report the default threshold for comparability with Part A.
# 
# The selected configuration's OOF scores still contain candidate-selection optimism. They are used for development and threshold selection only. Final test results remain the performance estimate. Raw sigmoid scores are not assumed to be calibrated probabilities.

# %%
ENSEMBLE_SEEDS = [2026, 2027, 2028]
ensemble_oof = np.zeros((len(y), len(ENSEMBLE_SEEDS)))
epoch_rows = []
for fold, (fit_idx, valid_idx) in enumerate(FOLDS, 1):
    for j, seed in enumerate(ENSEMBLE_SEEDS):
        model, epochs, _ = train_with_inner_stopping(best, X.iloc[fit_idx], y[fit_idx], seed)
        ensemble_oof[valid_idx, j] = model.predict_proba(X.iloc[valid_idx])[:, 1]
        epoch_rows.append(dict(Fold=fold, Seed=seed, Epochs=epochs))
    print('Ensemble OOF fold completed:', fold, flush=True)
oof_p = ensemble_oof.mean(axis=1)
thresholds = np.round(np.arange(.05, .951, .01), 2)
threshold_table = pd.DataFrame([dict(Threshold=float(t), **evaluate(y, oof_p, t))
                               for t in thresholds])
threshold_table['Distance_to_0_5'] = abs(threshold_table.Threshold - .5)
threshold = float(threshold_table.sort_values(['F1', 'Distance_to_0_5'],
    ascending=[False, True]).iloc[0].Threshold)
threshold_table.to_csv(OUT / 'oof_threshold_analysis.csv', index=False)
pd.DataFrame(epoch_rows).to_csv(OUT / 'oof_epochs.csv', index=False)
pd.DataFrame({'Row': np.arange(1, len(y)+1), 'Actual': y, 'OOF_Probability': oof_p,
              'OOF_Prediction': (oof_p >= threshold).astype(int)}).to_csv(
                  OUT / 'training_oof_predictions.csv', index=False)
print('Frozen threshold:', threshold)
display(pd.DataFrame([evaluate(y, oof_p, .5), evaluate(y, oof_p, threshold)],
    index=['OOF default 0.5', 'OOF selected threshold']).round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
threshold_table.plot(x='Threshold', y=['Precision', 'Recall', 'F1', 'Accuracy'], ax=axes[0])
axes[0].axvline(threshold, ls='--', color='black', label=f'Frozen threshold = {threshold:.2f}')
axes[0].set(title='Training OOF operating-point selection', ylabel='Score', ylim=(0, 1.03))
axes[0].legend(fontsize=8)
pr, re, _ = precision_recall_curve(y, oof_p)
axes[1].plot(re, pr, label=f'OOF AP = {average_precision_score(y, oof_p):.3f}')
axes[1].axhline(y.mean(), ls=':', color='grey', label='Training prevalence')
axes[1].set(title='Selected ensemble: development PR curve', xlabel='Recall', ylabel='Precision')
axes[1].legend()
savefig('05_oof_threshold_and_pr')

# %% [markdown]
# ## Step 7. Train the final three-seed MLP ensemble
# Each member chooses its epoch count using an internal training-only validation split, then refits on all 1,333 training observations. The learning curves below describe those internal early-stopping runs; they are not test curves or curves from the full-data refit.

# %%
final_models, histories, final_epochs = [], [], []
for seed in ENSEMBLE_SEEDS:
    model, epochs, history = train_with_inner_stopping(best, X, y, seed, keep_history=True)
    final_models.append(model)
    histories.append(history.assign(Seed=seed))
    final_epochs.append(epochs)
    print('Final seed:', seed, '| Selected epochs:', epochs, flush=True)
history = pd.concat(histories, ignore_index=True)
history.to_csv(OUT / 'learning_history.csv', index=False)
def ensemble_probability(frame):
    return np.column_stack([m.predict_proba(frame)[:, 1] for m in final_models]).mean(axis=1)

frozen = dict(configuration=best, threshold=threshold, seeds=ENSEMBLE_SEEDS,
    final_epochs=final_epochs, positive_class='Pulsar', features=features,
    file_sha256=hashes, selection_metric='Mean five-fold average precision',
    stopping_metric='Inner validation log loss', max_epochs=MAX_EPOCHS, patience=PATIENCE)
(OUT / 'frozen_protocol.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
joblib.dump(dict(models=final_models, threshold=threshold, features=features,
    positive_class='Pulsar', protocol=frozen), OUT / 'mlp_ensemble.joblib')
print('Configuration, preprocessing, weights, and threshold saved before test evaluation.')

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for h, seed, epochs in zip(histories, ENSEMBLE_SEEDS, final_epochs):
    axes[0].plot(h.epoch, h.train_loss, alpha=.65, label=f'Train seed {seed}')
    axes[0].plot(h.epoch, h.validation_loss, ls='--', label=f'Validation seed {seed}')
    axes[0].axvline(epochs, ls=':', alpha=.35)
    axes[1].plot(h.epoch, h.train_ap, alpha=.65, label=f'Train seed {seed}')
    axes[1].plot(h.epoch, h.validation_ap, ls='--', label=f'Validation seed {seed}')
axes[0].set(title='Internal stopping runs: cross-entropy', xlabel='Epoch', ylabel='Log loss')
axes[1].set(title='Internal stopping runs: average precision', xlabel='Epoch', ylabel='AP')
axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
savefig('06_learning_curves')

sizes = [len(features), *best['layers'], 1]
fig, ax = plt.subplots(figsize=(11, 4))
for i, size in enumerate(sizes):
    ax.text(i, .5, f'{size} units', ha='center', va='center', fontsize=13,
        bbox=dict(boxstyle='round,pad=.8', facecolor='#d9e9f3', edgecolor='#5470a8'))
    if i < len(sizes)-1:
        ax.annotate('', xy=(i+.7,.5), xytext=(i+.3,.5), arrowprops={'arrowstyle':'->'})
    ax.text(i, .2, 'Input' if i == 0 else ('Sigmoid' if i == len(sizes)-1 else best['activation']),
            ha='center')
parameters = sum(w.size+b.size for w,b in zip(final_models[0]['mlp'].coefs_, final_models[0]['mlp'].intercepts_))
ax.set(xlim=(-.6, len(sizes)-.4), ylim=(0,1))
ax.set_title(f'Selected MLP: {parameters:,} trainable parameters per member; three-seed probability average')
ax.axis('off')
savefig('07_selected_architecture')

# %% [markdown]
# ## Step 8. Make final test predictions and compare with the R baselines
# This is the first use of test labels for modeling/evaluation in this notebook. The supplied Part A predictions are imported, not replaced by Python implementations. Their row labels and metrics are independently checked. Exact duplicate feature rows across train and test would invalidate the experiment and trigger an assertion. Source identifiers are unavailable, so a physical-source overlap audit cannot be guaranteed.

# %%
test = pd.read_csv(DATA / 'test_agnpsr_3fgl.csv')
assert set(test.columns) == set(train.columns)
assert set(test[TARGET].unique()) == {'Pulsar', 'Non-Pulsar'}
X_test = test[features].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
y_test = test[TARGET].eq('Pulsar').astype(int).to_numpy()
assert not pd.util.hash_pandas_object(X, index=False).isin(
    pd.util.hash_pandas_object(X_test, index=False)).any(), 'Train/test overlap detected.'
def find_part_a_results():
    """Find Part A predictions made by the R script or the supplied archive."""
    candidates = [
        SCRIPT_DIR / 'Part_A_results',
        SCRIPT_DIR / 'part_a_reference' / 'Part_A_results',
        DATA.parent / 'Part_A_results',
        Path.home() / 'Desktop' / 'Part_B_Pulsar_Detection' /
            'part_a_reference' / 'Part_A_results',
    ]
    required = {'test_predictions.csv', 'test_metrics.csv'}
    for folder in candidates:
        if all((folder / name).is_file() for name in required):
            return folder.resolve()
    raise FileNotFoundError(
        'Part A prediction files were not found. Run Pulsar_Detection_Part_A.R '
        'first. It creates Part_A_results beside the R script; then run this '
        'Python file from the same folder.'
    )

PART_A_RESULTS = find_part_a_results()
baseline = pd.read_csv(PART_A_RESULTS / 'test_predictions.csv')
baseline_metrics = pd.read_csv(PART_A_RESULTS / 'test_metrics.csv')
assert np.array_equal(baseline.Row.to_numpy(), np.arange(1, len(test)+1))
assert np.array_equal(baseline.Actual.eq('Pulsar').to_numpy(), y_test.astype(bool))
BASELINES = ['LDA', 'CART', 'kNN', 'SVM', 'RF']
probabilities = {name: baseline[f'{name}_Probability'].to_numpy() for name in BASELINES}
rows = []
for name in BASELINES:
    scores = evaluate(y_test, probabilities[name])
    archived = baseline_metrics.set_index('Model').loc[name]
    assert np.array_equal(baseline[f'{name}_Prediction'].eq('Pulsar').to_numpy(), probabilities[name] >= .5)
    assert np.allclose([scores[k] for k in ['Accuracy','Recall','F1','ROC_AUC']],
        archived[['Accuracy','Recall','F1','ROC_AUC']].to_numpy(dtype=float))
    rows.append(dict(Model=name, Threshold=.5, **scores))
test_member_p = np.column_stack([m.predict_proba(X_test)[:, 1] for m in final_models])
test_p = test_member_p.mean(axis=1)
probabilities['MLP'] = test_p
rows.extend([dict(Model='MLP (0.5)', Threshold=.5, **evaluate(y_test, test_p, .5)),
    dict(Model='MLP (OOF threshold)', Threshold=threshold, **evaluate(y_test, test_p, threshold))])
results = pd.DataFrame(rows).set_index('Model')
results.to_csv(OUT / 'test_metrics_comparison.csv')
test_pred = (test_p >= threshold).astype(int)
prediction_table = pd.DataFrame({'Row': np.arange(1, len(test)+1), 'Actual': test[TARGET],
    'MLP_Probability': test_p, 'MLP_Seed_SD': test_member_p.std(axis=1),
    'MLP_Prediction_0_5': np.where(test_p >= .5, 'Pulsar', 'Non-Pulsar'),
    'MLP_Prediction_OOF_Threshold': np.where(test_pred, 'Pulsar', 'Non-Pulsar'),
    'Correct': test_pred == y_test})
prediction_table.to_csv(OUT / 'test_predictions.csv', index=False)
print('Test sources:', len(test), '| Pulsars:', y_test.sum())
print('Majority-class test accuracy:', round(1-y_test.mean(), 4))
display(results.round(4))
print(classification_report(y_test, test_pred, target_names=['Non-Pulsar', 'Pulsar'], digits=4))

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, label, p, t in zip(axes, ['R linear SVM', 'MLP: threshold 0.5', f'MLP: threshold {threshold:.2f}'],
        [probabilities['SVM'], test_p, test_p], [.5, .5, threshold]):
    cm = confusion_matrix(y_test, p >= t, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
        xticklabels=['Non-Pulsar', 'Pulsar'], yticklabels=['Non-Pulsar', 'Pulsar'])
    ax.set(title=label, xlabel='Predicted class', ylabel='True class')
savefig('08_confusion_matrices')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for name, p in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    precision, recall, _ = precision_recall_curve(y_test, p)
    axes[0].plot(fpr, tpr, lw=2 if name == 'MLP' else 1.2,
        label=f'{name}: {roc_auc_score(y_test, p):.3f}')
    axes[1].step(recall, precision, where='post', lw=2 if name == 'MLP' else 1.2,
        label=f'{name}: {average_precision_score(y_test, p):.3f}')
axes[0].plot([0,1], [0,1], 'k:', alpha=.5)
axes[1].axhline(y_test.mean(), color='grey', ls=':', label='Test prevalence')
axes[0].set(title='Held-out test ROC curves (AUC)', xlabel='False-positive rate', ylabel='Pulsar recall')
axes[1].set(title='Held-out test precision–recall curves (AP)', xlabel='Recall', ylabel='Precision')
axes[0].legend(); axes[1].legend()
savefig('09_roc_and_pr_curves')

# %%
metric_names = ['Accuracy', 'Recall', 'Precision', 'F1', 'Balanced_Accuracy', 'ROC_AUC', 'Average_Precision']
plt.figure(figsize=(12, 6))
sns.heatmap(results[metric_names], annot=True, fmt='.3f', cmap='YlGnBu', vmin=.7, vmax=1,
    cbar_kws={'label':'Metric value (color scale starts at 0.7)'})
plt.title('Same-split test comparison: original R baselines versus MLP')
savefig('10_metric_comparison')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name in ['SVM', 'RF', 'MLP']:
    fraction, mean_p = calibration_curve(y_test, probabilities[name], n_bins=7, strategy='quantile')
    axes[0].plot(mean_p, fraction, 'o-', label=f'{name}, Brier={brier_score_loss(y_test, probabilities[name]):.3f}')
axes[0].plot([0,1], [0,1], 'k:')
axes[0].set(title='Reliability diagram (quantile bins)', xlabel='Mean predicted pulsar score',
            ylabel='Observed pulsar fraction', xlim=(0,1), ylim=(0,1))
axes[0].legend(fontsize=8)
sns.histplot(x=test_p, hue=test[TARGET].to_numpy(), bins=25, element='step',
             stat='density', common_norm=False, ax=axes[1])
axes[1].axvline(threshold, color='black', ls='--')
axes[1].set(title='Held-out test score distributions', xlabel='MLP pulsar score')
savefig('11_calibration_and_scores')

# %% [markdown]
# ## Step 9. Quantify uncertainty and inspect errors
# Stratified paired bootstrap resampling keeps the observed positive/negative counts fixed. Percentile 95% intervals quantify sampling uncertainty **conditional on this fitted model and split**, not training-process uncertainty. Use the same sampled rows for MLP–SVM differences. The exact McNemar test compares paired classification errors with the Part A SVM (the strongest archived accuracy baseline). A positive point difference alone does not establish a statistically reliable improvement. McNemar is unadjusted and exploratory because the baseline was identified from archived test performance.

# %%
rng = np.random.default_rng(SEED)
positive, negative = np.flatnonzero(y_test == 1), np.flatnonzero(y_test == 0)
bootstrap_rows = []
for _ in range(1000):
    idx = np.concatenate([rng.choice(positive, len(positive), replace=True),
                          rng.choice(negative, len(negative), replace=True)])
    a = evaluate(y_test[idx], test_p[idx], threshold)
    b = evaluate(y_test[idx], probabilities['SVM'][idx], .5)
    bootstrap_rows.append({**{k:a[k] for k in ['Accuracy','Recall','Precision','F1','ROC_AUC','Average_Precision']},
        **{f'Delta_{k}':a[k]-b[k] for k in ['Accuracy','F1','ROC_AUC','Average_Precision']}})
bootstrap = pd.DataFrame(bootstrap_rows)
intervals = bootstrap.quantile([.025,.975]).T.rename(columns={.025:'Lower_95',.975:'Upper_95'})
mlp_scores = evaluate(y_test, test_p, threshold)
svm_scores = evaluate(y_test, probabilities['SVM'], .5)
intervals['Point'] = [mlp_scores[k] if not k.startswith('Delta_') else
    mlp_scores[k[6:]]-svm_scores[k[6:]] for k in intervals.index]
intervals.to_csv(OUT / 'bootstrap_confidence_intervals.csv')
svm_correct = (probabilities['SVM'] >= .5) == y_test
mlp_correct = test_pred == y_test
b = int(np.sum(svm_correct & ~mlp_correct))
c = int(np.sum(~svm_correct & mlp_correct))
mcnemar_p = binomtest(min(b,c), b+c, .5).pvalue if b+c else 1.0
print(f'SVM-only correct: {b}; MLP-only correct: {c}; exact McNemar p = {mcnemar_p:.4f}')
display(intervals.round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ci = intervals.loc[['Accuracy','Recall','Precision','F1','ROC_AUC','Average_Precision']]
axes[0].errorbar(ci.Point, np.arange(len(ci)), xerr=np.vstack([
    np.maximum(0,ci.Point-ci.Lower_95), np.maximum(0,ci.Upper_95-ci.Point)]), fmt='o', capsize=4)
axes[0].set_yticks(np.arange(len(ci)), ci.index)
axes[0].set(title='MLP test metrics: stratified bootstrap 95% intervals', xlabel='Score')
ci = intervals.loc[['Delta_Accuracy','Delta_F1','Delta_ROC_AUC','Delta_Average_Precision']]
axes[1].errorbar(ci.Point, np.arange(len(ci)), xerr=np.vstack([
    np.maximum(0,ci.Point-ci.Lower_95), np.maximum(0,ci.Upper_95-ci.Point)]), fmt='o', capsize=4)
axes[1].set_yticks(np.arange(len(ci)), ci.index.str.replace('Delta_', ''))
axes[1].axvline(0, color='grey', ls=':')
axes[1].set(title='Paired MLP − R SVM difference: 95% intervals', xlabel='Difference')
savefig('12_bootstrap_intervals')

errors = prediction_table.loc[~prediction_table.Correct].copy()
errors['Error_type'] = np.where(errors.Actual.eq('Pulsar'), 'Missed pulsar', 'False alarm')
errors.to_csv(OUT / 'misclassified_test_sources.csv', index=False)
display(errors.sort_values('MLP_Probability', ascending=False))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(test_p, test_member_p.std(axis=1), c=np.where(test_pred == y_test, '#5470a8', '#d94e41'),
                alpha=.6, s=22)
axes[0].axvline(threshold, ls='--', color='black')
axes[0].set(title='Seed disagreement: errors shown in red', xlabel='Mean pulsar score',
            ylabel='Score standard deviation across three seeds')
order = np.argsort(test_p)[::-1]
cumulative_recall = np.cumsum(y_test[order]) / y_test.sum()
fraction_reviewed = np.arange(1,len(y_test)+1) / len(y_test)
axes[1].plot(fraction_reviewed, cumulative_recall, label='MLP ranked source review')
axes[1].plot([0,1],[0,1], 'k:', label='Random ranking')
axes[1].set(title='Cumulative pulsar discovery curve', xlabel='Fraction of sources reviewed',
            ylabel='Fraction of pulsars recovered')
axes[1].legend()
savefig('13_uncertainty_and_discovery')

# %% [markdown]
# ## Step 10. Explain the model and visualize its representation
# Permutation importance measures the decrease in test AP after shuffling one original feature. This is **post-hoc analysis only** and does not trigger feature selection or model refitting. Correlated features can share or mask importance; negative values indicate sampling variability or redundancy. The 2-D hidden-representation PCA is fitted on training activations only and then applied to test activations. A clear picture is descriptive, not independent validation.

# %%
rng = np.random.default_rng(SEED)
reference_ap = average_precision_score(y_test, test_p)
importance_rows = []
for feature in features:
    decreases = []
    for repeat in range(10):
        permuted = X_test.copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        decreases.append(reference_ap - average_precision_score(y_test, ensemble_probability(permuted)))
    importance_rows.append(dict(Feature=feature, Mean_AP_Decrease=np.mean(decreases),
                                SD_AP_Decrease=np.std(decreases, ddof=1)))
importance = pd.DataFrame(importance_rows).sort_values('Mean_AP_Decrease', ascending=False)
importance.to_csv(OUT / 'permutation_importance.csv', index=False)
top = importance.head(15).iloc[::-1]
plt.figure(figsize=(10, 6))
plt.barh(top.Feature, top.Mean_AP_Decrease, xerr=top.SD_AP_Decrease, capsize=3, color='#649bb7')
plt.axvline(0, color='black', lw=.8)
plt.title('Test permutation importance: mean ± SD across 10 permutations')
plt.xlabel('Decrease in average precision (not causal importance)')
savefig('14_permutation_importance')

# %%
def hidden_representation(model, frame):
    h = model['preprocess'].transform(frame)
    network = model['mlp']
    for weights, bias in zip(network.coefs_[:-1], network.intercepts_[:-1]):
        h = h @ weights + bias
        h = np.maximum(h, 0) if network.activation == 'relu' else np.tanh(h)
    return h

hidden_train = hidden_representation(final_models[0], X)
hidden_test = hidden_representation(final_models[0], X_test)
pca = PCA(n_components=2, random_state=SEED).fit(hidden_train)
z_train, z_test = pca.transform(hidden_train), pca.transform(hidden_test)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, z, labels, title in zip(axes, [z_train,z_test], [y,y_test], ['Training sources','Held-out test sources']):
    sns.scatterplot(x=z[:,0], y=z[:,1], hue=np.where(labels, 'Pulsar', 'Non-Pulsar'),
                    alpha=.65, s=25, ax=ax)
    ax.set(title=title, xlabel='Hidden representation PC1', ylabel='Hidden representation PC2')
axes[1].scatter(z_test[~mlp_correct,0], z_test[~mlp_correct,1], s=75,
                facecolors='none', edgecolors='red', label='Misclassified')
axes[1].legend(fontsize=8)
fig.suptitle(f'Last hidden layer of seed {ENSEMBLE_SEEDS[0]}; training PCA explains {pca.explained_variance_ratio_.sum():.1%} variance')
savefig('15_hidden_representation')

# %% [markdown]
# ## Findings and conclusion
# The following English interpretation is computed from the actual run. It distinguishes changes in accuracy, rare-class detection, ranking quality and uncertainty. A more complex neural network is not assumed to outperform every classical model. Threshold selection can improve recall/F1 while reducing precision; ranking metrics do not change when only the threshold changes.

# %%
a, b = mlp_scores, svm_scores
delta_accuracy = 100*(a['Accuracy']-b['Accuracy'])
delta_f1 = a['F1']-b['F1']
conclusion = f"""# Part B — Results and findings

## Data and method
The original split contains {len(y):,} training sources ({y.sum()} pulsars) and {len(y_test):,} test sources ({y_test.sum()} pulsars). All {len(features)} predictors were retained. Training-only median imputation and scaling/quantile transformation prevent leakage. The selected network was **{best['name']}**, with hidden layers **{best['layers']}**, **{best['activation']}** activation, L2 alpha **{best['alpha']}**, mini-batch Adam and inner-validation early stopping. Each member has {parameters:,} parameters. Three seeds were averaged; final epoch counts were {final_epochs}.

Six declared candidates were compared using five-fold training AP. The final classification threshold **{threshold:.2f}** maximized training OOF F1; 0.5 was also evaluated. Candidate selection and threshold tuning make development OOF results optimistic. Test labels were used only after freezing the model. The five baseline predictions came from the supplied R/caret Part A archive and were independently verified against its metrics.

## Held-out findings
At the frozen threshold, the MLP correctly identified **{a['TP']} of {y_test.sum()} pulsars**, missed **{a['FN']}**, and produced **{a['FP']} false alarms**. Accuracy was **{a['Accuracy']:.2%}**, pulsar recall **{a['Recall']:.2%}**, precision **{a['Precision']:.2%}**, F1 **{a['F1']:.4f}**, balanced accuracy **{a['Balanced_Accuracy']:.4f}**, ROC-AUC **{a['ROC_AUC']:.4f}**, and AP **{a['Average_Precision']:.4f}**. The majority-class accuracy was **{1-y_test.mean():.2%}**, which shows why accuracy must be interpreted alongside pulsar recall.

The archived linear SVM obtained accuracy **{b['Accuracy']:.2%}**, recall **{b['Recall']:.2%}**, F1 **{b['F1']:.4f}**, ROC-AUC **{b['ROC_AUC']:.4f}** and AP **{b['Average_Precision']:.4f}**. MLP minus SVM accuracy was **{delta_accuracy:+.2f} percentage points** and F1 was **{delta_f1:+.4f}**. At the common default 0.5 threshold, MLP accuracy was **{results.loc['MLP (0.5)','Accuracy']:.2%}** and F1 was **{results.loc['MLP (0.5)','F1']:.4f}**; this separates network behavior from operating-point changes.

The paired bootstrap 95% interval for the accuracy difference was **[{intervals.loc['Delta_Accuracy','Lower_95']:+.4f}, {intervals.loc['Delta_Accuracy','Upper_95']:+.4f}]**; for F1 it was **[{intervals.loc['Delta_F1','Lower_95']:+.4f}, {intervals.loc['Delta_F1','Upper_95']:+.4f}]**. Exact McNemar p was **{mcnemar_p:.4f}**. Point differences should not be described as conclusive superiority if the corresponding intervals include zero.

## Interpretation and limitations
Regularization, preprocessing and seed averaging allow a nonlinear MLP to model interactions without dropping the early catalog features. Lowering a threshold can recover additional pulsars at the cost of more false alarms, whereas AP and ROC-AUC evaluate ranking independently of that operating point. The most influential shuffled feature was **{importance.iloc[0].Feature}**; correlated inputs limit individual importance interpretation. Reliability plots and Brier score assess probability quality, and seed disagreement highlights unstable predictions without constituting a calibrated uncertainty interval.

This is a small, imbalanced, single fixed-split study with only {y_test.sum()} test pulsars. Baselines retain the instructor's R implementations and fixed settings, while MLP receives a larger development search; this is a same-data comparison, not an equal-tuning-budget benchmark. Bootstrap intervals condition on the observed split and fitted weights. No source identifiers were supplied, so exact-row overlap checks cannot establish full physical-source independence. Future work should use repeated nested validation, calibration on separate training-only data, and a new external catalog sample. No further network changes were made after viewing test results.

## Conclusion
The experiment demonstrates a reproducible MLP approach and quantifies both benefits and trade-offs relative to all five R baselines. The numerical table, confusion matrices, PR/ROC curves and paired uncertainty estimates determine which performance claims are supported; network complexity alone does not establish improvement.

## References
- Fermi-LAT Collaboration (2015), Fermi Large Area Telescope Third Source Catalog: https://arxiv.org/abs/1501.02003
- NASA 3FGL catalog: https://fermi.gsfc.nasa.gov/ssc/data/access/lat/4yr_catalog/
- Sanjiv R. Das, Deep Learning with R: https://srdas.github.io/DLBook/DeepLearningWithR.html
- scikit-learn 1.5 MLPClassifier: https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html
"""
(OUT / 'Findings_and_Conclusion.md').write_text(conclusion, encoding='utf-8')
display(Markdown(conclusion))

# %% [markdown]
# ## Reproducibility and saved outputs
# The joblib artifact contains the three fitted preprocessing pipelines, weights, feature order and frozen threshold. Load it only from a trusted source. To predict new unlabeled rows, supply the same 35 predictor columns, replace nonfinite values with missing values, average the three member probabilities, and apply the saved threshold. All figures are exported as high-resolution PNG files, and numerical results and per-source predictions are CSV files.

# %%
versions = {name: __import__(name).__version__ for name in
    ['numpy','pandas','sklearn','scipy','matplotlib','seaborn','joblib']}
verification = dict(status='passed', training_rows=len(y), test_rows=len(y_test),
    original_predictors=len(features), training_positives=int(y.sum()), test_positives=int(y_test.sum()),
    figures=len(list(FIG.glob('*.png'))), test_predictions=len(prediction_table),
    baseline_metrics_verified=True, exact_feature_row_overlap=False,
    test_used_for_model_selection=False, file_sha256=hashes, versions=versions,
    frozen_threshold=threshold, selected_configuration=best, final_epochs=final_epochs)
assert len(prediction_table) == 571 and np.isfinite(test_p).all()
assert len(list(FIG.glob('*.png'))) == 15
saved = joblib.load(OUT / 'mlp_ensemble.joblib')
assert np.allclose(np.column_stack([m.predict_proba(X_test)[:,1] for m in saved['models']]).mean(axis=1), test_p)
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
(OUT / 'environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
print('Verified:', json.dumps(verification, indent=2))
print('Outputs:', OUT.resolve())
