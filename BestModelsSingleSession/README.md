# How to Reproduce the Results

This folder contains the final, cleaned-up set of scripts, built on a common
core.py.

## 1. Folder setup

Put every .py file in this folder in one place, then add your two data files
to that same folder:

```
FinalModels/
  core.py
  sensor_log.jsonl
  key_log.jsonl
  best_model.py
  confusion_matrix.py
  test_26key.py
  final_model_no_interp.py
```

core.py is not a script you run yourself. It is the shared library the other
four scripts import from (data loading, feature extraction, and the CNN model
definition). It expects sensor_log.jsonl and key_log.jsonl to sit in the same
folder as itself.

If you are testing a session, copy that session's
sensor_log.jsonl and key_log.jsonl into the folder before running.

## 2. Install dependencies

You need Python 3.9 or later. Install the required packages:

```
pip install numpy pandas scikit-learn xgboost scipy torch
```

xgboost is only needed for test_26key.py, and scipy is only needed for
final_model_no_interp.py, but it's simplest to just install all of them.

## 3. Running a script

From inside the folder, run any script directly, for example:

```
python best_model.py
```

Each script prints a baseline (random/majority-class) accuracy where
relevant, then per-fold results as it trains, then a final summary.

## 4. What each script does

- best_model.py: the final selected model. A large-capacity 1D CNN
  (CNN1D_Large from core.py) trained with data augmentation and label
  smoothing (0.1), evaluated with Stratified 5-fold cross-validation on the
  9-region label. 

- confusion_matrix.py: runs the same CNN setup as best_model.py, but instead
  of just printing accuracy, it collects predictions across all 5 folds and
  builds a full confusion matrix. It prints the matrix, per-region recall,
  and the 10 most-confused region pairs (noting whether the confusion is
  between regions in the same keyboard row/column). 

- test_26key.py: re-tests the setup at full 26-key resolution (individual
  keys, not 9 regions) instead of the usual region-level task. Runs both
  XGBoost (on the "combined" correlation+extended feature set) and the CNN,
  reporting both Top-1 and Top-3 accuracy for each, since distinguishing all
  26 keys is a much harder problem than the 9-region task.


## 7. Reproducibility notes

- All scripts fix SEED = 2133744, so results should be very close to
  identical between runs on the same machine.
- GPU vs CPU can cause tiny (usually negligible) differences in the CNN
  results due to floating-point non-determinism.
- best_model.py, confusion_matrix.py, and test_26key.py all use the same
  CNN1D_Large architecture, augmentation, and label smoothing, so their
  9-region CNN numbers should agree closely with each other. If they don't,
  check that all three are pointed at the same sensor_log.jsonl / key_log.jsonl.
