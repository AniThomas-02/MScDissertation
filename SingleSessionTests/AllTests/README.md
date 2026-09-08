# Reproducibility Instructions

This folder contains standalone Python scripts for a keystroke/region classification
project. Each script loads sensor and keypress data, builds a labelled dataset, trains
one or more models, and prints accuracy results to the console.

## 1. Folder setup

Put every .py file from AllTests in the same folder, then add your three data files
to that same folder:

```
AllTests/
  core.py
  sensor_log.jsonl
  key_log.jsonl
  event_log.jso
  cnn_variant_a.py
  cnn_variant_b.py
  ... (all the other scripts)
```

core.py is not a script you run yourself. It is the shared library that all the
other scripts import from (data loading, feature extraction, model definitions).
It expects sensor_log.jsonl and key_log.jsonl to sit in the same folder as itself.

If you are testing a session, copy that session's
sensor_log.jsonl, event_log.jsonl and key_log.jsonl into the folder before running.

## 2. Install dependencies

You need Python 3.9 or later. Install the required packages:

```
pip install numpy pandas scikit-learn xgboost lightgbm torch
```

If you do not have a GPU, plain pip install torch is fine, the scripts fall back
to CPU automatically.

## 3. Running a script

From inside the folder, run any script directly, for example:

```
python cnn_variant_a.py
```

Each script prints:
- a baseline (random/majority-class) accuracy for comparison
- per-fold accuracy as it trains
- a final mean accuracy plus/minus standard deviation across folds

## 4. What each script does

Basic classical models and features
- all_features.py: builds the full "kitchen sink" feature set (basic, touchlogger-style,
  correlation, and extended features) and runs Random Forest / XGBoost, then prints
  the top 20 most important features from each.
- curated_features.py: uses a hand-picked subset of 24 features (the top features
  found by all_features.py) and re-tests Random Forest / XGBoost on just those.
- new_classical_models.py: tests additional classical models (QDA, ExtraTrees,
  LightGBM) using the curated feature set. Requires the fix in section 3 above.
- pinlogger_replica.py / pinlogger_replica_extended.py: reproduce a fixed
  70/15/15 train/validation/test MLP setup similar to a prior published method
  (PinLogger), using "basic" features and "extended" features respectively.

CNN experiments
- cnn_variant_a.py to cnn_variant_e.py: each trains one specific 1D CNN
  configuration (original vs reduced vs large capacity, with/without data
  augmentation) via core.run_experiment. Run each one separately to compare.
- channel_comparison.py: repeats the CNN experiment three times using 6, 9, or
  12 sensor channels, to see whether adding accelerometer/derived channels helps.
- window_sweep.py: repeats an XGBoost experiment across window sizes
  (150ms to 500ms) and 3 random seeds, using the "combined" feature set.
- advanced_nn_models.py: defines and trains two alternative architectures, a
  Temporal Convolutional Network (TCN) and a CNN+GRU hybrid (CRNN).
- cnn_vs_tcn_multiseed.py: compares the best CNN against the TCN across
  multiple random seeds for a more robust comparison.
- loss_function_comparison.py: same CNN, tested with different loss functions.
- optimizer_comparison.py: same CNN, tested with different optimizers.

Segmentation experiments
- adaptive_segmentation.py: compares a fixed 400ms window against a
  variable-length window that expands/contracts based on when the signal
  looks "quiet" before and after a keypress.
- blind_segmentation.py: tests whether keypress times can be detected directly
  from the sensor signal itself (without a ground truth from a JS event) using an RMS
  spike-detection sweep, then classifies using those detected times.

Other experiments
- ensemble_model.py: combines CNN, Random Forest, XGBoost and MLP predictions
  by averaging their probabilities and compares the ensemble
  against each individual model.
- hierarchical_classification.py: splits the 9-region problem into a row
  classifier and a column classifier and combines them, compared against a
  flat 9-way classifier.

## 5. Reproducibility notes

- Most scripts fix SEED = 2133744 for the classical models and CNNs, so
  results should be very close to identical between runs. window_sweep.py and
  cnn_vs_tcn_multiseed.py intentionally test multiple seeds to check stability.
