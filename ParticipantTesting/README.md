# How to Reproduce the Results

This folder contains the multi-participant generalization tests. It
checks whether a model trained on some people's keystroke motion data can
recognize keystrokes from a *different*, unseen person, as opposed to the
single-session tests in the earlier folders which only ever mix data from
one person.

## 1. Folder setup

These scripts expect one subfolder per participant, each with its own pair
of data files:

```
ParticipantTests/
  core.py
  participant_generalization_tests.py
  pooled_mixed_model_test.py
  person_classifier_poc.py
  signal_magnitude_diagnostic.py
  participant_1/
    sensor_log.jsonl
    key_log.jsonl
  participant_2/
    sensor_log.jsonl
    key_log.jsonl
  participant_3/
    sensor_log.jsonl
    key_log.jsonl
  participant_4/
    sensor_log.jsonl
    key_log.jsonl
  participant_5/
    sensor_log.jsonl
    key_log.jsonl
```

core.py is not a script you run yourself, it's the shared library the other
four import from.

Each of the four runnable scripts has its own PARTICIPANT_FILES dictionary
near the top, pointing at these folders by name. If your participant
folders are named differently, or you have more or fewer than five
participants, edit PARTICIPANT_FILES at the top of each script to match
before running it. All four scripts currently list the same five
participants, so if you rename or add/remove one, update it in all four
places.

## 2. Install dependencies

You need Python 3.9 or later. Install the required packages:

```
pip install numpy pandas scikit-learn xgboost torch
```

## 3. Running a script

From inside the folder, run any script directly, for example:

```
python person_classifier_poc.py
```

Each script prints progress as it loads each participant's data, then
per-fold or per-participant results, then a final summary.

participant_generalization_tests.py is the main one, it runs all six
headline results end to end and can take a while. 

## 4. What each script does

- participant_generalization_tests.py: the full test suite, produces all
  six results needed for the results section in one run:
  1. Nine-region CNN, Leave-One-Participant-Out (LOPO)
  2. Nine-region XGBoost, LOPO
  3. Dummy baseline (both 9-region and 26-key label schemes)
  4. Within-participant nine-region CNN (5-fold CV run separately per
     participant, then averaged)
  5. 26-key CNN, LOPO
  6. 26-key XGBoost, LOPO

- pooled_mixed_model_test.py: the pooled "mixed-model" comparison on its
  own. Pools every participant's data together and evaluates with standard
  stratified 5-fold CV, the same way prior work like PINlogger.js typically
  evaluates. This does NOT respect participant boundaries between train and
  test, so it is expected to look better than the genuine LOPO numbers.
  Compare its output against the LOPO rows from
  participant_generalization_tests.py to see how much this style of
  evaluation inflates accuracy relative to true unseen-person testing.

- person_classifier_poc.py: a different question, not "which key was
  pressed" but "which person is typing." A quick proof-of-concept using an
  untuned Random Forest on the existing "combined" feature set, evaluated
  with standard closed-set stratified CV (every participant is seen during
  training, just not that exact keystroke). This is not a test of detecting
  an unknown/new person, just whether the sensor signal carries any
  person-identifying information at all.

## 5. LOPO vs pooled vs within-participant: what each one tells you

- LOPO (Leave-One-Participant-Out): train on every other participant,
  test on one held-out participant, rotate through all of them. This is the
  only one of the three that genuinely tests generalization to a new
  person the model has never seen before.
- Pooled mixed-model: all participants pooled together, then split with
  ordinary stratified k-fold. Train and test folds can (and will) contain
  keystrokes from the same participant.
- Within-participant: 5-fold CV done separately for each participant, then
  averaged. This tests generalization to new keystrokes from a person
  already seen during training, not generalization to a new person. It's
  the same style of evaluation used in the single-session tests in the
  earlier folders, just reported per participant here.


## 6. Reproducibility notes

- All four scripts fix SEED = 2133744, plus participant_generalization_tests.py
  and pooled_mixed_model_test.py also seed Python's random module and
  numpy's global random state at the start of train_cnn (on top of the
  torch seeding), so CNN results should be very close to identical between
  runs on the same machine.
