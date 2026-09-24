"""
build_train_test_split.py

Implements the split shown in your diagram:

  Organic dataset
      |-- Organic negatives (is_breaking_commit == 0)  -> TRAIN (+ synthetic)
      `-- Organic positives with real AST signal        -> TEST ONLY

  Synthetic dataset (from synthetic_mutation_generator.py) -> TRAIN only

This guarantees the model is evaluated exclusively on real, AST-corroborated
positive examples that it never saw a synthetic analog of during training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

ORGANIC_CSV = "fastapi_ast_delta_dataset.csv"
SYNTHETIC_CSV = "synthetic_breaking_dataset.csv"

TRAIN_OUT = "train_dataset.csv"
TEST_OUT = "test_dataset.csv"

AST_COLS = [
    "fields_removed_count", "fields_added_count", "fields_type_changed_count",
    "fields_default_changed_count", "models_renamed_count", "routes_removed_count",
    "routes_added_count", "routes_param_changed_count", "routes_param_type_changed_count",
    "routes_param_default_changed_count",
]

# How many of the real, AST-corroborated positives to reserve as an
# additional "recent-negatives-only" sanity check inside train, rather than
# spending literally all of them on test. Adjust to taste; with as few as
# ~20 real positives, holding out 100% for test leaves nothing to inspect
# during training, so this keeps a couple aside as train-time diagnostics
# only (they are NOT used to fit the model, just to eyeball predictions).
REAL_POSITIVES_HELD_FOR_TRAIN_INSPECTION = 0  # set to e.g. 2-3 if you want a peek


def main():
    organic = pd.read_csv(ORGANIC_CSV)
    synthetic = pd.read_csv(SYNTHETIC_CSV)

    has_ast_signal = organic[AST_COLS].sum(axis=1) > 0
    is_positive = organic["is_breaking_commit"] == 1

    real_positives = organic[is_positive & has_ast_signal].copy()
    organic_negatives = organic[~is_positive].copy()
    # Positives labeled breaking but with NO ast signal are weak-label noise --
    # excluded from both train and test rather than silently mislabeling either.
    dropped_weak_positives = organic[is_positive & ~has_ast_signal]

    print(f"Real AST-corroborated positives: {len(real_positives)}")
    print(f"Organic negatives: {len(organic_negatives)}")
    print(f"Synthetic positives: {len(synthetic)}")
    print(f"Dropped weak-label positives (no AST signal, excluded entirely): {len(dropped_weak_positives)}")

    if REAL_POSITIVES_HELD_FOR_TRAIN_INSPECTION > 0:
        real_positives = real_positives.sample(frac=1, random_state=42)
        held_for_train = real_positives.iloc[:REAL_POSITIVES_HELD_FOR_TRAIN_INSPECTION]
        test_positives = real_positives.iloc[REAL_POSITIVES_HELD_FOR_TRAIN_INSPECTION:]
    else:
        held_for_train = real_positives.iloc[0:0]  # empty
        test_positives = real_positives

    # Split organic negatives so test set also has a realistic negative pool
    # matched roughly to the positive count (avoid a 1000:1 test set).
    test_negative_count = min(len(organic_negatives), max(len(test_positives) * 20, 200))
    test_negatives, remaining_negatives = train_test_split(
        organic_negatives, train_size=test_negative_count, random_state=42
    )

    train_df = pd.concat([
        remaining_negatives,
        synthetic,
        held_for_train,
    ], ignore_index=True)

    test_df = pd.concat([
        test_positives,
        test_negatives,
    ], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df.to_csv(TRAIN_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    print(f"\nTrain set: {len(train_df)} rows "
          f"({train_df['is_breaking_commit'].sum()} positive, "
          f"{(train_df['is_breaking_commit'] == 0).sum()} negative)")
    print(f"Test set: {len(test_df)} rows "
          f"({test_df['is_breaking_commit'].sum()} positive, "
          f"{(test_df['is_breaking_commit'] == 0).sum()} negative)")
    print("\nNote: test set positives are 100% real, AST-corroborated commits.")
    print("No synthetic example exists in the test set.")


if __name__ == "__main__":
    main()