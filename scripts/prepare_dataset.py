"""Build the train / val / test splits from data/dataset_eda.csv.

Similarity groups never cross two splits (no leakage) and the proportions of the 12 classes
are kept in every split.

Usage:
    uv run python scripts/prepare_dataset.py
    uv run python scripts/prepare_dataset.py --seed 7 --n-splits-test 4
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_CSV = DATA_DIR / "dataset_eda.csv"

SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def assign_singleton_groups(df):
    """Give each image without a similarity group its own group.

    Otherwise all isolated images would share the NaN group and land in the same split.
    """
    out = df.copy()
    out["similarity_group"] = out["similarity_group"].astype("string")
    isolated = out["similarity_group"].isna()
    out.loc[isolated, "similarity_group"] = "single_" + out.index[isolated].astype(str)
    return out


def stratified_group_split(df, n_splits, seed):
    """Split df into (large part, 1/n_splits part).

    GroupShuffleSplit ignores `y`, which would leave the rare classes (175 Diode-Multi images)
    to chance; StratifiedGroupKFold does not.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    large_idx, small_idx = next(sgkf.split(df, y=df["label"], groups=df["similarity_group"]))
    return df.iloc[large_idx].copy(), df.iloc[small_idx].copy()


def check_no_leakage(splits):
    """Raise if a similarity group appears in two splits."""
    groups = {name: set(d["similarity_group"]) for name, d in splits.items()}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = groups[a] & groups[b]
            if shared:
                raise ValueError(f"Leakage between {a} and {b}: {len(shared)} shared groups "
                                 f"(e.g. {sorted(shared)[:3]})")
            print(f"  Shared groups {a}/{b}: 0")


def stratification_report(splits):
    """Share of each class in each split."""
    return pd.concat(
        {name: d["label"].value_counts(normalize=True) for name, d in splits.items()},
        axis=1,
    ).round(4)


def build_splits(source_csv, n_splits_test, n_splits_val, seed):
    df = pd.read_csv(source_csv)
    print(f"Source: {source_csv}  ({len(df)} rows)")
    df = assign_singleton_groups(df)

    # Test first, then validation from the rest: validation is used for tuning, test is never seen.
    train_val, test = stratified_group_split(df, n_splits_test, seed)
    train, val = stratified_group_split(train_val, n_splits_val, seed)
    return {"train": train, "val": val, "test": test}


def write_splits(splits, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, d in splits.items():
        path = out_dir / SPLIT_FILES[name]
        d.to_csv(path, index=False)
        print(f"  {path}  ({len(d)} rows)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE_CSV,
                        help="input CSV (default: data/dataset_eda.csv)")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                        help="output folder for the splits (default: data/)")
    parser.add_argument("--n-splits-test", type=int, default=5,
                        help="1/N of the data goes to test (default: 5, i.e. 20%%)")
    parser.add_argument("--n-splits-val", type=int, default=5,
                        help="1/N of the rest goes to validation (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"File not found: {args.source}", file=sys.stderr)
        return 1

    splits = build_splits(args.source, args.n_splits_test, args.n_splits_val, args.seed)

    print("\nSizes:")
    for name, d in splits.items():
        print(f"  {name:5s}: {len(d):6d}")

    print("\nStratification (share per class):")
    print(stratification_report(splits).to_string())

    print("\nLeakage check:")
    check_no_leakage(splits)

    print("\nWriting:")
    write_splits(splits, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
