"""Construction des splits train / val / test pour le dataset infrarouge.

Lit data/dataset_eda.csv et écrit data/train.csv, data/val.csv et data/test.csv.

Le découpage respecte deux contraintes en même temps :
  - les groupes de similarité ne sont jamais coupés entre deux splits (pas de fuite) ;
  - les proportions des 12 classes sont conservées dans chaque split (stratification).

Usage :
    uv run python scripts/prepare_dataset.py
    uv run python scripts/prepare_dataset.py --seed 7 --n-splits-test 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_CSV = DATA_DIR / "dataset_eda.csv"

SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def assigner_groupes_uniques(df: pd.DataFrame) -> pd.DataFrame:
    """Donne un groupe propre à chaque image sans groupe de similarité.

    Sans ça, toutes les images isolées partageraient le groupe NaN et finiraient
    forcément dans le même split.
    """
    out = df.copy()
    out["similarity_group"] = out["similarity_group"].astype("string")

    isolees = out["similarity_group"].isna()
    out.loc[isolees, "similarity_group"] = "single_" + out.index[isolees].astype(str)

    if out["similarity_group"].isna().any():
        raise ValueError("Des similarity_group sont restés vides après remplissage.")

    return out


def split_stratifie_par_groupe(
    df: pd.DataFrame, n_splits: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coupe df en (grande part, petite part) de taille 1/n_splits.

    GroupShuffleSplit accepte `y=` mais l'ignore : avec 175 images pour Diode-Multi,
    la répartition des classes rares serait laissée au hasard. StratifiedGroupKFold non.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_grand, idx_petit = next(
        sgkf.split(df, y=df["label"], groups=df["similarity_group"])
    )
    return df.iloc[idx_grand].copy(), df.iloc[idx_petit].copy()


def verifier_absence_de_fuite(splits: dict[str, pd.DataFrame]) -> None:
    """Échoue si un groupe de similarité apparaît dans deux splits."""
    groupes = {nom: set(d["similarity_group"]) for nom, d in splits.items()}
    noms = list(groupes)

    for i, a in enumerate(noms):
        for b in noms[i + 1:]:
            commun = groupes[a] & groupes[b]
            if commun:
                raise ValueError(
                    f"Fuite entre {a} et {b} : {len(commun)} groupes communs "
                    f"(ex. {sorted(commun)[:3]})"
                )
            print(f"  Groupes communs {a}/{b} : 0")


def rapport_stratification(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Proportions de chaque classe dans chaque split, pour contrôle visuel."""
    return pd.concat(
        {nom: d["label"].value_counts(normalize=True) for nom, d in splits.items()},
        axis=1,
    ).round(4)


def preparer_splits(
    source_csv: Path, n_splits_test: int, n_splits_val: int, seed: int
) -> dict[str, pd.DataFrame]:
    """Enchaîne lecture, remplissage des groupes et double découpage."""
    df = pd.read_csv(source_csv)
    print(f"Source : {source_csv}  ({len(df)} lignes)")

    df = assigner_groupes_uniques(df)

    # Deux découpages successifs : on sort d'abord le test, puis la validation
    # depuis ce qui reste. La validation sert à régler les hyperparamètres,
    # ce qui évite de contaminer le test en le regardant à chaque essai.
    train_val, test = split_stratifie_par_groupe(df, n_splits_test, seed)
    train, val = split_stratifie_par_groupe(train_val, n_splits_val, seed)

    return {"train": train, "val": val, "test": test}


def ecrire_splits(splits: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Écrit un CSV par split dans out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for nom, d in splits.items():
        chemin = out_dir / SPLIT_FILES[nom]
        d.to_csv(chemin, index=False)
        print(f"  {chemin}  ({len(d)} lignes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CSV,
                        help="CSV d'entrée (défaut : data/dataset_eda.csv)")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                        help="Dossier de sortie des splits (défaut : data/)")
    parser.add_argument("--n-splits-test", type=int, default=5,
                        help="1/N du total part en test (défaut : 5, soit 20%%)")
    parser.add_argument("--n-splits-val", type=int, default=5,
                        help="1/N du reste part en validation (défaut : 5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"Fichier introuvable : {args.source}", file=sys.stderr)
        return 1

    splits = preparer_splits(
        args.source, args.n_splits_test, args.n_splits_val, args.seed
    )

    print("\nTailles :")
    for nom, d in splits.items():
        print(f"  {nom:5s} : {len(d):6d}")

    print("\nStratification (proportion par classe) :")
    print(rapport_stratification(splits).to_string())

    print("\nContrôle de fuite :")
    verifier_absence_de_fuite(splits)

    print("\nÉcriture :")
    ecrire_splits(splits, args.out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
