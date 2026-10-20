
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


def test_preprocessing_logic():
    df = pd.DataFrame(
        {
            "age": [63, 45, 52],
            "ca": [0.0, None, 1.0],
            "thal": [3.0, 6.0, None],
            "num": [0, 2, 1],
        }
    )

    df["target"] = (df["num"] > 0).astype(int)

    columns_to_impute = ["ca", "thal"]
    imputer = SimpleImputer(strategy="most_frequent")
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

    assert df["target"].tolist() == [0, 1, 1]
    assert set(df["target"].unique()).issubset({0, 1})
    assert df[columns_to_impute].isna().sum().sum() == 0
    assert df["num"].tolist() == [0, 2, 1]


def test_processed_dataset_exists_and_is_complete():
    project_root = Path(__file__).resolve().parent.parent
    processed_file = project_root / "data" / "processed" / "heart_disease.csv"

    assert processed_file.exists()

    df = pd.read_csv(processed_file)

    assert len(df) == 303
    assert "num" in df.columns
    assert "target" in df.columns
    assert df["target"].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0
