import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from barusini.tabular.transformers import (
    ConfidenceIntervals,
    CustomLabelEncoder,
    CustomOneHotEncoder,
    LinearTextEncoder,
    MeanTargetEncoder,
    MissingValueImputer,
    Pipeline,
    QuantizationTransformer,
    TfIdfEncoder,
    TfIdfPCAEncoder,
)
from barusini.utils import sanitize


@pytest.fixture(scope="session")
def simple_data():
    idx = range(1, 11)
    X = pd.DataFrame({"a": ["a", "b"] * 5}, index=idx)
    y = pd.Series(range(10), index=idx)
    return X, y


@pytest.fixture(scope="session")
def simple_with_unseen(simple_data):
    X, y = simple_data
    X = pd.concat([X, pd.DataFrame({"a": ["c", "d"]})]).reset_index(drop=True)
    return X, y


@pytest.fixture(scope="session")
def more_data(simple_data):
    X, y = simple_data
    X = X.copy()
    X["b"] = list(range(10))
    X["c"] = [
        "cat",
        "dog",
        "dog and cat",
        "hot dog",
        "cat woman",
        "man and woman",
        "batman and catwoman",
        "",
        "and",
        "new",
    ]
    return X, y


@pytest.fixture(scope="session")
def more_data_multiclass(more_data):
    X, y = more_data
    y = [1, 2, 3, 1, 2] * 2
    return X, y


def test_target_encoding_test(simple_data):
    #  GIVEN simple dataset
    X, y = simple_data

    # WHEN `fit` and `transform` is called
    te = MeanTargetEncoder()
    te.fit(X, y)
    transformed_X = te.transform(X)[te.target_names]

    # THEN index is the same as index of input DataFrame:
    assert all(X.index == transformed_X.index)

    # THEN transformed data matches expectation
    assert all(transformed_X.values.reshape(-1) == [4, 5] * 5)


def test_target_encoding_train(simple_data):
    #  GIVEN simple dataset
    X, y = simple_data

    # WHEN `fit_transform(X, y)` is called, or `transform(X, True)` is called
    # on fitted encoder
    te = MeanTargetEncoder()
    transformed_X = te.fit_transform(X, y)
    transformed_X2 = te.transform(X, train_data=True)

    # THEN index is the same as index of the input DataFrame:
    assert all(X.index == transformed_X.index)
    assert all(X.index == transformed_X2.index)

    # THEN transformed data matches expectation
    expected = [5, 6, 4.5, 5.5, 4, 5, 3.5, 4.5, 3, 4]
    assert np.allclose(
        transformed_X[te.target_names].values.reshape(-1), expected
    )
    assert np.allclose(
        transformed_X2[te.target_names].values.reshape(-1), expected
    )
    assert (
        transformed_X[te.output_columns()].shape[1]
        == transformed_X2[te.output_columns()].shape[1]
        == 1
    )


def test_te_with_more_cols(more_data):
    #  GIVEN simple dataset with additional columns
    X, y = more_data

    # WHEN `fit(X, y)` is called on Target encoder using pd.Series or
    # pd.DataFrame
    te1 = MeanTargetEncoder()
    te2 = MeanTargetEncoder()
    te1.fit(X["a"], y)
    te2.fit(X[["a"]], y)

    # THEN `transform` on larger pd.DataFrame contains additional columns and
    # results match expectations
    transformed_X1 = te1.transform(X)
    transformed_X2 = te2.transform(X)
    assert all(X.index == transformed_X1.index)
    assert all(X.index == transformed_X2.index)

    expected = [4, 5] * 5
    assert np.allclose(
        transformed_X1[te1.target_names].values.reshape(-1), expected
    )
    assert np.allclose(
        transformed_X2[te2.target_names].values.reshape(-1), expected
    )

    assert transformed_X1.shape[1] == transformed_X2.shape[1] == 4


def test_ohe_encoding(simple_data):
    #  GIVEN simple dataset
    X, y = simple_data

    # WHEN `fit_transform(X)` is called on OHE transformer
    ohe = CustomOneHotEncoder()
    transformed_X = ohe.fit_transform(X)[ohe.output_columns()]

    # THEN index is the same as index of the input DataFrame:
    assert all(X.index == transformed_X.index)

    # THEN transformed data matches expectation
    expected_a = [1, 0] * 5
    expected_b = [0, 1] * 5
    assert all(transformed_X.iloc[:, 0] == expected_a)
    assert all(transformed_X.iloc[:, 1] == expected_b)
    assert transformed_X.shape[1] == 2


def test_ohe_with_more_cols(more_data):
    #  GIVEN simple dataset with additional columns
    X, y = more_data

    # WHEN `fit(X)` is called on OHE encoder using pd.Series or
    # pd.DataFrame
    ohe1 = CustomOneHotEncoder()
    ohe2 = CustomOneHotEncoder()
    ohe1.fit(X["a"])
    ohe2.fit(X[["a"]])

    # THEN `transform` on larger pd.DataFrame contains additional columns and
    # results match expectations
    transformed_X1 = ohe1.transform(X)
    transformed_X2 = ohe2.transform(X)
    assert all(X.index == transformed_X1.index)
    assert all(X.index == transformed_X2.index)

    expected_a = [1, 0] * 5
    expected_b = [0, 1] * 5
    assert all(transformed_X1[sanitize("[[a]] [OHE:a]")] == expected_a)
    assert all(transformed_X1[sanitize("[[a]] [OHE:b]")] == expected_b)
    assert all(transformed_X2[sanitize("[[a]] [OHE:a]")] == expected_a)
    assert all(transformed_X2[sanitize("[[a]] [OHE:b]")] == expected_b)

    assert transformed_X1.shape[1] == transformed_X2.shape[1]


def test_le_encoding(simple_data):
    #  GIVEN simple dataset
    X, y = simple_data

    # WHEN `fit_transform(X)` is called on LE transformer
    le = CustomLabelEncoder()
    transformed_X = le.fit_transform(X)[le.output_columns()]

    # THEN index is the same as index of the input DataFrame:
    assert all(X.index == transformed_X.index)

    # THEN transformed data matches expectation
    expected = [0, 1] * 5
    assert all(transformed_X.iloc[:, 0] == expected)
    assert transformed_X.shape[1] == 1


def test_le_with_more_cols(more_data):
    #  GIVEN simple dataset with additional columns
    X, y = more_data

    # WHEN `fit(X)` is called on LE encoder using pd.Series or
    # pd.DataFrame
    le1 = CustomLabelEncoder()
    le2 = CustomLabelEncoder()
    le1.fit(X["a"])
    le2.fit(X[["a"]])

    # THEN `transform` on larger pd.DataFrame contains additional columns and
    # results match expectations
    transformed_X1 = le1.transform(X)
    transformed_X2 = le2.transform(X)
    assert all(X.index == transformed_X1.index)
    assert all(X.index == transformed_X2.index)

    expected = [0, 1] * 5
    assert all(transformed_X1[sanitize("[[a]] [LE]")] == expected)
    assert all(transformed_X2[sanitize("[[a]] [LE]")] == expected)

    assert transformed_X1.shape[1] == transformed_X2.shape[1]


def fit_transform(enc, X, y, *args, **kwargs):
    enc.fit(X, y, *args, **kwargs)
    enc.transform(X)
    enc.fit_transform(X, y, *args, **kwargs)
    enc.transform(X)


def test_all_encoders(more_data, more_data_multiclass):
    X, y = more_data
    _, y_multi_class = more_data_multiclass
    encoders = [
        CustomOneHotEncoder(used_cols=["a"]),
        CustomLabelEncoder(used_cols=["a"]),
        MeanTargetEncoder(used_cols=["a"]),
        QuantizationTransformer(used_cols=["b"], n_bins=4),
        MissingValueImputer(used_cols=["b"]),
        LinearTextEncoder(used_cols=["c"]),
        TfIdfEncoder(used_cols=["c"], tfidf_min_df=1),
        TfIdfPCAEncoder(used_cols=["c"], pca_n_components=1),
    ]
    for encoder in encoders:
        fit_transform(encoder, X, y)

    mdl = Pipeline(encoders, RandomForestClassifier())
    mdl.fit(X, y)
    mdl.transform(X)
    mdl.predict(X)
    mdl.predict_proba(X)
    mdl.fit_transform(X, y)
    print(mdl)


def test_confidence_intervals():
    # GIVEN out of fold predictions with label
    df = pd.DataFrame(np.random.random((1000, 2)), columns=["pred", "label"])
    df["label"] = 1 * df["label"] > 0.5

    # WHEN we estimate confidence intervals for specific quantiles q and their
    # counter parts 1-q and transform predictions to show confidence intervals
    quantiles = [0.05, 0.1, 0.3]
    ci = ConfidenceIntervals()
    ci.fit(df["label"], df["pred"], quantiles)
    new_preds = ci.transform(df["pred"])

    # THEN for every upper/lower bound, the relative number of labels being
    # <= predictions, belonging to confidence intervals matches the quantiles
    for conf_interval in quantiles:
        for offset in [0, 1]:
            conf_interval = abs(offset - conf_interval)  # q or 1-q
            pred_col = f"prediction {conf_interval}"
            match_conf_interval = df["label"] <= new_preds[pred_col]
            matched = new_preds[match_conf_interval]
            relative_match = matched.shape[0] / df.shape[0]
            assert relative_match == conf_interval
