import pytest
import pandas as pd
from barusini.transformers import TargetEncoder


@pytest.fixture(scope="session")
def simple_data():
    idx = range(1, 11)
    X = pd.DataFrame({"a": ["a", "b"] * 5}, index=idx)
    y = pd.Series(range(10), index=idx)
    return X, y


def test_target_encoding_test(simple_data):
    #  GIVEN simple dataset
    X, y = simple_data

    # WHEN `fit` and `transform` is called
    te = TargetEncoder()
    te.fit(X, y)
    transformed_X = te.transform(X)

    # THEN index is the same as index of input DataFrame:
    assert all(X.index == transformed_X.index)

    # THEN transformed data matches expectation
    assert all(transformed_X == [4, 5] * 5)


def test_target_encoding_train(simple_data):
    #  GIVEN simple dataset
    X, y = simple_data

    # WHEN `fit_transform(X, y)` is called, or `transform(X, True)` is called
    # on fitted encoder
    te = TargetEncoder()
    transformed_X = te.fit_transform(X, y)
    transformed_X2 = te.transform(X, True)

    # THEN index is the same as index of the input DataFrame:
    assert all(X.index == transformed_X.index)
    assert all(X.index == transformed_X2.index)

    # THEN transformed data matches expectation
    expected = [5, 6, 4.5, 5.5, 4, 5, 3.5, 4.5, 3, 4]
    assert all(transformed_X == expected)
    assert all(transformed_X2 == expected)
