# barusini

This library aims to cover basic machine learning modeling workflow. 

To install run
```pip install git+https://github.com/MartinBarus/barusini.git```

 
There are two parts of the library:

- `barusini.tabular`
- `barusini.nn`

## Tabular Data

`barusini.tabular `implements automated ML pipeline of feature engineering, 
model hyper-parameter tuning and model ensembling. This automated pipeline 
consists of multiple steps, each of them can be tweaked. 

### Automated model building pipeline
Given user provides an estimator (f.e. XGBoostClassifier), scorer (f.e. logloss)
and input data with the label, general automated  model building pipeline follows this logic
1. `base-stage`- subset only numeric features and get the base score

2. `imputation` - if there are numeric features with missing values, try different imputations
and choose the imputations improving the score the most

3. `feature-selection` - iteratively drop the worst feature, if the score improves afterwards 

4. `categorical-encoding` - for every categorical (string) column, 
search for encoding that improves the score the most, then for every numeric feature
that could be treated as categorical (int with few unique values) 
try if any encoding improves the score

5. `hyper-parameter-tuning` - look for hyper-parameters of the model that optimize the score


Steps 1-4 can be achieved by function `barusini.tabular.feature_engineering.feature_engineering`.

Steps 5 can be achieved by function `barusini.tabular.feature_engineering.model_search`.

Steps 1-5 can be all achieved by function `barusini.tabular.feature_engineering.auto_ml`.

### Core concepts

There are few important concepts of this library

- Transformer
- Model
- Pipeline
- Ensemble

#### Transformer
Transformers are objects that transform the input data, they are implementing different feature
engineering methods. ML Algorithms can only work with numbers/vectors, there are different ways
to create new useful data representations of the original data. 

This is the list of the available transformers:
 
 - `Identity` - provides input data as is
 
 - `MissingValueImputer` - provides input numeric data, but missing values are imputed.
 Imputation method is a parameter, default imputation is median
 
 - `QuantizationTransformer` - transforms numeric data into bins
 
 - `DateExtractor` - extracts different datetime properties such as `dayofweek`, `year` and so on.
 The list of properties extracted is a parameter of the transformer
 
 - `CustomLabelEncoder` - encodes strings to numbers from 0 to number of unique values -1, 
 useful for tree based models mostly
 
 - `CustomOneHotEncoder` - encodes strings to vectors, when each element of vector represents
 
 bool flag of certain string value, useful for linear models or low cardinality features.
 
 - `MeanTargetEncoder` - encodes string by the mean response/target, useful for high cardinality features
 in tree based models
 
 - `TfIdfEncoder` - encodes text to TF-IDF vectors (similar to CustomOneHotEncoder for free text)
 
 - `TfIdfPCAEncoder` - it is `TfIdfEncoder` followed by dimensionality reduction (PCA)
 
 - `LinearTextEncoder` - free text variation of `MeanTargetEncoder`

#### Model
Model is an instance of ML algorithm that has some hyper-parameters (settings) and can be fitted
and later be used for predictions. This library does not implement any models/algorithms, it uses
models from other libraries like `sklearn`, `lightgbm` or `xgboost`, provided they are compatible with
`sklearn` API (they implement `fit`/`transform`/`fit_transform`).

#### Pipeline
Pipeline is an important concept missing in some libraries, it allows us to
package chain of feature engineering steps (transformers) together with a model.

In this library, Pipeline is defined by providing a list of transformers and a model.
It implements following methods:

- `fit` - fits the whole Pipeline
- `transform` - transforms the data using all transformers
- `predict` - make predictions
- `predict_proba` - predicts probabilities (applicable for classification)
- `varimp` - get feature importance
- `tune` - search for best model hyper-parameters and fit

Example from `tests.test_tabular.test_integration`:

```
model = Pipeline(
    transformers=[
        Identity(used_cols=["fnlwgt"]),
        Identity(used_cols=["education-num"]),
        Identity(used_cols=["relationship"]),
        MeanTargetEncoder(used_cols=["occupation"]),
        CustomLabelEncoder(used_cols=["sex"]),
        CustomOneHotEncoder(used_cols=["race"]),
    ],
    model=LGBMClassifier(),
)
```

#### Ensemble
Ensemble is an object that combines multiple Pipelines. It as an advanced
Machine Learning concept, where after training multiple separate Pipelines
on the same data, we can improve the overall performance by training additional
meta model, combining these individual Pipelines. In this library, the meta model
is by default a simple weighted average.

It implements following functions:

- `fit` - fits the underlying Pipelines and meta model
- `predict` - make prediction
- `predict_proba` - predicts probabilities (applicable for classification)

Example:
```
model1 = Pipeline(...)
model2 = Pipeline(...)
ensemble = Ensemble([model1, model2])
```

## Neural Networks

`barusini.nn` is meant to be a simple extensible module for modeling using the `pytorch` framework stack.

Currently this module consists of two sub-modules:

- `barusini.nn.image` - implements Image regression and classification (binary and multi-class)
- `barusini.nn.nlp` - implements NLP (text) regression and classification (binary and multi-class)

For more information on how to use these sub-modules see `tests/test_nn`.
