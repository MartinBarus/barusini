from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from barusini.tabular.transformers.encoders import GenericEncoder
from barusini.tabular.transformers.target_encoders import TargetEncoder
from barusini.utils import (
    sanitize,
    update_kwargs,
    kwargs_subset,
    kwargs_subset_except,
)


class TfIdfEncoder(GenericEncoder):
    show_unseen = False
    is_sparse = True

    def __init__(self, used_cols=None, vocab_size=100, **kwargs):
        super().__init__(used_cols=used_cols)
        idf_kwargs = kwargs_subset(kwargs, "tfidf_")
        idf_kwargs = update_kwargs(
            idf_kwargs,
            min_df=5,
            max_df=0.8,
            stop_words="english",
            ngram_range=(1, 1),
            max_features=vocab_size,
        )
        self.encoder = TfidfVectorizer(**idf_kwargs)

    def fit_names(self):
        self.target_names = [
            sanitize(f"{self.used_cols} {val}")
            for val in self.encoder.get_feature_names()
        ]

    def preprocess(self, X):
        assert len(self.used_cols) == 1
        x = super().preprocess(X)
        return x[self.used_cols[0]]

    def __str__(self):
        encoder_str = "TF-IDF"

        return (
            f"TF IDF followed by PCA of '{self.used_cols}':\n\t"
            f"Categories: {encoder_str}"
        )


class TfIdfPCAEncoder(GenericEncoder):
    show_unseen = False

    def __init__(self, used_cols=None, n_components=100, **kwargs):
        super().__init__(used_cols=used_cols)
        idf_kwargs = kwargs_subset(kwargs, "tfidf_")
        idf_kwargs = update_kwargs(
            idf_kwargs, min_df=3, stop_words="english", ngram_range=(1, 2),
        )
        pca_kwargs = kwargs_subset(kwargs, "pca_")
        pca_kwargs = update_kwargs(
            pca_kwargs, n_components=n_components, n_iter=7, random_state=42,
        )
        self.encoder = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**idf_kwargs)),
                ("pca", TruncatedSVD(**pca_kwargs)),
            ]
        )

    def preprocess(self, X):
        assert len(self.used_cols) == 1
        x = super().preprocess(X)
        return x[self.used_cols[0]]

    def fit_names(self):
        self.target_names = [
            sanitize(f"{self.used_cols} PCA:{val}")
            for val in range(self.encoder.named_steps["pca"].n_components)
        ]

    def __str__(self):
        encoder_str = "TF-IDF + PCA({})".format(
            self.encoder.named_steps["pca"].n_components
        )

        return (
            f"TF IDF followed by PCA of '{self.used_cols}':\n\t"
            f"Categories: {encoder_str}"
        )


class LinearTextEncoder(TargetEncoder):
    target_str = "[TextTE]"
    x_dim = 1

    def __init__(self, **kwargs):
        idf_kwargs = kwargs_subset(kwargs, "tfidf_")
        idf_kwargs = update_kwargs(idf_kwargs, max_features=100)
        model_kwargs = kwargs_subset(kwargs, "model_")
        rest_kwargs = kwargs_subset_except(kwargs, ["tfidf_", "model_"])
        super().__init__(
            encoder=Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(**idf_kwargs)),
                    ("predictor", LinearRegression(**model_kwargs)),
                ]
            ),
            create_single_col=False,
            **rest_kwargs,
        )
