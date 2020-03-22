from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from barusini.transformers.encoders import GenericEncoder
from barusini.transformers.target_encoders import TargetEncoder
from barusini.utils import sanitize


class TfIdfEncoder(GenericEncoder):
    show_unseen = False
    is_sparse = True

    def __init__(self, used_cols=None, vocab_size=100, **kwargs):
        super().__init__(used_cols=used_cols)
        self.encoder = TfidfVectorizer(
            min_df=5,
            max_df=0.8,
            stop_words="english",
            ngram_range=(1, 1),
            max_features=vocab_size,
            **kwargs,
        )

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
        self.encoder = Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        min_df=3, stop_words="english", ngram_range=(1, 2)
                    ),
                ),
                (
                    "pca",
                    TruncatedSVD(
                        n_components=n_components, n_iter=7, random_state=42
                    ),
                ),
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
        super().__init__(
            encoder=Pipeline(
                steps=[
                    ("vec", TfidfVectorizer(max_features=100)),
                    ("predictor", LinearRegression()),
                ]
            ),
            **kwargs,
        )
