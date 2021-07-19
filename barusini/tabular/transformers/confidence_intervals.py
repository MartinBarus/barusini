import pandas as pd


class ConfidenceIntervals:
    def __init__(self):
        self.quantiles = None
        self.mean = None
        self.std = None

    @staticmethod
    def get_quantiles(residuals, quantiles=[0.05]):
        quantiles = [q if q <= 0.5 else 1 - q for q in quantiles]
        quantiles = sorted(set(quantiles))
        quantiles2 = [1 - q for q in quantiles]
        quantiles = quantiles + quantiles2[::-1]
        return residuals.quantile(quantiles)

    def fit(self, predictions, labels, quantiles):
        residuals = predictions - labels
        self.quantiles = self.get_quantiles(residuals, quantiles)
        self.check_quantiles()
        self.quantiles[""] = 0
        self.quantiles.index = [f"prediction {x}" for x in self.quantiles.index]
        return self

    def check_quantiles(self):
        for quantile, value in self.quantiles.items():
            if (quantile <= 0.5) != (value <= 0):
                print(
                    f"Warning, unexpected sign for quantile {quantile}"
                    f"with value {value}!"
                )

    def transform(self, predictions):
        confidence_intervals = predictions.values.reshape(
            -1, 1
        ) + self.quantiles.values.reshape(1, -1)
        confidence_intervals = pd.DataFrame(
            data=confidence_intervals, columns=self.quantiles.index
        )
        return confidence_intervals
