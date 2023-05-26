from sklearn.base import clone
from sklearn.utils import resample
import numpy as np

class BootstrapModel:
    """
    A wrapper class for bootstrapping any estimator that conforms to scikit-learn's interface.

    The class supports two types of bootstrapping:
    1. Resampling with replacement: given a proportion of the original data, it samples with replacement
       and retrains the model. The prediction is the output of each of the bootstrap models.
    2. Monte-Carlo bootstrapping: perturbs the training data by adding noise from a specified distribution, 
       and retrains the model. The prediction is the output of each of the bootstrap models.

    Parameters
    ----------
    estimator : object
        The estimator to bootstrap. Should have fit and predict methods.
    n_boots : int
        Number of bootstrap models to train.
    sample_proportion : float, optional
        The proportion of data to sample with replacement in the first type of bootstrapping.
    noise_distribution : scipy.stats distribution, optional
        The distribution to sample noise from in the second type of bootstrapping.

    Attributes
    ----------
    models : list
        List of trained bootstrap models.
    """

    def __init__(self, estimator, n_boots, sample_proportion=None, noise_distribution=None):
        self.estimator = estimator
        self.n_boots = n_boots
        self.sample_proportion = sample_proportion
        self.noise_distribution = noise_distribution
        self.models = []

    def fit(self, X, y):
            """
            Fits the bootstrap models to the data.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                The input samples.
            y : array-like of shape (n_samples,)
                The target values.
            """

            self.models = []
            if self.sample_proportion is not None:
                # Bootstrapping type 1
                n_samples = int(X.shape[0] * self.sample_proportion)
                for _ in range(self.n_boots):
                    X_resample, y_resample = resample(X, y, n_samples=n_samples)
                    model = type(self.estimator)()  # Create a new instance of the same type as estimator
                    model.fit(X_resample, y_resample)
                    self.models.append(model)
            elif self.noise_distribution is not None:
                # Bootstrapping type 2
                for _ in range(self.n_boots):
                    X_noisy = X + self.noise_distribution.rvs(size=X.shape)
                    model = type(self.estimator)()  # Create a new instance of the same type as estimator
                    model.fit(X_noisy, y)
                    self.models.append(model)

    def predict(self, X):
        """
        Predict using the bootstrap models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        predictions : list of array-like
            The predictions of each bootstrap model.
        """

        predictions = []
        for model in self.models:
            predictions.append(model.predict(X))
        return predictions
