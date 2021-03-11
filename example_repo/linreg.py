"""Example module."""

__all__ = ["LinearRegression"]


import numpy as np


class LinearRegression:
    """Linear Regression.

    Uses matrix multiplication to fit a linear model that minimizes the mean
    square error of a linear equation system.

    Examples
    --------
    >>> import numpy as np
    >>> from example_repo import LinearRegression
    >>> rng = np.random.RandomState(0)
    >>> n = 100
    >>> x = rng.uniform(-5, 5, n)
    >>> y = 2*x + 1 + rng.normal(0, 1, n)
    >>> linreg = LinearRegression().fit(x, y)
    >>> linreg.predict(range(5)).T
    array([[1.19061859, 3.18431209, 5.17800559, 7.17169909, 9.1653926 ]])
    """

    def __init__(self):
        self._beta = None

    def _2mat(self, arr):
        # Returns input as matrix with ones in first column
        return np.insert(self._2vec(arr), 0, 1, axis=1)

    def _2vec(self, arr):
        # Returns input as vector
        return np.ravel(arr)[:, None]

    def fit(self, x, y):
        """Fit linear model.

        Parameters
        ----------
        x : array-like
            Features
        y : array-like
            Labels

        Returns
        -------
        same type as caller
        """
        X = self._2mat(x)
        Y = self._2vec(y)
        self._beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        return self

    def predict(self, x):
        """Predicts using the linear model.

        Parameters
        ----------
        x : array-like
            Samples

        Returns
        -------
        numpy.ndarray
            Predicted values

        See Also
        --------
        fit

        Raises
        ------
        Exception
            If the model has not been fitted
        """
        if self._beta is None:
            raise Exception("Fit the model first.")
        return self._2mat(x).dot(self._beta)
