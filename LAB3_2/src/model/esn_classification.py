import numpy as np
from src.model.esn import ESN

class ESNClassification(ESN):
    """Class of Echo State Network model for classification tasks."""
    
    def _compute_hidden(self, X: np.array, H: np.array = None) ->  np.array:
        """
        Protected method to compute hidden states.

        X: Input time series.
        H: Initial hidden state.

        returns:
            np.array: Last hidden state.
        """
        if H is None:
            H = np.zeros((X.shape[1], self.w_hh.shape[0]))
        for x in X:
            preactivation = x @ self.w_in + H @ self.w_hh.T + self.bias.T
            H = (1 - self.leakage_rate) * H + self.leakage_rate * np.tanh(preactivation)
        return H

    def forward(self, X, states: list[np.array] = None) -> np.array:
        """
        Forward method used to predict the output given the input.

        X: Input time series.
        states: List of hidden states.

        returns:
            np.array: Output values.
        """
        H = self._compute_hidden(X)
        out = H @ self.w_out
        return out, H
    
    def train(
            self, 
            X: np.array, 
            Y: np.array, 
            reg: float = 0.01, 
            transient: int = 100
        ) -> list[np.array]:
        """
        Train funtion able to fit the readout with ridge regression.

        X: Input time series.
        Y: Output values.
        reg: Regularization hyperparam.
        transient: Initial part of time series that is dependent from input that will be discarded.

        returns:
            list[np.array]: List of hidden states.
        """
        H = self._compute_hidden(X)
        self.w_out = np.linalg.pinv(H.T @ H + reg * np.eye(H.shape[1])) @ H.T @ Y
        return H