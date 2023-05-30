import numpy as np
from src.model.esn_classification import ESNClassification

class EuSNClassification(ESNClassification):
    """Class of Euler State Network model for classification tasks."""

    def __init__(
            self, 
            input_size: int, 
            hidden_size: int,
            output_size: int, 
            input_scaling: float = 1, 
            spectral_radius: float = 0.9, 
            step_size: float = 0.01, 
            diffusion_coefficient: float = 0.01, 
            sparsity=0, 
            stateful: bool = True
        ) -> None:
        super().__init__(input_size, hidden_size, output_size, input_scaling, spectral_radius, None, sparsity, stateful)
        self.diffusion_coefficient = diffusion_coefficient
        self.step_size = step_size
    
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
            preactivation = x @ self.w_in + H @ (self.w_hh.T - self.w_hh - self.diffusion_coefficient * np.eye(self.w_hh.shape[0])) + self.bias.T
            H = H + self.step_size * np.tanh(preactivation)
        return H