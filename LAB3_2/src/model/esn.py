import numpy as np

class ESN:
    """Class of Echo State Network model."""

    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            output_size: int, 
            input_scaling: float = 1, 
            spectral_radius: float = 0.9, 
            leakage_rate: float = 1, 
            sparsity = 0, 
            stateful: bool = True
        ) -> None:
        """
        Constructor of ESN model.

        input_size: Input size 
        hidden_size: Hidden size.
        output_size: Output size.
        input_scaling: Input scaling hyperparam.
        spectral_radius: Spectral radius hyperparam.
        leakage_rate: Leakage rate hyperparam.
        sparsity: Sparsity hyperparam.
        stateful: Stateful hyperparam.
        """
        self.leakage_rate = leakage_rate
        self.w_in = (np.random.rand(input_size, hidden_size) * 2 - 1) * input_scaling
        self.w_hh = np.random.rand(hidden_size, hidden_size) * 2 - 1
        self.w_hh[self.w_hh < sparsity] = 0
        self.w_hh = self.w_hh * spectral_radius / np.max(np.abs(np.linalg.eigvals(self.w_hh)))
        self.bias = np.random.rand(hidden_size, 1) * 2 - 1
        self.w_out = np.random.rand(hidden_size, output_size) * 2 - 1
        self.stateful = stateful
        self.H = None

    def __call__(self, X: np.array, states: list[np.array] = None) -> np.array:
        """
        Call method to compute the forward method.

        X: Input time series.
        states: List of hidden states.

        returns:
            np.array: Output time series.
        """
        return self.forward(X)[0]

    def forward(self, X, states: list[np.array] = None) -> np.array:
        """
        Forward method used to predict the output given the input.

        X: Input time series.
        states: List of hidden states.

        returns:
            np.array: Output values.
        """
        if states is None:
            states = self._compute_hidden(X, self.H)
        out = [state @ self.w_out for state in states]
        return np.concatenate(out), np.concatenate(states)
    
    def _compute_hidden(self, X: np.array, H: np.array = None) ->  np.array:
        """
        Protected method to compute hidden states.

        X: Input time series.
        H: Initial hidden state.

        returns:
            np.array: Hidden states for each time steps.
        """
        if H is None:
            H = np.zeros((X.shape[1], self.w_hh.shape[0]))
        states = []
        for x in X:
            preactivation = x @ self.w_in + H @ self.w_hh.T + self.bias.T
            H = (1 - self.leakage_rate) * H + self.leakage_rate * np.tanh(preactivation)
            states.append(H)
        return states
    
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
        states = self._compute_hidden(X, self.H)
        if self.stateful:
            self.H = states[-1]
        H = np.concatenate(states[transient:])
        Y = Y[transient:]
        self.w_out = np.linalg.pinv(H.T @ H + reg * np.eye(H.shape[1])) @ H.T @ Y
        return states
    
    def __str__(self) -> str:
        return f'ESN(w_in: {self.w_in.shape}, w_hh: {self.w_hh.shape}, bias: {self.bias.shape}, w_out: {self.w_out.shape})'