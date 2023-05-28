import torch

class RNN(torch.nn.Module):
    """
    Class of RNN model.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            stateful: bool = True,
            n_layers: int = 1,
            device: str = 'cpu',
    ) -> None:
        """
        RNN constructor method.

        input_size: Size of input value.
        hidden_size: Size of hidden state.
        output_size: Size of output value.
        stateful: Boolean set to true if it's want to use the final training hidden state as initial hidden state of evaluation.
        n_layers: Number of hidden layers. Default this is 1.
        device: Name of device to use for computation.
        """
        super(RNN, self).__init__()
        self.recoursive_layer = torch.nn.RNN(input_size, hidden_size, num_layers=n_layers).to(device)
        self.output_layer = torch.nn.Linear(hidden_size, output_size).to(device)
        self.stateful = stateful
        self.h = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward function used to the forward phase of pytorch module.

        X: Input data.

        returns:
            torch.Tensor: Output data.
        """
        state, h = self.recoursive_layer(X, self.h)
        if self.stateful and self.training:
            self.h = h.detach()
        return self.output_layer(state)