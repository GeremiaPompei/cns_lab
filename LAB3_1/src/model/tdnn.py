import torch

class TDNN(torch.nn.Module):
    """
    Class of TDNN model.
    """

    def __init__(self, window: int, hidden_size: int, output_size: int, n_layers: int = 1, device: str = 'cpu') -> None:
        """
        TDNN constructor method.

        window: Window of TDNN that is related to the number of previous time steps seen in the past.
        hidden_size: Size of hidden state.
        output_size: Size of output value.
        n_layers: Number of hidden layers. Default this is 1.
        """
        super(TDNN, self).__init__()
        hidden_layers = [torch.nn.Linear(window + 1, hidden_size)]
        for _ in range(n_layers - 1):
            hidden_layers.append(torch.nn.Linear(hidden_size, hidden_size).to(device))
        self.hidden_layers = torch.nn.Sequential(*hidden_layers).to(device)
        self.output_layer = torch.nn.Linear(hidden_size, output_size).to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward function used to the forward phase of pytorch module.

        X: Input data.

        returns:
            torch.Tensor: Output data.
        """
        for hidden_layer in self.hidden_layers:
            preactivation = hidden_layer(X)
            X = torch.nn.functional.tanh(preactivation)
        return self.output_layer(X)