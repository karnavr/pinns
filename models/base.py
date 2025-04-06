import torch
import torch.nn as nn


## MAIN PINN CLASS
class PINN(nn.Module):
    """
    General PINN class for two inputs (x,t) and one output (u).
    """

    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation = nn.Tanh):

        super(PINN, self).__init__()

        # setup network parameters / activation 
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation

        # input (x, t) and output (u)
        self.n_input_nodes = 2
        self.n_output_nodes = 1

        # network layers
        layer_list = [nn.Linear(self.n_input_nodes, n_neurons_per_layer)]
        layer_list.append(activation())

        for _ in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            layer_list.append(activation())

        layer_list.append(nn.Linear(n_neurons_per_layer, self.n_output_nodes))

        # assign layers and initialize weights
        self.model = nn.Sequential(*layer_list)
        self._initialize_weights()

    # forward pass method         
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1) # Combine inputs into a single tensor
        output = self.model(inputs) # Pass through network layers
        return output
    
    # initialize weights (xavier normal)
    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight) # Xavier normal initialization
                nn.init.zeros_(layer.bias) # Zero initialization for biases
 


## PINN1D CLASS (for 1D problems with single input/output)
class PINN1D(nn.Module):
    """
    A class for 1D PINNs with a single input and output.
    """
    def __init__(self, n_hidden_layers=2, n_neurons_per_layer=9, activation=nn.Tanh):
        super(PINN1D, self).__init__()

        # setup network parameters / activation 
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons_per_layer = n_neurons_per_layer
        self.activation = activation

        # input (t) and output (x)
        self.n_input_nodes = 1
        self.n_output_nodes = 1

        # network layers
        layer_list = [nn.Linear(self.n_input_nodes, n_neurons_per_layer)]
        layer_list.append(activation())

        for _ in range(n_hidden_layers):
            layer_list.append(nn.Linear(n_neurons_per_layer, n_neurons_per_layer))
            layer_list.append(activation())

        layer_list.append(nn.Linear(n_neurons_per_layer, self.n_output_nodes))

        # assign layers and initialize weights
        self.model = nn.Sequential(*layer_list)
        self._initialize_weights()

    # forward pass function         
    def forward(self, t):
        output = self.model(t)
        return output
    
    # initialize weights function (xavier normal)
    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # Xavier normal initialization
                nn.init.zeros_(layer.bias)  # Zero initialization for biases

