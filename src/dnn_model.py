"""
dnn_model.py — Deep Neural Network Architecture for Battery Health Prediction
==============================================================================

What this module does:
    Defines the pure feed-forward Deep Neural Network (DNN) architecture
    used to predict future battery degradation (capacity / SOH) from
    historical memory windows.

Why pure feed-forward DNN?
    - Many existing papers employ heavy recurrent networks (RNN, LSTM, GRU)
      or Transformers, which are computationally expensive, complex to deploy
      on microcontroller BMS (Battery Management Systems), and prone to overfitting
      on small battery datasets.
    - Our research investigates whether an *Adaptive Memory-Window* input combined
      with a lightweight, fully connected DNN can achieve competitive or superior
      prediction accuracy with minimal inference latency.

Architecture:
    Input (W x 12 features)
      -> Linear(input_dim -> 128) + ReLU
      -> Dropout(0.2)
      -> Linear(128 -> 64) + ReLU
      -> Linear(64 -> 32) + ReLU
      -> Linear(32 -> 1) -> Output (continuous capacity / SOH prediction)
"""

import torch
import torch.nn as nn


class BatteryDNN(nn.Module):
    """
    Feed-Forward Deep Neural Network for Battery Degradation Prediction.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (W * 12, where W is the memory window size).
    hidden_dims : list of int, default [128, 64, 32]
        Number of neurons in each hidden dense layer.
    dropout_rate : float, default 0.2
        Dropout probability applied after the first dense layer.
    """
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.2):
        super(BatteryDNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        
        layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(self.hidden_dims):
            linear_layer = nn.Linear(prev_dim, h_dim)
            # He / Kaiming normal initialization for ReLU activations
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
            nn.init.zeros_(linear_layer.bias)
            
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            
            # Apply dropout after first hidden layer for regularization
            if i == 0 and dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
                
            prev_dim = h_dim
            
        # Final output layer: 1 continuous scalar (predicted capacity or SOH)
        output_layer = nn.Linear(prev_dim, 1)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, input_dim)
        
        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
        """
        return self.network(x)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.2):
    """
    Convenience factory function to instantiate BatteryDNN.
    
    Parameters
    ----------
    input_dim : int
    hidden_dims : tuple of int
    dropout_rate : float
    
    Returns
    -------
    BatteryDNN model instance
    """
    model = BatteryDNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    return model


if __name__ == '__main__':
    # Architecture verification for each memory window
    print("=" * 65)
    print("BATTERY DNN ARCHITECTURE VERIFICATION")
    print("=" * 65)
    
    for w in [3, 5, 10, 20]:
        dim = w * 12
        model = build_model(input_dim=dim)
        dummy_input = torch.randn(8, dim)
        out = model(dummy_input)
        params = model.count_parameters()
        print(f"Memory-{w:2d} | Input Dim: {dim:3d} | Trainable Params: {params:6d} | Output Shape: {list(out.shape)}")
        
    print("=" * 65)
    print("\nDetailed layer structure for Memory-5 (60 inputs):")
    m5 = build_model(60)
    print(m5)
