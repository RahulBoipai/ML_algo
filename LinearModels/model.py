import numpy as np


class LinearModel:
    '''
    Linear model class.
    '''
    def __init__(self, inp_dim: int, out_dim: int = 1) -> None:
        '''
        Args:
            inp_dim: input dimension
            out_dim: output dimension
        
        Attributes:
            W: weight matrix
        '''
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.W = np.random.randn(inp_dim, out_dim) * (1 / np.sqrt(inp_dim))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.

        Args:
            x: input data

        Returns:
            scalar output of linear model
        '''
        return x @ self.W

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Forward pass.
        '''
        return self.forward(x)
    
    def __repr__(self) -> str:
        '''
        Representation of model.
        '''
        return f'LinearModel({self.inp_dim}, {self.out_dim})'
